import os
import shutil
import subprocess
from tqdm import tqdm
import cv2
from pathlib import Path
import skvideo.io
import json
import matplotlib.pyplot as plt
from skimage.measure import label
import numpy as np
from PIL import Image
import tempfile
from roop.evaluator import run as roop_run

from propainter.inference_propainter_inline import inference_propainter_inline
from animatediff.cli import generate as animatediff_generate
import torch

# dcn = lambda x: x.detach().cpu().numpy()
# p = plt.imshow
# I = Image.fromarray


def frame_extraction(video_path):
    frames = []
    vid = cv2.VideoCapture(video_path)
    flag, frame = vid.read()
    while flag:
        frames.append(frame)
        flag, frame = vid.read()
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return frames


def read_frames(folder_path, glob='*'):
    frames_pathes = list(Path(folder_path).glob(glob))
    frames_pathes.sort(key=lambda x: int(x.stem))
    frames = [
        cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB) 
        for frame_path in frames_pathes
    ]
    return frames


def get_bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, cmin, rmax, cmax

def expand_bbox(rmin, cmin, rmax, cmax, n, img_shape):
    rmin = max(0, rmin - n)
    rmax = min(img_shape[0], rmax + n)
    cmin = max(0, cmin - n)
    cmax = min(img_shape[1], cmax + n)
    
    return rmin, cmin, rmax, cmax


def blend_frames_with_masks(frames, masks, color=(0, 255, 0), alpha=0.5):
    blended_frames = []

    for frame, mask in zip(frames, masks):
        # Ensure the mask is binary and three-dimensional
        mask = (mask > 0).astype(np.uint8)

        # Create an image to overlay
        overlay = np.full_like(frame, color)
        overlay = overlay.astype(np.uint8)

        # Blend the frame and the overlay using the mask
        blended = cv2.addWeighted(frame, 1.0, cv2.bitwise_and(overlay, overlay, mask=mask), alpha, 0.0)

        # Append the blended frame
        blended_frames.append(blended)

    return blended_frames

def predict_mask(predictor, frame, mask, number_iteration=5, pixels=40):
    predictor.set_image(frame)

    rmin, cmin, rmax, cmax = get_bbox(mask)
    rmin, cmin, rmax, cmax = expand_bbox(rmin, cmin, rmax, cmax, pixels, frame.shape)

    bboxes = np.array([[cmin, rmin, cmax, rmax]])

    masks, conf, low_masks = predictor.predict(box=bboxes)
    for it in range(number_iteration):
        masks, conf, low_masks = predictor.predict(box=bboxes, mask_input=low_masks)

    return masks[0]

def get_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a*b) // gcd(a, b)

def resample_frames(video_src, src_fps, dst_fps):
    # Find least common multiple of the two frame rates
    common_multiple = lcm(src_fps, dst_fps)
    
    repeat_factor = common_multiple // src_fps
    pick_factor = int(common_multiple // dst_fps)
    
    # Repeat each frame according to repeat_factor
    repeated_frames = np.repeat(video_src, repeat_factor, axis=0)
    
    # Pick frames according to pick_factor
    resampled_frames = repeated_frames[::pick_factor]
    
    return resampled_frames


def blend_frames_by_mask(frame1, frame2, mask):
    mask = mask.astype(bool)

    # Ensure mask dimensions match frames
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=-1)

    blended = np.where(mask, frame1, frame2)
    return blended

def process_effect_frames(predictor, effect_frames, masks, number_iteration=1):
    masks_effect = []
    for frame, mask in tqdm(zip(effect_frames, masks), desc='process_effect_frames'):
        out_mask = predict_mask(predictor, frame, mask, number_iteration=number_iteration)
        masks_effect.append(out_mask)
    return masks_effect


def merge_binary_masks(masks):
    return np.logical_or.reduce(masks).astype(int)

def subtraction_binary_masks(masks_from, masks_what):
    org_dtype = masks_from.dtype
    masks_from = masks_from.astype(bool).astype(int)
    masks_what = masks_what.astype(bool).astype(int)
    res = masks_from - masks_what
    return res.astype(org_dtype)


color_palette = np.array([
    [255, 0, 0],      # Red
    [0, 255, 0],      # Green
    [0, 0, 255],      # Blue
    [255, 255, 0],    # Yellow
    [255, 0, 255],    # Magenta
    [0, 255, 255],    # Cyan
    [128, 0, 0],      # Maroon
    [128, 128, 0],    # Olive
    [0, 128, 0],      # Dark Green
    [128, 0, 128],    # Purple
], dtype=np.uint8)

def color_masks(masks, colors = color_palette):
    frame_colored_masks = []
    mask = (mask > 0).astype(np.uint8)
    for mask, color in zip(masks, colors):
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)

        # Create a colored overlay for the mask
        colored_overlay = np.zeros((*mask.shape, 3), np.uint8)
        colored_overlay[mask > 0] = color  # Assign color to the masked region
        frame_colored_masks.append(colored_overlay)

    return frame_colored_masks


def save_frame_masks(frames, masks, output_path):
    output_path = Path(output_path)
    frames_path = output_path / 'frames'
    frames_path.mkdir(exist_ok=True)
    masks_path = output_path / 'masks'
    masks_path.mkdir(exist_ok=True)
    for idx, (frame, mask) in enumerate(zip(frames, masks)):
        idx_str = f"{idx:06}"
        Image.fromarray(frame).save(frames_path / f"{idx_str}.jpg")
        Image.fromarray(mask > 0).save(masks_path / f"{idx_str}.png")

        
def process_mask_with_object_curves(marked_mask):
    marked_mask = marked_mask > 0
    labeled_mask = label(marked_mask)
    labels_count = labeled_mask.max()
    objects_curve_mask = []
    for label_mark in range(labels_count):
        label_mask  = labeled_mask == (label_mark + 1)
        objects_curve_mask.append(label_mask)
    return objects_curve_mask
    
    
def get_mask_track_in_frames(predictor, frames, mask, sam_iters=3):
    objects_masks = []
    for frame in tqdm(frames, desc='mask track in frames'):
        mask = process_effect_frames(predictor, [frame], [mask], sam_iters)[0]
        objects_masks.append(mask)
    return objects_masks
    
    
def get_bbox_track(masks_track, expand_pixels=10):
    bboxes = []
    for mask in masks_track:
        rmin, cmin, rmax, cmax = get_bbox(mask)
        rmin, cmin, rmax, cmax = expand_bbox(rmin, cmin, rmax, cmax, expand_pixels, mask.shape)
        bboxes.append((rmin,rmax,cmin,cmax))
    return bboxes


get_box_size = lambda box: (box[1] - box[0], box[3] - box[2])

    
def get_expanded_box_coords(boxes, image_shape):
    image_width, image_height = image_shape
    boxes_sizes = np.array(
        list(map(
            get_box_size, boxes
        ))
    )
    max_box_size = boxes_sizes.max(0)
    expanded_boxes = []
    for box in boxes:
        width, height = box[1] - box[0], box[3] - box[2]
        width_diff = max_box_size[0] - width
        height_diff = max_box_size[1] - height

        # Calculate the amount of space to add on each side.
        left_expand = width_diff // 2
        right_expand = width_diff // 2
        top_expand = height_diff // 2
        bottom_expand = height_diff // 2

        # Construct the new expanded box.
        new_box = (
            max(box[0] - left_expand, 0),  # x_min
            min(box[1] + right_expand, image_width), # x_max
            max(box[2] - top_expand, 0),   # y_min
            min(box[3] + bottom_expand, image_height) # y_max
        )
        expanded_boxes.append(new_box)

    return np.array(expanded_boxes), max_box_size


def get_crops(bboxes, frames):
    crops = []
    for box, frame in zip(bboxes, frames):
        rmin,rmax,cmin,cmax = box
        crop = frame[rmin:rmax,cmin:cmax]
        crops.append(crop)
    return crops


def get_masked_frames(frames, masks):
    masked_frames = []
    for frame,mask in zip(frames, masks):
        frame_c = frame.copy()
        frame_c[~mask] = [0,0,0]
        masked_frames.append(frame_c)
    return masked_frames


def get_demasked_frames(frames, masks):
    masked_frames = []
    for frame,mask in zip(frames, masks):
        frame_c = frame.copy()
        frame_c[mask] = [0,0,0]
        masked_frames.append(frame_c)
    return masked_frames


def poisson_blending(source, target, mask):
    """
    Blend the source image into the target image using Poisson blending.
    
    :param source: The source image to be blended into the target.
    :param target: The target image.
    :param mask: A binary mask that defines the blending region.
    :return: The blended image.
    """
    # Compute the center of the blending region
    mask = mask.astype(np.uint8) * 255
    mask_indices = np.where(mask > 0)
    center_y = (np.min(mask_indices[0]) + np.max(mask_indices[0])) // 2
    center_x = (np.min(mask_indices[1]) + np.max(mask_indices[1])) // 2
    # Use seamlessClone to perform Poisson blending
    blended = cv2.seamlessClone(source, target, mask, (center_x, center_y), cv2.MIXED_CLONE)
    
    return blended


def resize_frames_by_size(frames, sizes):
    frames_resize = []
    for frame, size in zip(frames, sizes):
        frame_resize = cv2.resize(frame, size[::-1])
        frames_resize.append(frame_resize)
    return frames_resize


def fill_none_with_nearest(arr):
    if not arr or all(v is None for v in arr):
        return arr

    # Forward fill
    last_valid = None
    for i, val in enumerate(arr):
        if val is not None:
            last_valid = val
        elif last_valid is not None:
            arr[i] = last_valid

    # Backward fill if there were None at the beginning
    next_valid = None
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] is not None:
            next_valid = arr[i]
        elif next_valid is not None:
            arr[i] = next_valid

    return arr


def get_face_crop(image, pose_detector):
    munch_points = pose_detector.procces_np_image(image)
    munch_points = np.array(munch_points.faces_dots)
    munch_points = munch_points[munch_points != None]
    
    if len(munch_points) == 0:
        return None
    munch_points = np.array(munch_points.tolist())
    munch_points = munch_points.reshape(-1,2)

    x_min, y_min = munch_points.min(0)
    x_max, y_max = munch_points.max(0)
    expand = (x_max - x_min) // 2

    x_min, y_min, x_max, y_max = expand_bbox(x_min, y_min, x_max, y_max, expand, image.shape)

    return image[x_min:x_max, y_min:y_max]


def get_diagonal_ones_mask_arr(image_shape_wh, boxes):
    mask = np.zeros(image_shape_wh, dtype=np.uint8)
    
    for box in boxes:
        x1, y1, x2, y2 = box
        height = y2 - y1 + 1
        width = x2 - x1 + 1
        
        if height > width:
            for y in range(height):
                x = int(x1 + (width-1) * y / (height-1))
                mask[y1 + y, x] = 1
        else:
            for x in range(width):
                y = int(y1 + (height-1) * x / (width-1))
                mask[y, x1 + x] = 1
                    
    return mask


def get_bbox_mask(image_shape_wh, bbox):
    mask = np.zeros(image_shape_wh, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    mask[y1:y2 + 1, x1:x2 + 1] = 1
    return mask


def get_bboxes_mask_arr(image_shape_wh, bboxes):
    mask_arr = []
    for bbox in bboxes:
        mask = get_bbox_mask(image_shape_wh, bbox)
        mask_arr.append(mask)
    return mask_arr


def clean_directory(path):
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
            
            
def modify_json_prompt_value(json_file, prop_xpath, new_value):
    # Load the JSON data from the file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    curr_prop = data
    for prop in prop_xpath[:-1]:
        curr_prop = curr_prop[prop]
    curr_prop[prop_xpath[-1]] = new_value
    
    # Write the modified data back to the file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)
        
        
def run_in_conda_env(env_name, script_path, length, width, height, device, out_dir, config_path):
    """Run script in specified conda environment."""
    
    # Save the current working directory
    original_directory = os.getcwd()

    # Change the current working directory to the directory of the script
    script_directory = os.path.dirname(script_path)
    os.chdir(script_directory)

    # Run the script in the desired conda environment
    command = f"conda run -n {env_name} python {script_path} --config_path {config_path} --length {length} --width {width} --height {height} --device {device} --out_dir {out_dir}"
    subprocess.run(command, shell=True)
    
    os.chdir(original_directory)
    
    
def clean_and_fill_img_dir(dir_path, images):
    clean_directory(dir_path)
    dir_path = Path(dir_path)
    for idx, img in enumerate(images):
        Image.fromarray(img).save(dir_path / f'{idx:05}.png')


def fill_black_holes(binary_mask):
    """
    Fills black holes in the binary mask, avoiding the largest contour (usually the background).

    Parameters:
    binary_mask (numpy.ndarray): The binary mask where holes are to be filled.

    Returns:
    numpy.ndarray: Binary mask with holes filled, excluding the background.
    """
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) != 0:
        # Assume the largest contour is the background and should be ignored
        largest_contour = max(contours, key=cv2.contourArea)
        for cnt in contours:
            if cnt is not largest_contour:
                cv2.drawContours(binary_mask, [cnt], 0, 255, -1)
    return binary_mask

def keep_largest_contour(binary_mask):
    """
    Keeps only the largest white contour in the binary mask.

    Parameters:
    binary_mask (numpy.ndarray): The binary mask with multiple white contours.

    Returns:
    numpy.ndarray: Binary mask with only the largest contour.
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        new_mask = np.zeros_like(binary_mask)
        cv2.drawContours(new_mask, [largest_contour], -1, 255, -1)
        return new_mask
    else:
        return np.zeros_like(binary_mask)
    
    
def dilate_mask(binary_mask, dilation_size=3):
    """
    Dilates a binary mask by a specified number of pixels.

    Parameters:
    binary_mask (numpy.ndarray): The binary mask to be dilated.
    dilation_size (int): The size of the dilation (number of pixels to dilate).

    Returns:
    numpy.ndarray: Dilated binary mask.
    """
    # Create the dilation kernel
    kernel = np.ones((dilation_size, dilation_size), np.uint8)

    # Dilate the mask
    dilated_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    return dilated_mask


def dilate_mask_arr(binary_mask_arr, dilation_size=3):
    org_dtype = binary_mask_arr[0].dtype
    dilated_mask_arr = []
    for mask in binary_mask_arr:
        dil_mask = dilate_mask(mask.astype(np.uint8), dilation_size=dilation_size)
        dilated_mask_arr.append(dil_mask.astype(org_dtype))
    return dilated_mask_arr


def process_upscale_img(upcales, image_processor, img):
    inputs = image_processor(img, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].to(upcales.device)

    with torch.no_grad():
        outputs = upcales(**inputs)

    output = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.moveaxis(output, source=0, destination=-1)
    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

    return output


def infer_propainter(
        masks, frames, propainter_weights, upscaler, image_processor, device,
        video_name='propainter_frames_tmp.mp4'):
    w_org, h_org = frames[0].shape[:2]
    masks_merge_pil = [Image.fromarray(mask > 0) for mask in masks]
    skvideo.io.vwrite(video_name, frames)
    dir_path = Path(os.getcwd()) / video_name
    inpainted_frames = inference_propainter_inline(
        str(dir_path), masks_merge_pil, device=device,
        model_dir=propainter_weights
    )

    upscaler = upscaler.to(device)
    inpainted_frames_upcale = [
        process_upscale_img(
            upscaler, image_processor, img) for img in tqdm(inpainted_frames, desc='process upscale')
    ]
    upscaler = upscaler.to('cpu')
    torch.cuda.empty_cache()
    
    inpainted_frames = [cv2.resize(
        frame, (h_org, w_org), interpolation = cv2.INTER_AREA) for frame in inpainted_frames_upcale]
    
    
    org_frames_arr = np.arrayy(frames).astype(np.uint8)
    inpainted_frames_np = np.array(inpainted_frames).astype(np.uint8)

    masks_np = np.array(masks).astype(np.uint8)
    mask_np_extend = masks_np[:, :, :, None]
    
    frames_inpainted_area = inpainted_frames_np * mask_np_extend + (1 - mask_np_extend) * org_frames_arr

    return frames_inpainted_area

def process_image_video(image, video_arr):
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_image_path = os.path.join(tmp_dir, 'temp_image.png')
        temp_video_path = os.path.join(tmp_dir, 'temp_video.mp4')
        Image.fromarray(image).save(temp_image_path)
        skvideo.io.vwrite(temp_video_path, video_arr)
        result = roop_run(temp_image_path, temp_video_path)
    return result


def run_animatediff_generation(length, width, height, config_path, device, out_dir, seed):
    torch.cuda.set_device(device)

    config_path = Path(config_path)

    # Generate animation using the provided arguments
    animatediff_generate(
        config_path=config_path,
        length=length,
        width=width,
        height=height,
        out_dir = Path(out_dir),
        device = device
    )
    
    first_child = next(out_dir.glob('*'))
    result_frames_folder = first_child / f'00-{seed}'
    obj_frames = read_frames(result_frames_folder)
    return obj_frames
    
def copy_config_file(original_file_path, new_file_name):
    # Define the path to the original file
    original_path = Path(original_file_path)

    # Get the folder of the original file
    folder = original_path.parent

    # Create the new file path
    new_file_path = folder / new_file_name

    # Copy the content of the original file to the new file
    new_file_path.write_text(original_path.read_text())
    return new_file_path