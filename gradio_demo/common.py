import cv2
import os
import numpy as np
import json
from scipy import ndimage
from uuid import uuid1
import subprocess
from pathlib import Path

from scipy.ndimage import zoom
from skimage.measure import label

dcn = lambda x: x.detach().cpu().numpy()


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


def wrap_logo(source, logo, pts_src):
    # Points in logo image
    w, h, _ = logo.shape
    pts_logo = np.array([[0, 0], [h-1, 0], [h-1, w-1], [0, w-1]])
    
    # Compute perspective transform
    h, status = cv2.findHomography(pts_logo, pts_src)

    # Warp logo image
    logo_warped = cv2.warpPerspective(logo, h, (source.shape[1], source.shape[0]))
    return logo_warped


def place_logo(source, logo, pts_src):
    # Points in logo image
    w, h, _ = logo.shape
    pts_logo = np.array([[0, 0], [h-1, 0], [h-1, w-1], [0, w-1]])
    
    # Compute perspective transform
    h, status = cv2.findHomography(pts_logo, pts_src)

    # Warp logo image
    logo_warped = cv2.warpPerspective(logo, h, (source.shape[1], source.shape[0]))
    
    logo_warped_bgr = logo_warped[:, :, :3]
    logo_warped_mask = logo_warped[:, :, 3]

    logo_masked = cv2.bitwise_and(logo_warped_bgr, logo_warped_bgr, mask=logo_warped_mask)
    source_masked = cv2.bitwise_and(source, source, mask=cv2.bitwise_not(logo_warped_mask))
    result = cv2.add(logo_masked, source_masked)
    return result


def get_frame(video_path, frame_number):
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, image = video.read()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    video.release()
    return image


def frame_extraction(video_path):
    frames = []
    vid = cv2.VideoCapture(video_path)
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        frames.append(frame)
        flag, frame = vid.read()
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    return frames


def frame_extraction_rgba(video_path):
    frames = []
    vid = cv2.VideoCapture(video_path)
    flag,frame = vid.read()
    while flag:
        frames.append(frame)
        flag, frame = vid.read()
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) for frame in frames]
    return frames


def find_cluster_centers(mask):
    # Label the clusters in the mask
    labeled_mask, num_clusters = ndimage.label(mask)

    # Find the center of each cluster
    centers = ndimage.center_of_mass(mask, labeled_mask, range(1, num_clusters+1))

    # Convert to tuples of integers
    centers = [tuple(map(int, center)) for center in centers]

    centers = [center[::-1] for center in centers]

    return centers


def delete_border(logo):
    logo_bgr = logo[..., :3]
    logo_alpha = logo[..., 3]

    # Perform edge detection on the alpha channel
    edges = cv2.Canny(logo_alpha, threshold1=0, threshold2=1)

    # Dilate the edges to make them thicker
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel)

    # Set the color and alpha values of the border pixels to zero
    logo_bgr[edges != 0] = 0
    logo_alpha[edges != 0] = 0

    # Merge the color and alpha channels back together
    logo_cleaned = cv2.merge([logo_bgr, logo_alpha])
    return logo_cleaned


def order_points(pts):
    # Initialize a list of coordinates that will be ordered such that the first
    # entry in the list is the top-left, the second entry is the top-right,
    # the third is the bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas the bottom-right
    # point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Compute the difference between the points -- the top-right point will have
    # the smallest difference and the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect


def resize_image(image, max_size):

    # Get original image aspect ratio
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    # Determine new image dimensions
    if original_width > original_height:
        new_width = max_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = max_size
        width = int(new_height * aspect_ratio)

    # Check if new sizes are bigger than original, then don't resize
    if new_height > original_height or new_width > original_width:
        print('The image is smaller than the max size.')
        return image

    # Resize the image with new dimensions
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image


def get_video_length(video_path):
    video = cv2.VideoCapture(video_path)
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return length


def convex_hull_mask(image, points):
    # Convert the points to a suitable format
    points = np.array(points, dtype=np.int32)
    points = points.reshape((-1, 1, 2))

    # Create an empty mask of the same size as the input image
    mask = np.zeros_like(image, dtype=np.uint8)

    # Find the convex hull of the points
    hull = cv2.convexHull(points)

    # Draw the filled convex hull on the mask
    cv2.fillConvexPoly(mask, hull, 255)
    
    mask = mask[:,:,0] > 0
    return mask.astype(np.uint8)


def write_data_to_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)



def read_data_from_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def change_fps(input_path: str, output_path: str, new_fps: int) -> None:    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
    
    # Read and write each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()


def add_grid(image, cells_count=30, color=(0, 0, 255)):
    grid_image = image.copy()
    rows, cols, _ = image.shape
    
    step_size = int(min(rows, cols) // cells_count)
    
    # Drawing horizontal lines
    for i in range(0, rows, step_size):
        grid_image[i:i+1, :, :] = color
    
    # Drawing vertical lines
    for j in range(0, cols, step_size):
        grid_image[:, j:j+1, :] = color
    
    return grid_image


def get_video_fps(video_path: str) -> int:
    """
    Get the FPS (Frames Per Second) of a video.

    Parameters:
    - video_path (str): The path to the video file.

    Returns:
    - float: The FPS of the video.
    """
    # Initialize VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Release VideoCapture object
    cap.release()

    return int(fps)


def append_underscore_to_video_name(video_path: str) -> str:
    video_path_obj = Path(video_path)
    video_name_without_extension = video_path_obj.stem
    video_extension = video_path_obj.suffix
    new_video_name_without_extension = video_name_without_extension + "_"
    new_video_name_with_extension = new_video_name_without_extension + video_extension
    new_video_path = video_path_obj.parent / new_video_name_with_extension
    return str(new_video_path)


def map_points_to_original(points_array, canvas_size, original_size):
    x_canvas, y_canvas = canvas_size
    w_original, h_original = original_size

    scales = np.array([h_original / y_canvas, w_original / x_canvas])
    print('scales',scales)
    # Element-wise multiplication
    original_points = points_array * scales

    return original_points


def get_color_masks(masks, colors=color_palette):
    colored_overlay = np.zeros((*masks[0].shape, 3), np.uint8)
    for mask, color in zip(masks, colors):
        # Ensure mask is binary
        mask = (mask > 0).astype(np.uint8)
        colored_overlay[mask > 0] = color  # Assign color to the masked region

    return colored_overlay

def blend_frames_with_colored_masks(frames, colored_overlaies, alpha=0.5):
    blended_frames = []

    for frame, colored_overlay in zip(frames, colored_overlaies):
        blended = cv2.addWeighted(frame, 1.0, colored_overlay, alpha, 0.0)

        blended_frames.append(blended)

    return blended_frames



def resize_binary_mask(mask, new_shape):
    resized_mask = cv2.resize(
        mask.astype(np.uint8), new_shape[::-1], 
        interpolation=cv2.INTER_NEAREST
    )
    return resized_mask > 0.5


def merge_binary_masks(masks):
    return np.logical_or.reduce(masks).astype(bool)


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


def extract_line_endpoints(binary_mask):
    labeled_mask = label(binary_mask)
    num_lines = labeled_mask.max()

    line_masks = []

    for i in range(1, num_lines + 1):
        line_mask = labeled_mask == i
        line_masks.append(line_mask)

    return line_masks