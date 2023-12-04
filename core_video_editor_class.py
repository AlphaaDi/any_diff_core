import os
from pathlib import Path

from tqdm import tqdm
import torch
import cv2
import skvideo.io
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shutil
from PIL import Image

from segment_anything_hq import SamPredictor, sam_model_registry
from propainter.inference_propainter_inline import inference_propainter_inline
from dw_pose import DWposeDetectorInference
from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution

from core_utils import *

dcn = lambda x: x.detach().cpu().numpy()
p = plt.imshow
I = Image.fromarray


class GeneralCoreVideoEditor:
    def __init__(
        self, device, device_additional, propainter_weights,
        sam_checkpoint, pose_detector_path, yolo_pretrained_model,
        image_processor_pretrained, upscaler_pretrained,
        neg_prompt_addition, budget, seed, styles_csv_path,
        animatediff_config_path,
        animatediff_path = '/home/ishpuntov/code/animatediff-cli-prompt-travel',
        output_dir_animatediff = 'output_result_for_script_call',
        output_result_folder = '/home/ishpuntov/code/sam-hq/output_results',
        faces_crop_ration = 0.5,
        delate_pix_inpaint_max = 30,
    ):
        torch.cuda.set_device(device)
        self.device = device
        self.device_additional = device_additional

        self.propainter_weights = propainter_weights
        self.sam = sam_model_registry['vit_l'](checkpoint=sam_checkpoint)
        self.sam.to(self.device)
        self.predictor = SamPredictor(self.sam)

        self.pose_detector = DWposeDetectorInference(pose_detector_path, self.device)

        self.yolo_detector_model = torch.hub.load('ultralytics/yolov5', yolo_pretrained_model, pretrained=True)
        self.yolo_detector_model = self.yolo_detector_model.to(self.device)

        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_pretrained)
        self.upscaler = Swin2SRForImageSuperResolution.from_pretrained(upscaler_pretrained)

        self.neg_prompt_addition = neg_prompt_addition
        self.budget = budget
        self.seed = seed

        self.styles_map = pd.read_csv(styles_csv_path)
        self.styles_map.set_index('name', inplace=True)
        
        self.output_folder = Path(output_result_folder)
        self.output_folder.mkdir(exist_ok=True)
        self.animatediff_path = animatediff_path
        
        self.faces_crop_ration = faces_crop_ration
        self.animatediff_config_path = animatediff_config_path
        self.output_dir = output_dir_animatediff
        self.delate_pix_inpaint_max = delate_pix_inpaint_max
        


    def get_class_bboxes_yolo(self, image_pil, class_name='person', treshhold = 0.5):
        results = self.yolo_detector_model([image_pil])
        df = results.pandas().xyxy[0]
        df = df[df['name'] == class_name]
        df = df[df['confidence'] > treshhold]
        bboxes = df[['xmin', 'ymin', 'xmax', 'ymax']].to_numpy()
        return bboxes.astype(int)


    def process_objects_in_video(self, frames, objects_mask, w_org, h_org):
        crops_arr = []
        bbox_track_arr = []
        mask_crops_arr = []
        mask_track_arr = []
        expand_boxes_arr = []
        resize_crops_arr = []
        resize_mask_crops_arr = []
        max_box_size_arr = []

        for object_mask in objects_mask:
            mask_track = get_mask_track_in_frames(self.predictor, frames, object_mask)
            
            mask_track = np.array(mask_track)
            bbox_track = get_bbox_track(mask_track)
            expand_boxes, max_box_size = get_expanded_box_coords(bbox_track, (w_org, h_org))
            crops = get_crops(expand_boxes, frames)
            
            mask_crops = get_crops(expand_boxes, mask_track)
            mask_crops = [crop.astype(np.uint8) for crop in mask_crops]
            resize_mask_crops = [cv2.resize(crop, (max_box_size[1], max_box_size[0]), cv2.INTER_NEAREST) > 0 for crop in mask_crops]
            resize_crops = [cv2.resize(crop, (max_box_size[1], max_box_size[0])) for crop in crops]

            crops_arr.append(crops)
            bbox_track_arr.append(bbox_track)
            mask_crops_arr.append(mask_crops)
            mask_track_arr.append(mask_track)
            expand_boxes_arr.append(expand_boxes)
            resize_crops_arr.append(resize_crops)
            resize_mask_crops_arr.append(resize_mask_crops)
            max_box_size_arr.append(max_box_size)
        
        return (crops_arr, bbox_track_arr, mask_crops_arr, mask_track_arr,
                expand_boxes_arr, resize_crops_arr, resize_mask_crops_arr,
                max_box_size_arr)
    
    
    def infer_upscaler_on_device(self, imgs):
        self.upscaler = self.upscaler.to(self.device_additional)
        upcale_imgs = [
            process_upscale_img(self.upscaler, self.image_processor, img) 
            for img in tqdm(imgs, desc='process upscale')
        ]
        self.upscaler = self.upscaler.to('cpu')
        torch.cuda.empty_cache()
        return upcale_imgs


    def infer_propainter(self, masks, frames, video_name='propainter_frames_tmp.mp4'):
        print('infer_propainter')
        w_org, h_org = frames[0].shape[:2]
        
        masks_merge_pil = [Image.fromarray(mask > 0) for mask in masks]
        skvideo.io.vwrite(video_name, frames)
        dir_path = Path(os.getcwd()) / video_name
        inpainted_frames = inference_propainter_inline(
            str(dir_path), masks_merge_pil, device=self.device,
            model_dir=self.propainter_weights
        )
        
#         inpainted_frames_upcale = self.infer_upscaler_on_device(inpainted_frames)
        inpainted_frames_upcale = inpainted_frames
        
        inpainted_frames = [cv2.resize(
            frame, (h_org, w_org), interpolation = cv2.INTER_AREA) for frame in inpainted_frames_upcale
        ]

        org_frames_arr = np.array(frames).astype(np.uint8)
        inpainted_frames_np = np.array(inpainted_frames).astype(np.uint8)

        masks_np = np.array(masks).astype(np.uint8)
        mask_np_extend = masks_np[:, :, :, None]

        frames_inpainted_area = inpainted_frames_np * mask_np_extend + (1 - mask_np_extend) * org_frames_arr

        return frames_inpainted_area

    
    def merge_other_object_masks(
        self, objects_count, obj_idx, mask_track_arr, expand_boxes, max_box_size):
        resize_mask_crops_merge = []
        for other_obj_idx in range(objects_count):
            if other_obj_idx == obj_idx:
                continue
            mask_track = mask_track_arr[other_obj_idx]
            mask_crops = get_crops(expand_boxes, mask_track)
            mask_crops = [crop.astype(np.uint8) for crop in mask_crops]
            resize_mask_crops = [
                cv2.resize(crop, (max_box_size[1], max_box_size[0]), cv2.INTER_NEAREST) > 0 
                for crop in mask_crops
            ]
            resize_mask_crops = dilate_mask_arr(resize_mask_crops, self.delate_pix_inpaint_max)
            resize_mask_crops_merge.append(resize_mask_crops)
    
        resize_mask_crops_merge = np.array(resize_mask_crops_merge)
        resize_mask_crops_merge = merge_binary_masks(resize_mask_crops_merge)
        return resize_mask_crops_merge


    def process_inpainting(
        self, objects_count, resize_crops_arr, expand_boxes_arr,
        max_box_size_arr, mask_track_arr, resize_mask_crops_arr
    ):
        resize_crops_inpaint_arr = []

        for obj_idx in range(objects_count):
            resize_crops = resize_crops_arr[obj_idx]
            expand_boxes = expand_boxes_arr[obj_idx]
            max_box_size = max_box_size_arr[obj_idx]
            obj_mask_track = np.array(resize_mask_crops_arr[obj_idx])
            
            resize_mask_crops_merge = self.merge_other_object_masks(
                objects_count, obj_idx, mask_track_arr, expand_boxes,
                max_box_size
            )
            
            resize_mask_crops_merge = subtraction_binary_masks(
                resize_mask_crops_merge, obj_mask_track
            )
            
            resize_mask_crops_merge = resize_mask_crops_merge > 0

            resize_crops_inpaint = self.infer_propainter(
                resize_mask_crops_merge, resize_crops
            )
            resize_crops_inpaint_arr.append(resize_crops_inpaint)

        return resize_crops_inpaint_arr


    def crop_faces(self, resize_crops_arr):
        face_crops_arr = []
        for resize_crops in resize_crops_arr:
            face_crops = [get_face_crop(frame, self.pose_detector) for frame in resize_crops]
            none_count = [elem is None for elem in face_crops]
            none_count = sum(none_count)
            none_ration = none_count / len(face_crops)
            if none_ration > self.faces_crop_ration:
                face_crops_arr.append(None)
            else:
                face_crops = fill_none_with_nearest(face_crops)
                face_crops_arr.append(face_crops)
        return face_crops_arr


    def finalize_object_processing(
        self, objects_count, mask_crops_arr, expand_boxes_arr, resize_crops_arr,
        face_crops_arr, prompts, is_person_arr, animatediff_config_path_base, task_id
    ):
        resize_obj_frames_arr = []
        masks_effects_arr = []

        for obj_idx in range(objects_count):
            mask_crops = mask_crops_arr[obj_idx]
            expand_boxes = expand_boxes_arr[obj_idx]
            resize_crops = resize_crops_arr[obj_idx]
            face_crops = face_crops_arr[obj_idx]
            is_person = is_person_arr[obj_idx]
            
            if is_person:
                clean_and_fill_img_dir(
                    os.path.join(self.animatediff_path, 'data/ip_adapter_image/face/'), face_crops
                )
                
            clean_and_fill_img_dir(
                os.path.join(self.animatediff_path, 'data/ip_adapter_image/test/'), resize_crops)
            clean_and_fill_img_dir(
                os.path.join(self.animatediff_path, 'data/controlnet_image/test/controlnet_openpose/'), resize_crops)
            clean_and_fill_img_dir(
                os.path.join(self.animatediff_path, 'data/controlnet_image/test/qr_code_monster_v2/'), resize_crops)
            
            animatediff_config_path = copy_config_file(animatediff_config_path_base, f'config_object_{obj_idx}.json')
            
            pos_prompt = prompts[obj_idx]['pos_prompt']
            neg_prompt = prompts[obj_idx]['neg_prompt']
            
            if is_person:
                pos_prompt = pos_prompt.format(prompt='person')
            else:
                pos_prompt = pos_prompt.format(prompt='object')
                
            neg_prompt_full = neg_prompt
            
            modify_json_prompt_value(
                animatediff_config_path,
                ['ip_adapter_map', 'is_plus_face', 'enable'],
                is_person
            )

            modify_json_prompt_value(
                animatediff_config_path,
                ['prompt_map', '0'],
                pos_prompt
            )

            modify_json_prompt_value(
                animatediff_config_path,
                ['n_prompt'],
                [neg_prompt_full]
            )

            result_local_dir = os.path.join(self.output_dir, f'{task_id}_{obj_idx}')
            result_path_out = Path(os.path.join(self.animatediff_path, result_local_dir))
            result_path_out.mkdir(exist_ok=True)
            clean_directory(str(result_path_out))

            height, width = resize_crops[0].shape[:2]
            current_budget = height * width
            budget_ration = current_budget / self.budget
            height, width = height / budget_ration, width / budget_ration
            height, width = int(height), int(width)
            height, width = height // 8 * 8, width //8 * 8

            # run_animatediff_pipe_path = os.path.join(self.animatediff_path, 'run_animatediff_pipe.py')
            # run_in_conda_env(
            #     'animatediff',
            #     run_animatediff_pipe_path,
            #     length = len(resize_crops), width = width, height = height,
            #     device = self.device_additional,
            #     out_dir=result_path_out,
            #     config_path=animatediff_config_path
            # )
            
            obj_frames = run_animatediff_generation(
                length=len(resize_crops), 
                width=width,
                height=height,
                config_path=animatediff_config_path,
                device=self.device_additional,
                out_dir=result_path_out,
                seed=self.seed,
            )
            
            # obj_frames = self.infer_upscaler_on_device(obj_frames)
            
            obj_frames_face = process_image_video(face_crops[0], obj_frames)
                
            bboxes_size = list(map(get_box_size, expand_boxes))
            resize_obj_frames = resize_frames_by_size(obj_frames_face, bboxes_size)
            masks_effects = process_effect_frames(self.predictor, resize_obj_frames, mask_crops, number_iteration=10)

            resize_obj_frames_arr.append(resize_obj_frames)
            masks_effects_arr.append(masks_effects)
            
        return resize_obj_frames_arr, masks_effects_arr


    def blend_and_save_video(
        self, frames, mask_crops_arr, expand_boxes_arr,
        resize_obj_frames_arr, masks_effects_arr, inpainted_frames,
        output_filename, fps
    ):
        inpainted_frames_cp = np.array(inpainted_frames)
        for obj_idx in range(len(mask_crops_arr)):
            obj_boxes = expand_boxes_arr[obj_idx]
            resize_obj_frames = resize_obj_frames_arr[obj_idx]
            masks_effects = masks_effects_arr[obj_idx]
            mask_crops = mask_crops_arr[obj_idx]

            for idx in tqdm(range(len(frames)), desc='blend frames with affect masks'):
                frame = frames[idx]
                box = obj_boxes[idx]
                obj_frame = resize_obj_frames[idx]
                
                org_mask_crop = mask_crops[idx].astype(np.uint8)
                org_mask_dil= dilate_mask(org_mask_crop, self.delate_pix_inpaint_max) 
                
                obj_mask = masks_effects[idx]
                
                merge_mask = np.logical_and(obj_mask, org_mask_dil)
                
                merge_mask = merge_mask.astype(np.uint8)
                obj_mask_3d = np.repeat(merge_mask[:, :, None], 3, 2)

                canva_frame = np.zeros_like(frame)
                canva_mask = np.zeros_like(frame[:, :, 0])

                canva_frame[box[0]:box[1], box[2]:box[3]] = \
                    canva_frame[box[0]:box[1], box[2]:box[3]] * (1 - obj_mask_3d) + obj_mask_3d * obj_frame

                canva_mask[box[0]:box[1], box[2]:box[3]] += merge_mask

                inpainted_frame = inpainted_frames_cp[idx]
                inpainted_frame[canva_mask > 0] = canva_frame[canva_mask > 0]

        torch.cuda.empty_cache()
        print('save result to ', output_filename)
        
        skvideo.io.vwrite(
            output_filename, inpainted_frames_cp,
            outputdict={'-r':str(fps)}
        )
        return inpainted_frames_cp


    def get_output_path(self, video_path, output_name='result.mp4'):
        video_path = Path(video_path)
        video_dir = video_path.parent.resolve()
        return video_dir / output_name
        
        
    def process_video(self, video_path, task_id, objects_info, animatediff_config_path):
        frames = frame_extraction(video_path)
        w_org, h_org = frames[0].shape[:2]
        fps = get_frame_rate(video_path)
        
        bboxes = [object_info['bbox'] for object_info in objects_info.values()]       
        bboxes_mask_arr = get_bboxes_mask_arr((w_org, h_org), bboxes)
        
        
        prompts = [object_info['prompt'] for object_info in objects_info.values()]

        objects_count = len(bboxes)
        
        (crops_arr, bbox_track_arr, mask_crops_arr, mask_track_arr,
         expand_boxes_arr, resize_crops_arr, resize_mask_crops_arr,
         max_box_size_arr) = self.process_objects_in_video(
            frames, bboxes_mask_arr, w_org, h_org
        )
        
        resize_crops_inpaint_arr = self.process_inpainting(
            objects_count, resize_crops_arr,
            expand_boxes_arr, max_box_size_arr, mask_track_arr, resize_mask_crops_arr
        )
        
        resize_crops_arr = resize_crops_inpaint_arr
        
        self.resize_crops_inpaint_arr = resize_crops_inpaint_arr
        
        face_crops_arr = self.crop_faces(resize_crops_arr)
        
        is_person_arr = [face_crops is not None for face_crops in face_crops_arr]
        
        resize_obj_frames_arr, masks_effects_arr = self.finalize_object_processing(
            objects_count, mask_crops_arr, expand_boxes_arr, resize_crops_arr,
            face_crops_arr, prompts, is_person_arr,
            animatediff_config_path, task_id
        )
        
        masks_merge = np.array(mask_track_arr).sum(0)
        masks_merge_dilate = dilate_mask_arr(masks_merge, self.delate_pix_inpaint_max)
        inpainted_frames = self.infer_propainter(masks_merge_dilate, frames)
        
        output_filename = self.get_output_path(video_path)
        
        self.blend_and_save_video(
            frames, mask_crops_arr, expand_boxes_arr,
            resize_obj_frames_arr, masks_effects_arr, 
            inpainted_frames, 
            output_filename, fps
        )
        
        return output_filename