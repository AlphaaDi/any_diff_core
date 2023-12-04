import gradio as gr
import os
from time import time
import numpy as np
import uuid
from pathlib import Path
import shutil
import pandas as pd
from time import sleep
import json

from common import *
from sql_handler import *

initial_state = {
    'current_selection': [0, 0, 0, 0],
    'completed_sections': []
}


csv_file_path = '/home/ishpuntov/code/sam-hq/extended_propmts.csv'
styles_df = pd.read_csv(csv_file_path)

blob_storage = Path('/home/ishpuntov/code/sam-hq/blob_storage')

task_db = TaskDatabase('~/task_database.db')


def gradio_get_frame(video_path):
    frame = get_frame(video_path, 0)
    return add_grid(frame)


animatediff_default_config = ''.join(open('animatediff_default_config.json', 'r').readlines())


colormap = {
    "Object_1": "#FF0000",  # Red
    "Object_2": "#00FF00",  # Green
    "Object_3": "#0000FF",  # Blue
    "Object_4": "#FFFF00",  # Yellow
    "Object_5": "#FF00FF",  # Magenta
    "Object_6": "#00FFFF",  # Cyan
    "Object_7": "#800000",  # Maroon
    "Object_8": "#808000",  # Olive
    "Object_9": "#008080",  # Teal
    "Object_10": "#800080",  # Purple
    "Wait_second_point": "#000000",  # Black
}

def remove_last_box(img, state):
    if state['completed_sections']:
        state['completed_sections'].pop()
    return (img, state['completed_sections']), state


def get_select_coordinates(img, state, evt: gr.SelectData):
    current_selection = state['current_selection']
    completed_sections = state['completed_sections']
    temp_section = []
    # If this is the first click of a new selection
    if current_selection == [0, 0, 0, 0]:
        current_selection[0], current_selection[1] = evt.index[0], evt.index[1]
    else:
        # Second click completes the selection
        current_selection[2], current_selection[3] = evt.index[0], evt.index[1]

        # Determine the Object_number based on the number of completed sections
        object_number = len(completed_sections) % 10 + 1
        object_label = f"Object_{object_number}"

        # Add the completed selection to the list of sections
        x_start, y_start = current_selection[:2]
        x_end, y_end = current_selection[2:]
        completed_sections.append(((x_start, y_start, x_end, y_end), object_label))

        # Reset current selection
        current_selection = [0, 0, 0, 0]

    # Draw all completed sections
    # If current selection is in progress, add a temporary section
    if current_selection != [0, 0, 0, 0]:
        point_width = int(img.shape[0] * 0.05)
        temp_section = ((current_selection[0], current_selection[1],
                         current_selection[0] + point_width, 
                         current_selection[1] + point_width),
                        "Wait_second_point")
        temp_section = [temp_section]

    sections = completed_sections + temp_section
    
    state['current_selection'] = current_selection
    state['completed_sections'] = completed_sections
    
    return (img, sections), state


def append_style_prompt(selected_style, current_text):
    style_row = styles_df[styles_df['name'] == selected_style]
    pos_prompt = style_row['detailed_prompt'].iloc[0]
    neg_prompt = style_row['negative_prompt'].iloc[0]
    objects = parse_text_prompts(current_text)
    for key,val in objects.items():
        if not val:
            objects[key] = {}
            objects[key]['pos_prompt'] = pos_prompt
            objects[key]['neg_prompt'] = neg_prompt
            break
    updated_text = json.dumps(objects, indent=4)
    return updated_text

def parse_text_prompts(current_text):
    # Parse the JSON string into a Python dictionary
    try:
        objects = json.loads(current_text)
    except json.JSONDecodeError:
        # Handle the case where the string is not valid JSON
        objects = {}
    return objects

def update_textbox_content(current_text, state):
    objects = parse_text_prompts(current_text)
    current_annot_objects = set([label for _, label in state['completed_sections']])
    remove_objs = set(objects) - current_annot_objects
    add_objs = current_annot_objects - set(objects)

    for obj_key in remove_objs:
        objects.pop(obj_key, None)

    for obj_key in add_objs:
        objects[obj_key] = ""

    # Reconstruct the textbox content as JSON string
    updated_text = json.dumps(objects, indent=4)
    return updated_text


def commit_task(video_path, prompts, config_text_box, state):
    task_id = str(uuid.uuid4())
    
    task_folder = blob_storage / task_id
    task_folder.mkdir(exist_ok=True)
    storage_video_path = str(task_folder / f'video.mp4')
    storage_json_path = str(task_folder / f'config.json')
    shutil.copyfile(video_path, storage_video_path)
    objects_prompts = parse_text_prompts(prompts)
    
    with open(storage_json_path, 'w') as file:
        json.dump(
            json.loads(config_text_box), file, indent=4
        )
    
    objects_info = {}
    for bbox, object_label in state['completed_sections']:
        objects_info[object_label] = {
            'prompt': objects_prompts[object_label],
            'bbox': bbox
        }
    
    task_id = task_db.insert_task(
        objects_info, original_video_path=storage_video_path,
        task_id=task_id, config_path=storage_json_path
    )
    return f'Push task to process and task id is {task_id}'


# Function to load and display video
def load_video(video_path):
    return gr.Video(file_path=video_path)


def erase_task(task_id):
    task_db.remove_task(task_id)
    print(f'Delete {task_id}')


def parse_task_to_gradio(task):
    first_img = get_frame(task["original_video_path"], 0)  
    objects_info = task['objects']
    bboxes = [
        (object_info['bbox'], label) 
        for label, object_info in objects_info.items()
    ]
    prompts = [f"{label}:{object_info['prompt']}" for label, object_info in objects_info.items()]
    prompts_str = '\n'.join(prompts)
    
    return str(task["task_id"]), task["timestamp"], task["status"], prompts_str, (first_img, bboxes)


def get_task_parser_n(idx):
    def parse_task_to_gradio_n():
        tasks = task_db.retrieve_all_task()
        if idx < len(tasks):
            return parse_task_to_gradio(tasks[idx])
        else:
            return tuple([None]*5)
    return parse_task_to_gradio_n


def get_visability(idx):
    def vidability_n():
        tasks = task_db.retrieve_all_task()
        if idx < len(tasks):
            return gr.Row.update(visible=True)
        else:
            return gr.Row.update(visible=False)
    return vidability_n    


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Tabs():
        with gr.Tab("Create task"):
            with gr.Row():
                video = gr.Video(label="Source", width=400,height=400)
            
            with gr.Row():
                masks_canva = gr.Image(label="Mask placer", height=1000,width=1000)
                annotate_img = gr.AnnotatedImage(label="ROI", color_map=colormap,height=1000,width=1000)

            state = gr.State(initial_state)

            with gr.Row():
                with gr.Column():
                    prompt_box = gr.Textbox(
                        label='Object_prompt', value='{}', lines=10
                    )
                with gr.Column():
                    style_dropdown = gr.Dropdown(label='Select Style', choices=styles_df['name'].tolist())
                    remove_button = gr.Button("Remove Last Box")
                    process = gr.Button("Commit task to process")
            with gr.Row():
                config_text_box = gr.Textbox(
                    label='Config_animatediff', value=animatediff_default_config, lines=30, 
                    interactive=True,
                )
                result_text_box = gr.Textbox(label='Result log', lines=1)
                
        with gr.Tab("View results"):
            reload_button = gr.Button("Reload Tasks")
            with gr.Column() as tasks_container:
                rows = []
                for idx in range(20):
                    blocks_row = []
                    with gr.Row(variant='panel') as row:
                        task_id_box = gr.Textbox(label='task_id', lines=5)
                        ts = gr.Textbox(label='timestamp', lines=5,)
                        stat = gr.Textbox(label='status', lines=5,)
                        prompts_box = gr.Textbox(label='prompts', lines=5, min_width=500)
                        video_place = gr.Video(show_label=None)
                        annotate_img_2 = gr.AnnotatedImage(
                            show_label=None,
                            color_map=colormap,height=200,width=200
                        )
                        with gr.Column(variant='panel', min_width=20):
                            btn = gr.Button(f"Load Video")
                            delete_btn = gr.Button("Erase task")

                        blocks_row = [task_id_box, ts, stat, prompts_box, annotate_img_2]
                        
                        reload_button.click(
                            get_visability(idx),
                            outputs=row
                        )
                        
                        reload_button.click(
                            get_task_parser_n(idx),
                            outputs=blocks_row
                        )
                        
                        btn.click(fn=load_video, inputs=task_id_box, outputs=video_place)
                        delete_btn.click(fn=erase_task, inputs=task_id_box)
                
    
    masks_canva_select_event = masks_canva.select(
        fn=get_select_coordinates, 
        inputs=[masks_canva, state], 
        outputs=[annotate_img, state],
    )
    
    masks_canva_select_event.then(
        fn=update_textbox_content, 
        inputs=[prompt_box, state], 
        outputs=prompt_box,
    )
    
    remove_button_click_event = remove_button.click(
        fn=remove_last_box, 
        inputs=[masks_canva, state],
        outputs=[annotate_img, state],
    )
    remove_button_click_event.then(
        fn=update_textbox_content, 
        inputs=[prompt_box, state], 
        outputs=prompt_box,
    )

    style_dropdown.change(
        fn=append_style_prompt,
        inputs=[style_dropdown, prompt_box],
        outputs=prompt_box
    )

    video.change(
        gradio_get_frame,
        [video], masks_canva
    )
            
    process.click(
        fn=commit_task,
        inputs=[video, prompt_box, config_text_box, state],
        outputs=[result_text_box],
    )

demo.launch(server_port=7860)
