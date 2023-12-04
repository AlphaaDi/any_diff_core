import os
import time

from gradio_demo.sql_handler import TaskDatabase
from core_video_editor_class import GeneralCoreVideoEditor

def process_task(processor, db, task):
    """
    Process a single task.
    :param db: The database connection.
    :param task: The task to process.
    """
    task_id = task['task_id']
    objects_info = task['objects']
    video_path = task['original_video_path']
    animatediff_config_path = task['config_path']
    output_filename = processor.process_video(video_path, task_id, objects_info, animatediff_config_path)
    raise 1

    db.update_task_status(
        task_id=task_id,
        status='in progress'
    )
    try:
        objects_info = task['objects']
        video_path = task['original_video_path']
        animatediff_config_path = task['config_path']
        output_filename = processor.process_video(video_path, task_id, objects_info, animatediff_config_path)
    except Exception as e:
        print(f'Error during processing task {task_id}: ', e)
        db.update_task_status(
            task_id=task_id,
            status='wait'
        )
        return

    db.update_task_status(
        task_id=task_id,
        status='ready', 
        file_path=str(output_filename)
    )
    print(f"Task {task_id} processed and updated.")


def process_tasks_continuously(processor, task_db, check_interval=10):
    """
    Continuously process tasks from the database.
    :param db: The database connection.
    :param check_interval: Time in seconds to wait before checking for new tasks.
    """
    while True:
        task = task_db.retrieve_oldest_wait_task()
        if task:
            print(f"Found task: {task['task_id']}")
            process_task(processor, task_db, task)
        else:
            print(f"No pending tasks. Checking again in {check_interval} seconds.")
            time.sleep(check_interval)


video_editor = GeneralCoreVideoEditor(
    device='cuda:1',
    device_additional='cuda:2',
    propainter_weights='/home/ishpuntov/code/propainter_lib/weights',
    sam_checkpoint='/home/ishpuntov/code/animatediff-cli-prompt-travel/sam_hq_vit_l.pth',
    pose_detector_path='/home/ishpuntov/code/dw_pose_lib/dw_pose/',
    yolo_pretrained_model='yolov5x6',
    image_processor_pretrained="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    upscaler_pretrained="caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
    neg_prompt_addition= ' ,CyberRealistic_Negative-neg.',
    budget=400 * 1000,
    seed=34587484827834,
    styles_csv_path='/home/ishpuntov/code/sam-hq/styles.csv',
    animatediff_config_path = '/home/ishpuntov/code/animatediff-cli-prompt-travel/config/prompts/prompt_travel_multi_controlnet_org.json'
)

task_db = TaskDatabase('/home/ishpuntov/code/sam-hq/task_database.db')

process_tasks_continuously(video_editor, task_db)
