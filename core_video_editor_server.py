import os
import time
import argparse

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


def main(args):
    video_editor = GeneralCoreVideoEditor(
        device=args.device,
        device_additional=args.device_additional,
        propainter_weights=f"{args.artifacts_dir}/propainter_weights",
        sam_checkpoint=f"{args.artifacts_dir}/sam_hq_vit_l.pth",
        pose_detector_path=f"{args.artifacts_dir}/dw_pose/",
        yolo_pretrained_model=args.yolo_pretrained_model,
        neg_prompt_addition=args.neg_prompt_addition,
        budget=args.budget,
        seed=args.seed,
        styles_csv_path=f"{args.artifacts_dir}/styles.csv",
    )

    task_db = TaskDatabase(args.task_db_path)
    process_tasks_continuously(video_editor, task_db)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for General Core Video Editor and Task Database")

    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--device_additional", type=str, default="cuda:2")
    parser.add_argument("--artifacts_dir", type=str, default="/app/weights")
    parser.add_argument("--yolo_pretrained_model", type=str, default="yolov5x6")
    parser.add_argument("--neg_prompt_addition", type=str, default=" ,CyberRealistic_Negative-neg.")
    parser.add_argument("--budget", type=int, default=400*1000)
    parser.add_argument("--seed", type=int, default=34587484827834)
    parser.add_argument("--task_db_path", type=str, default="/home/ishpuntov/code/sam-hq/task_database.db")

    args = parser.parse_args()
    main(args)
