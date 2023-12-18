import os
import time
import argparse

from flask import Flask, request, jsonify
import threading
import requests
import os
from pathlib import Path
import json
import argparse
from core_video_editor_class import GeneralCoreVideoEditor

app = Flask(__name__)

# Initialize GeneralCoreVideoEditor outside of the request handling function
parser = argparse.ArgumentParser(description="Arguments for General Core Video Editor")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_additional", type=str, default="cuda:1")
parser.add_argument("--artifacts_dir", type=str, default="/app/weights")
parser.add_argument("--yolo_pretrained_model", type=str, default="yolov5x6")
parser.add_argument("--neg_prompt_addition", type=str, default=" ,CyberRealistic_Negative-neg.")
parser.add_argument("--budget", type=int, default=400*1000)
parser.add_argument("--seed", type=int, default=34587484827834)
parser.add_argument("--port", type=int, default=5000)
args = parser.parse_args()

GLOBAL_STATUS = 'ready'

video_editor = GeneralCoreVideoEditor(
    device=args.device,
    device_additional=args.device_additional,
    propainter_weights=os.path.join(args.artifacts_dir, "propainter_weights"),
    sam_checkpoint=os.path.join(args.artifacts_dir, "sam_hq_vit_l.pth"),
    pose_detector_path=os.path.join(args.artifacts_dir, "dw_pose"),
    yolo_pretrained_model=args.yolo_pretrained_model,
    neg_prompt_addition=args.neg_prompt_addition,
    budget=args.budget,
    seed=args.seed,
)

def process_video_and_send(video_path, task_id, objects_info, animatediff_config_path, response_url):
    # Process the video
    processed_video = video_editor.process_video(
        video_path, task_id, objects_info, animatediff_config_path)

    # Prepare data to send, including the task_id
    files = {'video': open(processed_video, 'rb')}
    data = {'task_id': task_id}
    requests.post(response_url, files=files, data=data)

    # Update the global status or any other necessary post-processing
    global GLOBAL_STATUS
    GLOBAL_STATUS = 'ready'
    
@app.route('/process_video', methods=['POST'])
def video_processing_endpoint():
    video_file = request.files['video']
    config_json = request.form.get('config')
    
    if config_json:
        # Parse the JSON string back into a Python dictionary
        config_data = json.loads(config_json)
    else:
        return jsonify({'message': 'Video error config'}), 407
    
    task_id = config_data['task_id']
    objects_info = json.loads(config_data['objects'])
    animatediff_config = json.loads(config_data['animate_config'])

    blob_storage = Path(os.path.join(args.artifacts_dir, "blob_storage"))
    task_folder = blob_storage / task_id
    task_folder.mkdir(exist_ok=True)

    storage_video_path = str(task_folder / f'video.mp4')
    storage_json_path = str(task_folder / f'config.json')
    
    with open(storage_json_path, 'w') as file:
        json.dump(
            animatediff_config, file, indent=4
        )
    video_file.save(storage_video_path)
    response_url = request.form['response_url']
    
    global GLOBAL_STATUS
    GLOBAL_STATUS = f'work'
    sub_proc = threading.Thread(
        target=process_video_and_send, 
        args=(storage_video_path, task_id, objects_info, storage_json_path, response_url))
    sub_proc.start()

    return jsonify({'message': 'Video processing started'}), 202


@app.route('/get_worker_status', methods=['GET'])
def get_worker_status():
    global GLOBAL_STATUS
    return jsonify({'status': GLOBAL_STATUS}), 200

app.run(host='0.0.0.0',debug=True, port=args.port)