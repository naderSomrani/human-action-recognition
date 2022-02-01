from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename

import _init_paths
import time
import argparse
import os.path as osp
import os

import cv2
from pose_estimation import get_pose_estimator
from tracker import get_tracker
from classifier import get_classifier
from utils.config import Config
from utils.video import Video
from utils.drawer import Drawer

from utils import utils

UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'mp4', 'avi'}
ALLOWED_TASKS = {'pose', 'track', 'action'}

app = Flask(__name__)
app.secret_key = "super secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_suffix(task , cfg):
    suffix = []
    suffix.append(cfg.POSE.name)
    if task != 'pose':
        suffix.extend([cfg.TRACKER.name, cfg.TRACKER.dataset_name, cfg.TRACKER.reid_name])
        if task == 'action':
            suffix.extend([cfg.CLASSIFIER.name, 'torch'])
    return suffix

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/action", methods=['POST', 'GET'])
def person_action_recognition_api():
    task = request.form.get("task") if request.form.get("task") else 'action'
    draw_kp_numbers = request.form.get("draw_kp_numbers") if request.form.get("draw_kp_numbers") else False
    debug_track = request.form.get("debug_track") if request.form.get("debug_track") else True
    save_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'output')

    # get config file
    if 'config' not in request.files:
        config = '../configs/infer_trtpose_deepsort_dnn.yaml'
    else:
        file = request.files['config']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and file.filename.rsplit('.', 1)[1].lower() != 'yaml':
            filename = secure_filename(file.filename)
            config = os.path.join(app.config['UPLOAD_FOLDER'], 'configs', filename)
            file.save(config)
        else:
            flash('File type error')
            return redirect(request.url)

    cfg = Config(config)
    pose_kwargs = cfg.POSE
    clf_kwargs = cfg.CLASSIFIER
    tracker_kwargs = cfg.TRACKER

    # get input video
    if 'input' not in request.files:
        flash('No video input')
        return redirect(request.url)
    file = request.files['input']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        source = os.path.join(app.config['UPLOAD_FOLDER'], 'input', filename)
        file.save(source)
    else:
        flash('File type error')
        return redirect(request.url)

    video = Video(source)

    ## Initiate trtpose, deepsort and action classifier
    pose_estimator = get_pose_estimator(**pose_kwargs)
    if task != 'pose':
        tracker = get_tracker(**tracker_kwargs)
        if task == 'action':
            action_classifier = get_classifier(**clf_kwargs)

    ## initiate drawer and text for visualization
    drawer = Drawer(draw_numbers=draw_kp_numbers)
    user_text = {
        'text_color': 'green',
        'add_blank': True,
        'Mode': task,
        # MaxDist: cfg.TRACKER.max_dist,
        # MaxIoU: cfg.TRACKER.max_iou_distance,
    }

    # loop over the video frames
    for bgr_frame in video:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        # predict pose estimation
        start_pose = time.time()
        predictions = pose_estimator.predict(rgb_frame, get_bbox=True) # return predictions which include keypoints in trtpose order, bboxes (x,y,w,h)
        # if no keypoints, update tracker's memory and it's age
        if len(predictions) == 0 and task != 'pose':
            debug_img = bgr_frame
            tracker.increment_ages()
        else:
            # draw keypoints only if task is 'pose'
            if task != 'pose':
                # Tracking
                # start_track = time.time()
                predictions = utils.convert_to_openpose_skeletons(predictions)
                predictions, debug_img = tracker.predict(rgb_frame, predictions,
                                                                debug=debug_track)
                # end_track = time.time() - start_track

                # Action Recognition
                if len(predictions) > 0 and task == 'action':
                    predictions = action_classifier.classify(predictions)

        end_pipeline = time.time() - start_pose
        # add user's desired text on render image
        user_text.update({
            'Frame': video.frame_cnt,
            'Speed': '{:.1f}ms'.format(end_pipeline*1000),
        })

        # draw predicted results on bgr_img with frame info
        render_image = drawer.render_frame(bgr_frame, predictions, task, **user_text)

        if video.frame_cnt == 1 and save_folder:
            # initiate writer for saving rendered video.
            output_suffix = get_suffix(task, cfg)
            output_path = video.get_output_file_path(
                save_folder, suffix=output_suffix)
            writer = video.get_writer(render_image, output_path, fps=30)

            if debug_track and task != 'pose':
                debug_output_path = output_path[:-4] + '_debug.avi'
                debug_writer = video.get_writer(debug_img, debug_output_path)
            print(f'[INFO] Saving video to : {output_path}')
        # show frames
        try:
            if debug_track and task != 'pose':
                debug_writer.write(debug_img)
                # utils.show(debug_img, window='debug_tracking')
            if save_folder:
                writer.write(render_image)
            # utils.show(render_image, window='webcam' if isinstance(source, int) else osp.basename(source))
        except StopIteration:
            break
    if debug_track and task != 'pose':
        debug_writer.release()
    if save_folder and len(predictions) > 0:
        writer.release()
    video.stop()
    return jsonify({
        'url': 'http://' + request.host + '/' + output_path
    })

@app.route('/')
def index():
    return render_template('action.html')