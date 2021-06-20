import os
import numpy as np
import tensorflow as tf
import h5py
import cv2
import mediapipe as mp
from nanoid import generate
import subprocess as sp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

MLBUCKET = "coachneuro-dev-ml"
APPBUCKET = "coachneuro-dev.appspot.com"


if not firebase_admin._apps:
    cred = credentials.Certificate("./coachneuro-dev-firebase-adminsdk.json")
    firebase_admin.initialize_app(cred)
    mlbucket = storage.bucket(MLBUCKET)
    appbucket = storage.bucket(APPBUCKET)


def download_model(prefix, model_name):
    if not os.path.isdir("tmp/models/"):
        os.makedirs("tmp/models/")

    model_blobs = list(mlbucket.list_blobs(prefix=prefix))

    model_blob = model_blobs[-1] # the most recent model will be the last in the List
    model_version = model_blob.name.split("/")[-2] # the version of the model will be the second to last element in the split array
    
    download_dir_path = os.path.join("tmp", prefix, model_version)
    download_path = os.path.join(download_dir_path, model_name)

    if os.path.isdir(download_dir_path):
        return download_path

    os.makedirs(download_dir_path)

    model_blob.download_to_filename(download_path)

    return download_path


def get_model(prefix, model_name):

    # download the model to the tmp folder and returns the path
    download_path = download_model(prefix, model_name)

    # load model
    model = tf.keras.models.load_model(h5py.File(download_path, "r"))
    
    return model


def download_video(path):

    if not os.path.isdir("tmp/videos/source-videos/"):
        os.makedirs("tmp/videos/source-videos/")

    if not os.path.isdir("tmp/videos/output-videos/"):
        os.makedirs("tmp/videos/output-videos/")

    download_path = "tmp/videos/source-videos/" + os.path.basename(path)

    video_blob = appbucket.blob(path)

    video_blob.download_to_filename(download_path)

    return download_path


def get_poses_from_video(video_download_path):

    poses = []
    images = []

    video_cap = cv2.VideoCapture(video_download_path)
    ret, frame = video_cap.read()
    height, width, ch = frame.shape
    
    output_path = "tmp/videos/output-videos/" + generate() + ".mp4"
    ffmpeg = "ffmpeg"
    dimension = f"{width}x{height}"
    f_format = "yuv420p"
    fps = str(video_cap.get(cv2.CAP_PROP_FPS))

    command = [ffmpeg, "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", dimension, "-pix_fmt", f_format, "-r", fps, "-i", "-", "-an", "-vcodec", "mpeg4", "-b:v", "5000k", output_path]
    proc = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

    with mp_pose.Pose(min_detection_confidence=0.99, min_tracking_confidence=0.99) as pose:
        while video_cap.isOpened():
            success, image = video_cap.read()
            if not success:
                print("Can't receive frame")
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = pose.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            images.append(image)
            
            proc.stdin.write(image.tostring())

            frame_pose = {}

            if results.pose_landmarks is not None:
                for lm_id, lm in enumerate(results.pose_landmarks.landmark):
                    h, w, _ = image.shape
                    frame_pose[str(2 * lm_id)] = lm.x * w
                    frame_pose[str(2 * lm_id + 1)] = lm.y * h
                
                poses.append(frame_pose)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

        video_cap.release()
        proc.stdin.close()
        proc.stderr.close()
        proc.wait()

    os.remove(video_download_path)

    return poses, images, output_path


def upload_output_video(output_path):

    upload_path = "output-videos/" + os.path.basename(output_path)
    upload_blob = appbucket.blob(upload_path)
    upload_blob.upload_from_filename(output_path)
    os.remove(output_path)

    return upload_path
            

def normalize_data(raw_shot_data):
    
    normalized_shot_data = []
    
    for pose in raw_shot_data:
        data = np.asarray(list(pose.values()))
        norm_array = np.linalg.norm(data)
        data_norm = data / norm_array
        data_norm = data_norm.reshape(1, -1)
        normalized_shot_data.append(data_norm)
        
    return normalized_shot_data


def predict_pose(model, pose_data):
    pose_prediction = list(model.predict(pose_data)[0])

    error_in = (1 - pose_prediction[0])**2
    error_out = (1 - pose_prediction[1])**2

    return error_in, error_out

