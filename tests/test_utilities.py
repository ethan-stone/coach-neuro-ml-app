from app import utilities
import os
import tensorflow as tf
import cv2


def test_get_front_elbow_model():
    model = utilities.get_model("models/basketball/front_elbow.h5/", "front_elbow.h5")
    assert type(model) is tf.keras.Sequential
    

def test_get_front_legs_model():
    model = utilities.get_model("models/basketball/front_legs.h5/", "front_legs.h5")
    assert type(model) is tf.keras.Sequential


def test_download_video():
    download_path = utilities.download_video("source-videos/basketball_front_test.MOV")
    assert os.path.exists(download_path)


def test_get_poses_from_video():
    download_path = utilities.download_video("source-videos/basketball_front_test.MOV")
    poses, _, output_path = utilities.get_poses_from_video(download_path)
    assert os.path.exists(output_path)
    os.remove(output_path) # clean up tmp files


def test_upload_video():
    download_path = utilities.download_video("source-videos/basketball_front_test.MOV")
    _, __, output_path = utilities.get_poses_from_video(download_path)
    output_video_path = utilities.upload_output_video(output_path)
    assert output_video_path is not None


def test_normalize_data():
    download_path = utilities.download_video("source-videos/basketball_front_test.MOV")
    poses, _, output_path = utilities.get_poses_from_video(download_path)
    utilities.normalize_data(poses)
    os.remove(output_path) # clean up tmp files
