from app import utilities, models
import cv2
import mediapipe as mp


def basketball_front(video_path):
    print("Starting analysis")

    front_elbow_model = utilities.get_model("models/basketball/front_elbow.h5/", "front_elbow.h5")
    front_legs_model = utilities.get_model("models/basketball/front_legs.h5/", "front_legs.h5")

    video_download_path = utilities.download_video(video_path)
    
    poses, images, output_path = utilities.get_poses_from_video(video_download_path)
    output_video_path = utilities.upload_output_video(output_path)

    min_error_elbow_in = 1
    min_error_elbow_out = 1

    min_error_elbow_in_index = 0
    min_error_elbow_out_index = 0

    min_error_legs_narrow = 1
    min_error_legs_good = 1
    min_error_legs_wide = 1

    min_error_legs_narrow_index = 0
    min_error_legs_good_index = 0
    min_error_legs_wide_index = 0

    poses = utilities.normalize_data(poses)
    elbow_predictions = []
    legs_predictions = []

    for i in range(0, len(poses)):
        pose = poses[i]
        image = images[i]

        front_elbow_prediction = front_elbow_model.predict(pose)
        front_legs_prediction = front_legs_model.predict(pose)

        elbow_predictions.append(front_elbow_prediction)
        legs_predictions.append(front_legs_prediction)

        error_elbow_out = (1 - front_elbow_prediction[0][0])**2
        error_elbow_in = (1 - front_elbow_prediction[0][1])**2

        error_legs_narrow = (1 - front_legs_prediction[0][0])**2
        error_legs_good = (1 - front_legs_prediction[0][1])**2
        error_legs_wide = (1 - front_legs_prediction[0][2])**2

        if error_elbow_out < min_error_elbow_out:
            min_error_elbow_out = error_elbow_out
            min_error_elbow_out_index = i

        if error_elbow_in < min_error_elbow_in:
            min_error_elbow_in = error_elbow_in
            min_error_elbow_in_index = i
        
        if error_legs_narrow < min_error_legs_narrow:
            min_error_legs_narrow = error_legs_narrow
            min_error_legs_narrow_index = i

        if error_legs_good < min_error_legs_good:
            min_error_legs_good = error_legs_good
            min_error_legs_good_index = i

        if error_legs_wide < min_error_legs_wide:
            min_error_legs_wide = error_legs_wide
            min_error_legs_wide_index = i
    
    elbow_decision = min(min_error_elbow_out, min_error_elbow_in)

    elbow_decision_index = -1

    if elbow_decision == min_error_elbow_out:
        elbow_decision_index = min_error_elbow_out_index
    elif elbow_decision == min_error_elbow_in:
        elbow_decision_index = min_error_elbow_in_index

    legs_decision = min(min_error_legs_narrow, min_error_legs_good, min_error_legs_wide)

    legs_decision_index = -1

    if legs_decision == min_error_legs_narrow:
        legs_decision_index = min_error_legs_narrow_index
    if legs_decision == min_error_legs_good:
        legs_decision_index = min_error_legs_good_index
    if legs_decision == min_error_legs_wide:
        legs_decision_index = min_error_legs_wide_index

    elbow_decision_image = images[elbow_decision_index]
    legs_decision_image = images[legs_decision_index]

    elbow_decision_prediction = elbow_predictions[elbow_decision_index]

    legs_decision_prediction = legs_predictions[legs_decision_index]

    cv2.imwrite("tmp/images/elbow_decision.jpg", elbow_decision_image)
    cv2.imwrite("tmp/images/legs_decision.jpg", legs_decision_image)

    frontElbowPrediction = {
        "out": elbow_decision_prediction[0][0],
        "in": elbow_decision_prediction[0][1]
    }

    frontLegsPrediction = {
        "narrow": legs_decision_prediction[0][0],
        "good": legs_decision_prediction[0][1],
        "wide": legs_decision_prediction[0][2]
    }

    summary = models.BasketballFrontAnalysisSummary(frontElbowPrediction=frontElbowPrediction, frontLegsPrediction=frontLegsPrediction)

    return summary, output_video_path
