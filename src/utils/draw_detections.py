import cv2
import numpy as np

from ..estimators.facial_landmarks_estimator import LandmarksDetector
from ..estimators.face_detection_estimator import FaceDetector


def draw_detections(frame, detections, output_transform):
    def get_rectangle(roi_position, roi_size):
        xmin = max(int(roi_position[0]), 0)
        ymin = max(int(roi_position[1]), 0)
        xmax = min(int(roi_position[0] + roi_size[0]), size[1])
        ymax = min(int(roi_position[1] + roi_size[1]), size[0])
        return xmin, ymin, xmax, ymax


    size = frame.shape[:2]
    frame = output_transform.resize(frame)
    for roi, landmarks, gaze in zip(*detections):
        xmin, ymin, xmax, ymax = get_rectangle(roi.position, roi.size)
        xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        gaze[1] = -gaze[1]
        gaze_length = landmarks.size[0] * 2
        for point in landmarks.center:
            x, y = map(int, output_transform.scale([point[0], point[1]]))
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 2)
            cv2.arrowedLine(frame, (x, y), (x + int(gaze[0] * gaze_length), y + int(gaze[1] * gaze_length)), (0, 0, 255), 5)

        for i in range(2):
            xmin, ymin, xmax, ymax = get_rectangle(landmarks.position[i], landmarks.size)
            xmin, ymin, xmax, ymax = output_transform.scale([xmin, ymin, xmax, ymax])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)


    return frame
