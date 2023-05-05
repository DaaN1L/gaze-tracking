from openvino.runtime import Core
from omegaconf import DictConfig

import logging as log

from . import FaceDetector, LandmarksDetector, HeadPoseEstimator, GazeEstimator


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args: DictConfig):
        log.info('OpenVINO Runtime')
        core = Core()

        self.face_detector = FaceDetector(core, args.face_detection_estimator)
        self.landmarks_detector = LandmarksDetector(core, args.facial_landmarks_estimator)
        self.head_pose_estimator = HeadPoseEstimator(core, args.head_pose_estimator)
        self.gaze_estimator = GazeEstimator(core, args.gaze_estimator)

        self.face_detector.deploy(device=args.device)
        self.landmarks_detector.deploy(args.device, self.QUEUE_SIZE)
        self.head_pose_estimator.deploy(args.device, self.QUEUE_SIZE)
        self.gaze_estimator.deploy(args.device, self.QUEUE_SIZE)

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        head_pose = self.head_pose_estimator.infer((frame, rois))
        gaze = self.gaze_estimator.infer((frame, landmarks, head_pose))

        return [rois, landmarks, gaze, head_pose]
