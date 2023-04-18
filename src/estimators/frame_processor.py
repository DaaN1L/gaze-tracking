from openvino.runtime import Core
from omegaconf import DictConfig

import logging as log

from .face_detection_estimator import FaceDetector
from .facial_landmarks_estimator import LandmarksDetector


class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self, args: DictConfig):
        log.info('OpenVINO Runtime')
        core = Core()

        self.face_detector = FaceDetector(core, args.face_detection_estimator)
        self.landmarks_detector = LandmarksDetector(core, args.facial_landmarks_estimator)

        self.face_detector.deploy(device=args.device)
        self.landmarks_detector.deploy(args.device, self.QUEUE_SIZE)

    def process(self, frame):
        orig_image = frame.copy()

        rois = self.face_detector.infer((frame,))
        if self.QUEUE_SIZE < len(rois):
            log.warning('Too many faces for processing. Will be processed only {} of {}'
                        .format(self.QUEUE_SIZE, len(rois)))
            rois = rois[:self.QUEUE_SIZE]

        landmarks = self.landmarks_detector.infer((frame, rois))
        return [rois, landmarks]
