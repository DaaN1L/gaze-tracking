from typing import Literal

import numpy as np

from .utils import cut_rois, resize_input
from .base_estimator import BaseEstimator


class LandmarksDetector(BaseEstimator):
    class Result:
        def __init__(self, output, output_type: Literal['center', 'edges']):
            self.raw_output = output
            self.output_size = output_type
            self.center = self._get_centers()  # (left, right)

            size = np.linalg.norm(abs(self.center[0] - self.center[1]))
            self.size = np.asarray([size, size])  # (w, h)

            self.position = self.center - self.size[0] / 2  # (x, y)

        def _get_centers(self):
            if self.output_size == "center":
                right_eye_center, left_eye_center = self.raw_output[:2, :]
            else:
                right_eye_center = self.raw_output[:2, :].sum(axis=0) / 2
                left_eye_center = self.raw_output[2:4, :].sum(axis=0) / 2
            return np.asarray([left_eye_center, right_eye_center])

        def resize(self, frame_width, frame_height):
            self.position *= frame_width, frame_height
            self.center *= frame_width, frame_height
            self.size *= frame_width, frame_height

        def shift(self, x_shift, y_shift):
            self.position += x_shift, y_shift
            self.center += x_shift, y_shift

        def clip(self, width, height):
            min = [0, 0]
            max = [width, height]
            self.position[:] = np.clip(self.position, min, max)
            self.center[:] = np.clip(self.center, min, max)
            self.size[:] = np.clip(self.size, min, max)

    def __init__(self, core, args):
        super(LandmarksDetector, self).__init__(core, args.path, 'Landmarks Detection')
        self.points_number = args.points_number

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        self.output_type = "center" if args.name == "landmarks-regression-retail-0009" else "edges"
        if not self.points_number * 2 == self.output_shape[1]:
            raise RuntimeError("The model expects output shape {}, got {}".format(
                [1, self.points_number * 2, 1, 1], self.output_shape))

    def preprocess(self, frame, rois):
        self.input_size = frame.shape  # (h, w)
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape, self.nchw_layout) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_tensor_name: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        self.rois = {}
        for input, roi in zip(inputs, rois):
            self.rois[self.active_requests] = roi
            self.enqueue(input)

    def postprocess(self):
        outputs = self.get_outputs()

        results = []
        for i, output in enumerate(outputs):
            roi = self.rois[i]
            output = output.reshape((-1, 2)).astype(np.float64)
            result = LandmarksDetector.Result(output, self.output_type)
            result.resize(*roi.size)
            result.shift(*roi.position)
            result.clip(self.input_size[1], self.input_size[0])
            results.append(result)
        return results
