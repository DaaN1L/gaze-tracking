import numpy as np

from .utils import cut_rois, resize_input
from .base_estimator import BaseEstimator


class LandmarksDetector(BaseEstimator):
    def __init__(self, core, args):
        super(LandmarksDetector, self).__init__(core, args.path, 'Landmarks Detection')
        self.points_number = args.points_number

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3
        if not self.points_number * 2 == self.output_shape[1]:
            raise RuntimeError("The model expects output shape {}, got {}".format(
                [1, self.points_number * 2, 1, 1], self.output_shape))

    def preprocess(self, frame, rois):
        inputs = cut_rois(frame, rois)
        inputs = [resize_input(input, self.input_shape, self.nchw_layout) for input in inputs]
        return inputs

    def enqueue(self, input):
        return super(LandmarksDetector, self).enqueue({self.input_tensor_name: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input)

    def postprocess(self):
        results = [out.reshape((-1, 2)).astype(np.float64) for out in self.get_outputs()]
        return results
