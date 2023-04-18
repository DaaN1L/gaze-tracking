import numpy as np

from utils import cut_rois, resize_input
from base_estimator import BaseEstimator


class EyesStateEstimator(BaseEstimator):
    POINTS_NUMBER = 5

    def __init__(self, core, model):
        super(EyesStateEstimator, self).__init__(core, model, 'Eyes state estimation')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3

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
