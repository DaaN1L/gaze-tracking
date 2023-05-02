import numpy as np
from omegaconf import DictConfig

from .base_estimator import BaseEstimator
from .utils import cut_eyes, resize_input


class EyesStateEstimator(BaseEstimator):
    def __init__(self, core, args: DictConfig):
        super(EyesStateEstimator, self).__init__(core, args.path, 'Eyes state estimation')

        if len(self.model.inputs) != 1:
            raise RuntimeError("The model expects 1 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3

    def preprocess(self, frame, rois):
        inputs = cut_eyes(frame, rois)
        inputs = [[resize_input(eye_crop, self.input_shape, self.nchw_layout) for eye_crop in input] for input in inputs]
        return inputs

    def enqueue(self, input):
        return super().enqueue({self.input_tensor_name: input})

    def start_async(self, frame, rois):
        inputs = self.preprocess(frame, rois)
        for input in inputs:
            self.enqueue(input[0])
            self.enqueue(input[1])

    def postprocess(self):
        outputs = np.squeeze(self.get_outputs())
        results = []
        for i in range(0, len(outputs), 2):
            result = outputs[i][0] and outputs[i+1][0]
            results.append(result)
        return results
