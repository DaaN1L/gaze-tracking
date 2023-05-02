import numpy as np

from .utils import cut_eyes, resize_input
from .base_estimator import BaseEstimator


class GazeEstimator(BaseEstimator):
    def __init__(self, core, args):
        super(GazeEstimator, self).__init__(core, args.path, 'Gaze estimator')
        if len(self.model.inputs) != 3:
            raise RuntimeError("The model expects 3 input layer")
        if len(self.model.outputs) != 1:
            raise RuntimeError("The model expects 1 output layer")

        self.input_tensor_name = [inp.get_any_name() for inp in self.model.inputs]
        self.input_shape = self.model.inputs[0].shape
        self.nchw_layout = self.input_shape[1] == 3

    def preprocess(self, frame, rois, head_pose):
        inputs = cut_eyes(frame, rois)
        inputs = [[resize_input(eye_crop, self.input_shape, self.nchw_layout) for eye_crop in input] for input in inputs]
        head_pose = np.reshape(head_pose, (-1, *self.model.inputs[2].shape))
        return inputs, head_pose

    def enqueue(self, inputs):
        return super(GazeEstimator, self).enqueue({self.input_tensor_name[0]: inputs[0][0],
                                                   self.input_tensor_name[1]: inputs[0][1],
                                                   self.input_tensor_name[2]: inputs[1]})

    def start_async(self, frame, rois, head_pose):
        assert len(rois) == len(head_pose)
        eyes, head_poses = self.preprocess(frame, rois, head_pose)
        for eye, head_pose in zip(eyes, head_poses):
            self.enqueue((eye, head_pose))

    def postprocess(self):
        results = self.get_outputs()
        print(results)
        return results
