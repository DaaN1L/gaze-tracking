import numpy as np

from .base_estimator import BaseEstimator


class FaceDetectionEstimator(BaseEstimator):
    def predict(self, image):
        if image.shape != self.input_layer.shape:
            raise ValueError(f"The image size {image.shape} should be {self.input_layer}")


