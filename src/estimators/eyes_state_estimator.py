import numpy as np

from .base_estimator import BaseEstimator


class EyesStateEstimator(BaseEstimator):
    def predict(self, image, threshold: float = 0.5) -> int:
        if image.shape != self.input_layer.shape:
            raise ValueError(f"The image size {image.shape} should be {self.input_layer}")

        proba = np.squeeze(self.compiled_model([image])[self.output_layer])[0]
        y = (proba >= threshold).astype(int)
        return y
