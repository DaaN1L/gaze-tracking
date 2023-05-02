from abc import ABC, abstractmethod
import logging as log

from openvino.runtime import AsyncInferQueue
import numpy as np


class BaseEstimator(ABC):
    def __init__(self, core, model_path, model_type):
        self.core = core
        self.model_type = model_type
        log.info('Reading {} model from {}'.format(model_type, model_path))
        self.model = core.read_model(model_path)
        self.output_shape = self.model.outputs[0].shape
        self.input_tensor_name = self.model.inputs[0].get_any_name()
        self.active_requests = 0

        self.outputs = {}
        self.output_tensors = None
        self.max_requests = None
        self.infer_queue = None

    def deploy(self, device, max_requests=1):
        self.max_requests = max_requests
        compiled_model = self.core.compile_model(self.model, device)
        self.output_tensors = compiled_model.outputs
        self.infer_queue = AsyncInferQueue(compiled_model, self.max_requests)
        self.infer_queue.set_callback(self.completion_callback)
        log.info('The {} model is loaded to {}'.format(self.model_type, device))

    def completion_callback(self, infer_request, id):
        self.outputs[id] = [infer_request.results[out] for out in self.output_tensors]

    def enqueue(self, input):
        if self.max_requests <= self.active_requests:
            log.warning('Processing request rejected - too many requests')
            return False

        self.infer_queue.start_async(input, self.active_requests)
        self.active_requests += 1
        return True

    def wait(self):
        if self.active_requests <= 0:
            return
        self.infer_queue.wait_all()
        self.active_requests = 0

    def get_outputs(self):
        self.wait()
        return [np.squeeze(v) for _, v in sorted(self.outputs.items())]

    def clear(self):
        self.outputs = {}

    def infer(self, inputs):
        self.clear()
        self.start_async(*inputs)
        return self.postprocess()

    @abstractmethod
    def postprocess(self):
        ...

    @abstractmethod
    def start_async(self, *frame):
        ...
