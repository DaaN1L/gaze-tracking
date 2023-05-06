import cv2
import pyrealsense2 as rs
import numpy as np


class VideoCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self._check_device()
        self.align = rs.align(rs.stream.color)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.profile = self.pipeline.start(self.config)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def _check_device(self):
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        found_rgb = False
        found_depth = False
        for s in device.sensors:
            sensor_name = s.get_info(rs.camera_info.name)
            if sensor_name == 'RGB Camera':
                found_rgb = True
            elif sensor_name == 'Stereo Module':
                found_depth = True
        if not found_rgb:
            raise Exception("The application requires camera with Color sensor")
        elif not found_depth:
            raise Exception("The application requires camera with Stereo sensor")

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data()) * self.depth_scale  # (h, w)
        color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)  # (h, w, c)
        return color_image, depth_image

    def __del__(self):
        self.pipeline.stop()
