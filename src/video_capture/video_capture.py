import cv2
import pyrealsense2 as rs


class VideoCapture:
    def __init__(self):
        self.pipeline = rs.pipeline()
        self._check_device()

        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def _check_device(self):
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
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
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image

    def __del__(self):
        self.pipeline.stop()
