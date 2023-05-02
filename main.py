import logging as log

import pyrealsense2 as rs
import numpy as np
import cv2
import hydra
from omegaconf import DictConfig, OmegaConf

from src.estimators.frame_processor import FrameProcessor
from src.utils import OutputTransform, draw_detections


def deploy(cfg: DictConfig):
    frame_processor = FrameProcessor(cfg)
    if cfg.debug:
       cap = cv2.VideoCapture(0)
    else:
        pipeline = rs.pipeline()
        config = rs.config()

        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
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
            log.error("The application requires camera with Color sensor")
        elif not found_depth:
            log.error("The application requires camera with Stereo sensor")

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline.start(config)

    try:
        frame_num = 0
        while True:
            if cfg.debug:
                ret, frames = cap.read()
                color_image = np.asanyarray(frames)
                if not ret:
                    continue
            else:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

            if frame_num == 0:
                output_transform = OutputTransform(color_image.shape[:2], cfg.output_resolution)
                if cfg.output_resolution:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (color_image.shape[1], color_image.shape[0])

            detections = frame_processor.process(color_image)
            frame = draw_detections(color_image, detections, output_transform)

            frame_num += 1

            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)

    finally:
        pipeline.stop()


@hydra.main(config_path="conf", config_name="config")
def run_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    deploy(cfg)


if __name__ == "__main__":
    run_model()
