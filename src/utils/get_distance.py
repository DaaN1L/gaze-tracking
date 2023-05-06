from .get_rectangle import get_rectangle
import pyrealsense2 as rs
import cv2


MAX_DIST = 1

def get_distance(depth_frame, position, size):  # (h, w)
    if isinstance(size, int) or isinstance(size, float):
        size = (size, size)
    xmin, ymin, xmax, ymax = get_rectangle(position, size, depth_frame.shape)
    depth = depth_frame[xmin:xmax, ymin:ymax].astype(float)
    dist, _, _, _ = cv2.mean(depth.clip(0, MAX_DIST))
    return dist
