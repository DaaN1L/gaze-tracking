import numpy as np
import cv2

from src.utils.resize_image import resize_image

def crop(frame, roi_position, roi_size):
    p1 = roi_position.astype(int)
    p1 = np.clip(p1, [0, 0], [frame.shape[1], frame.shape[0]])
    p2 = (roi_position + roi_size).astype(int)
    p2 = np.clip(p2, [0, 0], [frame.shape[1], frame.shape[0]])
    return frame[p1[1]:p2[1], p1[0]:p2[0]]


def cut_eyes(frame, eyes):
    return [[crop(frame, position, roi.size) for position in roi.position] for roi in eyes]


def cut_rois(frame, rois):
    return [crop(frame, roi.position, roi.size) for roi in rois]


def resize_input(image, target_shape, nchw_layout):  # (h, w, c)
    if nchw_layout:
        _, _, h, w = target_shape
    else:
        _, h, w, _ = target_shape
    resized_image = resize_image(image, (w, h))
    if nchw_layout:
        resized_image = resized_image.transpose((2, 0, 1)) # HWC->CHW
    resized_image = resized_image.reshape(target_shape)
    return resized_image
