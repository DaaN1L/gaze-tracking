def get_rectangle(roi_position, roi_size, frame_size):  # (h, w)
    x_min = max(int(roi_position[0]), 0)
    y_min = max(int(roi_position[1]), 0)
    x_max = min(int(roi_position[0] + roi_size[0]), frame_size[1])
    y_max = min(int(roi_position[1] + roi_size[1]), frame_size[0])
    return x_min, y_min, x_max, y_max
