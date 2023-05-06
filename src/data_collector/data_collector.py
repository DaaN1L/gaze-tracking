import pandas as pd


class DataCollector:
    _COLUMNS = [
        "face_position_x",
        "face_position_y",
        "face_size_w",
        "face_size_h",
        "distance_to_face",
        "left_eye_position_x",
        "left_eye_position_y",
        "right_eye_position_x",
        "right_eye_position_y",
        "eyes_size_w",
        "eyes_size_h",
        "head_pose_yaw",
        "head_pose_pitch",
        "head_pose_roll",
        "gaze_direction_ox",
        "gaze_direction_oy",
        "gaze_direction_oz",
        "screen_point_x",
        "screen_point_y",
    ]

    def __init__(self):
        self._data = []

    @property
    def num_collected(self):
        return len(self._data)

    def get(self):
        return pd.DataFrame(self._data, columns=self._COLUMNS)

    def add(self,
            face_position,
            face_size,
            distance_to_face,
            left_eye_position,
            right_eye_position,
            eyes_size,
            head_pose,
            gaze_direction,
            screen_point):
        new_row = (*face_position,
                   *face_size,
                   distance_to_face,
                   *left_eye_position,
                   *right_eye_position,
                   *eyes_size,
                   *head_pose,
                   *gaze_direction,
                   *screen_point)
        self._data.append(new_row)

    def load(self, path):
        df = pd.read_csv(path)
        self._data = df.values.tolist()

    def save(self, path):
        pd.DataFrame(self._data, columns=self._COLUMNS).to_csv(path)

    def reset(self):
        self._data = []
