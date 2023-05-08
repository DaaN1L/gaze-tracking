import pandas as pd
import numpy as np


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

    @property
    def data(self):
        df = pd.DataFrame(self._data, columns=self._COLUMNS)
        X, y = df.drop(columns=self._COLUMNS[-2:]), df[self._COLUMNS[-2:]]
        X = self.generate_features(X)
        return pd.concat([X, y], axis=1)

    @property
    def target_columns(self):
        return self._COLUMNS[-2:]

    @staticmethod
    def make_row(face_position, face_size, distance_to_face, left_eye_position, right_eye_position, eyes_size,
                 head_pose, gaze_direction, screen_point=None, feature_augmentation=False):
        row = [*face_position, *face_size, distance_to_face,
               *left_eye_position, *right_eye_position, *eyes_size, *head_pose, *gaze_direction]
        if feature_augmentation:
            row = DataCollector.generate_features(row)
        if screen_point is not None:
            row.extend(screen_point)
        return row

    @staticmethod
    def generate_features(df: pd.DataFrame | list):
        if isinstance(df, pd.DataFrame):
            for column in df.columns:
                df[column + "-pow2"] = df[column].map(lambda x: x ** 2)
                df[column + "-pow3"] = df[column].map(lambda x: x ** 3)
                df[column + "-log"] = df[column].map(lambda x: np.log(max(1, x)))
                df[column + "-inv"] = df[column].map(lambda x: 1 / (x + 1e-6))
        else:
            for x in df[:]:
                df.append(x ** 2)
                df.append(x ** 3)
                df.append(np.log(max(1, x)))
                df.append(1 / (x + 1e-6))
        return df

    def add(self, face_position, face_size, distance_to_face,
            left_eye_position, right_eye_position, eyes_size,
            head_pose, gaze_direction, screen_point):
        new_row = self.make_row(
            face_position=face_position,
            face_size=face_size,
            distance_to_face=distance_to_face,
            left_eye_position=left_eye_position,
            right_eye_position=right_eye_position,
            eyes_size=eyes_size,
            head_pose=head_pose,
            gaze_direction=gaze_direction,
            screen_point=screen_point
        )
        self._data.append(new_row)

    def load(self, path):
        df = pd.read_csv(path)
        self._data = df.values.tolist()

    def save(self, path):
        pd.DataFrame(self._data, columns=self._COLUMNS).to_csv(path, index=False)

    def reset(self):
        self._data = []
