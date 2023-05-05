import pandas as pd


class DataCollector:
    def __init__(self):
        self._data = []

    @property
    def num_collected(self):
        return len(self._data)

    def get(self):
        return pd.DataFrame(self._data)

    def add(self,
            face_position,
            face_size,
            distance_to_face,
            left_eye_position,
            left_eye_size,
            right_eye_position,
            right_eye_size,
            head_pose,
            gaze_direction,
            screen_point):
        new_row = (*face_position,
                   *face_size,
                   distance_to_face,
                   *left_eye_position,
                   *left_eye_size,
                   *right_eye_position,
                   *right_eye_size,
                   *head_pose,
                   *gaze_direction,
                   *screen_point)
        self._data.append(new_row)

    def load(self, path):
        df = pd.read_csv(path)
        self._data = df.values.tolist()

    def save(self, path):
        pd.DataFrame(self._data).to_csv(path)

    def reset(self):
        self._data = []
