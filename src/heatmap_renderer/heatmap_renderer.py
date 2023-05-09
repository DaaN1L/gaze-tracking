import numpy as np
import cv2

from src.utils import resize_image


class HeatmapRenderer:
    def __init__(self, size, bg=None, radius=175, filling_rate=8, decay_rate=0.98):  # (w, h)
        self.bg = resize_image(bg, size) if bg is not None else None
        self._decay_rate = decay_rate
        self._filling_rate = filling_rate
        self._size = size
        self._radius = radius
        self.clear()

    def step(self, point):  # (x, y)
        p = np.clip(point, [0, 0], [self._size[0], self._size[1]])
        self._heatmap = self._heatmap * self._decay_rate

        mask = np.ogrid[:self._size[1], :self._size[0]]
        dist = np.sqrt((mask[0] - p[1]) ** 2 + (mask[1] - p[0]) ** 2)
        dist = (self._radius - dist).clip(0, self._radius)
        self._heatmap = self._heatmap + dist * self._filling_rate
        self._heatmap.clip(0, 255)

    def clear(self):
        self._heatmap = np.full(shape=(self._size[1], self._size[0]), fill_value=255, dtype="uint8")

    @property
    def heatmap(self):
        color_map = cv2.applyColorMap(cv2.convertScaleAbs(self._heatmap, alpha=0.03), cv2.COLORMAP_TURBO)
        color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
        if self.bg is not None:
            color_map = cv2.addWeighted(color_map, 0.7, self.bg, 0.3, 0)
        return color_map
