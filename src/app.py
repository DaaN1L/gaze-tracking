import tkinter as tk
import tkinter.ttk as ttk

from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import numpy as np
from omegaconf import DictConfig
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from PIL.Image import fromarray
from PIL.ImageTk import PhotoImage

from .data_collector import DataCollector
from .estimators import FrameProcessor, ScreenPointEstimator
from .heatmap_renderer import HeatmapRenderer
from .video_capture import VideoCapture
from .utils import draw_detections, resize_image, get_distance


class App(ThemedTk):
    _TOTAL_EXAMPLES_TMP = "Total examples: {:4d}"
    _TRAIN_MSE_TMP = "Train MSE: {:5.3f}"
    _VAL_MSE_TMP = "Val MSE: {:5.3f}"
    _SECONDARY_BG = "#99FFFF"

    def __init__(self, args: DictConfig, title, update_delay=15):
        super().__init__()
        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.geometry("%dx%d" % (self.width, self.height))
        self.title(title)
        self.bg = tk.Canvas(self, bg="white")
        self.bg.place(x=0, y=0, relwidth=1, relheight=1)
        self.bind("<Double-ButtonPress-1>", self._press_to_collect)

        self._is_drawing_heatmap = False
        self._is_collecting = False
        self._is_pressed = False
        self._canvas_contents = {}

        self.data_collector = DataCollector()
        self.frame_processor = FrameProcessor(args)
        self.video_capture = VideoCapture()
        self.screen_point_estimator = ScreenPointEstimator(Ridge)
        self.heatmap_renderer = HeatmapRenderer(size=(self.width, self.height))

        self._style_init()
        self._build_camera_frame(self)
        self._build_control_menu(self)

        self.delay = update_delay
        self.update()

    def _style_init(self):
        style = ttk.Style()
        style.theme_use("winxpblue")
        style.configure('TLabel', font=("Georgia", 16))
        style.configure('TButton', font=("Georgia", 14))

    def _build_camera_frame(self, container, side=tk.LEFT, anchor=tk.N):
        self.cam_canvas = tk.Canvas(
            container,
            width=int(self.width // 5),
            height=int(self.height // 4),
            background=self._SECONDARY_BG
        )
        self.cam_canvas.pack(side=side, anchor=anchor)

    def _build_control_menu(self, container, side=tk.RIGHT, anchor=tk.NE):
        pad = 10
        frame = tk.Frame(container, bg=self._SECONDARY_BG)

        self.info = {}
        info_frame = tk.Frame(frame, bg=self._SECONDARY_BG)
        self.info["total_examples"] = ttk.Label(info_frame,
                                                text=self._TOTAL_EXAMPLES_TMP.format(self.data_collector.num_collected))
        self.info["train_mse"] = ttk.Label(info_frame,
                                           text=self._TRAIN_MSE_TMP.format(self.screen_point_estimator.train_mse))
        self.info["val_mse"] = ttk.Label(info_frame,
                                         text=self._VAL_MSE_TMP.format(self.screen_point_estimator.val_mse))
        for widget in info_frame.winfo_children():
            widget["background"] = self._SECONDARY_BG
            widget.pack(side=tk.LEFT, pady=pad, padx=pad)
        info_frame.pack(padx=pad, pady=pad)

        self.buttons = {}
        buttons_frame = tk.Frame(frame, background=self._SECONDARY_BG)
        self.buttons["load_dataset"] = ttk.Button(buttons_frame,
                                                  text='Load dataset',
                                                  command=self.load_dataset)
        self.buttons["save_dataset"] = ttk.Button(buttons_frame,
                                                  text='Save dataset',
                                                  command=self.save_dataset,
                                                  state="disabled")
        self.buttons["reset"] = ttk.Button(buttons_frame,
                                           text='Reset',
                                           command=self.reset)
        self.buttons["collect_data"] = ttk.Button(buttons_frame,
                                                  text='Collect data',
                                                  command=self.start_collecting)
        self.buttons["draw_heatmap"] = ttk.Button(buttons_frame,
                                                  text='Draw heatmap',
                                                  command=self.start_drawing_heatmap,
                                                  state="disabled")
        self.buttons["stop_drawing_heatmap"] = ttk.Button(buttons_frame,
                                                          text='Stop drawing heatmap',
                                                          command=self.stop_drawing_heatmap,
                                                          state="disabled")
        self.buttons["clear_heatmap"] = ttk.Button(buttons_frame,
                                                   text='Clear Heatmap',
                                                   command=self.clear_heatmap,
                                                   state="disabled")
        self.buttons["start_training"] = ttk.Button(buttons_frame,
                                                    text='Start training',
                                                    command=self.start_training,
                                                    state="disabled")

        n, m = np.meshgrid(range(len(self.buttons) // 2), range(2))
        for widget, i, j in zip(buttons_frame.winfo_children(), n.ravel(), m.ravel()):
            widget.grid(padx=pad, pady=pad, sticky="ew", row=i, column=j)
        buttons_frame.pack(padx=pad, pady=pad)

        frame.pack(side=side, anchor=anchor)

    def _disable_button(self, button: str | list[str]):
        if isinstance(button, str):
            self.buttons[button]["state"] = "disabled"
        else:
            for b in button:
                self.buttons[b]["state"] = "disabled"

    def _activate_button(self, button: str | list[str]):
        if isinstance(button, str):
            self.buttons[button]["state"] = "normal"
        else:
            for b in button:
                self.buttons[b]["state"] = "normal"

    def reset(self):
        if self._is_collecting:
            self.stop_collecting()
        if self._is_drawing_heatmap:
            self.stop_drawing_heatmap()
        self._activate_button(["load_dataset", "collect_data"])
        self._disable_button(["start_training", "clear_heatmap", "save_dataset", "stop_drawing_heatmap", "draw_heatmap"])
        self.data_collector.reset()
        self.heatmap_renderer.clear()
        self.info["total_examples"]["text"] = self._TOTAL_EXAMPLES_TMP.format(self.data_collector.num_collected)

    def load_dataset(self):
        self.stop_collecting()
        self._activate_button("start_training")
        filename = askopenfilename(
            title="Select a file...",
            filetypes=(("Comma separated value file", "*.csv"),)
        )
        if filename:
            self.data_collector.load(filename)
            self.info["total_examples"]["text"] = self._TOTAL_EXAMPLES_TMP.format(self.data_collector.num_collected)

    def save_dataset(self):
        filename = asksaveasfilename(
            title="Save file as...",
            filetypes=(("Comma separated value file", "*.csv"),),
            defaultextension=".csv"
        )
        if filename:
            self.data_collector.save(filename)

    def start_drawing_heatmap(self):
        if self._is_collecting:
            self.stop_collecting()
        self._activate_button(["clear_heatmap", "stop_drawing_heatmap"])
        self._disable_button("draw_heatmap")
        self._is_drawing_heatmap = True

    def stop_drawing_heatmap(self):
        self._activate_button("draw_heatmap")
        self._disable_button("stop_drawing_heatmap")
        self._is_drawing_heatmap = False

    def clear_heatmap(self):
        self.heatmap_renderer.clear()

    def start_collecting(self):
        if self._is_drawing_heatmap:
            self.stop_drawing_heatmap()
            self.clear_heatmap()
        self._disable_button("collect_data")
        self._activate_button(["start_training", "save_dataset"])
        self._is_collecting = True

    def stop_collecting(self):
        self._activate_button("collect_data")
        self._is_collecting = False
        self._is_pressed = False

    def start_training(self):
        self._disable_button("start_training")
        self._activate_button("draw_heatmap")
        self.screen_point_estimator.fit(self.data_collector.data, self.data_collector.target_columns)
        self.info["train_mse"]["text"] = self._TRAIN_MSE_TMP.format(self.screen_point_estimator.train_mse)
        self.info["val_mse"]["text"] = self._VAL_MSE_TMP.format(self.screen_point_estimator.val_mse)

    def _press_to_collect(self, event):
        if self._is_collecting:
            self._is_pressed = True

    def get_mouse_position(self):
        x = self.winfo_pointerx() - self.winfo_rootx()
        y = self.winfo_pointery() - self.winfo_rooty()
        return x, y

    def collect_data(self, roi, eyes, head_pose, gaze, dist, screen_point):
        self._is_pressed = False
        self.data_collector.add(
            face_position=roi.position,
            face_size=roi.size,
            left_eye_position=eyes.position[0],
            right_eye_position=eyes.position[1],
            eyes_size=eyes.size,
            head_pose=head_pose,
            gaze_direction=gaze,
            distance_to_face=dist,
            screen_point=screen_point
        )
        self.info["total_examples"]["text"] = self._TOTAL_EXAMPLES_TMP.format(self.data_collector.num_collected)

    def _draw_in_canvas(self, canvas, array):
        self._canvas_contents[canvas] = PhotoImage(image=fromarray(array))
        canvas.create_image(0, 0, image=self._canvas_contents[canvas], anchor=tk.NW)

    def update(self):
        # Get a frame from the video source
        color_frame, depth_frame = self.video_capture.get_frame()
        if color_frame is None:
            self.after(self.delay, self.update)
            return

        # Draw face and detections
        rois, landmarks, gazes, head_poses = self.frame_processor.process(color_frame)
        draw_detections(color_frame, (rois, landmarks, gazes))
        processed_face_frame = resize_image(color_frame, (self.width // 5, self.height // 4), keep_aspect_ratio=True)
        self._draw_in_canvas(self.cam_canvas, processed_face_frame)

        # Find detections
        if not rois:
            self.after(self.delay, self.update)
            return
        roi, eyes, gaze, head_pose = rois[0], landmarks[0], gazes[0], head_poses[0]
        x, y = self.get_mouse_position()
        dist = get_distance(depth_frame, rois[0].position, rois[0].size)

        # Draw heatmap
        if self._is_drawing_heatmap:
            X = self.data_collector.make_row(face_position=roi.position,
                                             face_size=roi.size,
                                             distance_to_face=dist,
                                             left_eye_position=eyes.position[0],
                                             right_eye_position=eyes.position[1],
                                             eyes_size=eyes.size,
                                             head_pose=head_pose,
                                             gaze_direction=gaze, feature_augmentation=True)
            y_pred = self.screen_point_estimator.predict([X])[0]
            self.heatmap_renderer.step(y_pred)
        self._draw_in_canvas(self.bg, self.heatmap_renderer.heatmap)

        # Collect data
        if self._is_collecting and self._is_pressed and rois:
            self.bg.create_text(x - 30, y - 25, anchor=tk.NW, text="CLICK!", fill="#004D40", font="Georgia, 16")
            self.collect_data(roi=roi,
                              eyes=eyes,
                              head_pose=head_pose,
                              gaze=gaze,
                              dist=dist,
                              screen_point=(x, y))

        self.after(self.delay, self.update)


if __name__ == "__main__":
    app = App("Gaze tracker")
