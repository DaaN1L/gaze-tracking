import tkinter as tk
import tkinter.ttk as ttk

import numpy
from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import numpy as np
import cv2
from PIL.Image import fromarray
from PIL.ImageTk import PhotoImage
from omegaconf import DictConfig

from .data_collector import DataCollector
from .estimators import FrameProcessor
from .video_capture import VideoCapture
from .utils import draw_detections, resize_image, get_distance


class App(ThemedTk):
    _TOTAL_EXAMPLES_TEXT = "Total examples: {:4d}"
    _TRAIN_MSE_TEXT = "Train MSE: {:5.3f}"
    _VAL_MSE_TEXT = "Val MSE: {:5.3f}"
    _SECONDARY_BG = "#99FFFF"

    def __init__(self, args: DictConfig, title, update_delay=15):
        super().__init__()
        self._style_init()
        self._draw_heatmap = False
        self._is_training = False
        self._is_pressed = False
        self._canvas_contents = {}

        self.data_collector = DataCollector()
        self.frame_processor = FrameProcessor(args)
        self.video_capture = VideoCapture()

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.geometry("%dx%d" % (self.width, self.height))
        self.title(title)
        self.bg = tk.Canvas(self, bg="white")
        self.bg.place(x=0, y=0, relwidth=1, relheight=1)

        self._build_camera_frame(self)
        self._build_control_menu(self)

        self.bind("<Double-ButtonPress-1>", self._press_to_collect)

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

        info_frame = tk.Frame(frame, bg=self._SECONDARY_BG)
        self.num_examples_info = ttk.Label(
            info_frame,
            text=self._TOTAL_EXAMPLES_TEXT.format(self.data_collector.num_collected)
        )
        self.train_mse_info = ttk.Label(
            info_frame,
            text=self._TRAIN_MSE_TEXT.format(0)
        )
        self.val_mse_info = ttk.Label(
            info_frame,
            text=self._VAL_MSE_TEXT.format(0)
        )

        for widget in info_frame.winfo_children():
            widget.configure(background=self._SECONDARY_BG)
            widget.pack(side=tk.LEFT, pady=pad, padx=pad)

        buttons_frame = tk.Frame(frame, background=self._SECONDARY_BG)
        ttk.Button(buttons_frame, text='Load dataset', command=self.load_dataset).grid(column=0, row=0)
        ttk.Button(buttons_frame, text='Save dataset', command=self.save_dataset).grid(column=0, row=1)
        ttk.Button(buttons_frame, text='Reset dataset', command=self.reset_dataset).grid(column=0, row=2)
        ttk.Button(buttons_frame, text='Draw heatmap', command=self.start_drawing_heatmap).grid(column=1, row=0)
        ttk.Button(buttons_frame, text='Stop drawing heatmap', command=self.stop_drawing_heatmap).grid(column=1, row=1)
        ttk.Button(buttons_frame, text='Clear Heatmap', command=self.clear_heatmap).grid(column=1, row=2)
        ttk.Button(buttons_frame, text='Start training', command=self.start_training).grid(column=0, row=3, columnspan=2)

        for widget in buttons_frame.winfo_children():
            widget.grid(padx=pad, pady=pad, sticky="ew")

        for widget in frame.winfo_children():
            widget.pack(padx=pad, pady=pad)
        frame.pack(side=side, anchor=anchor)

    def reset_dataset(self):
        self.data_collector.reset()
        self.num_examples_info.configure(text=self._TOTAL_EXAMPLES_TEXT.format(self.data_collector.num_collected))

    def load_dataset(self):
        self.stop_training()
        filename = askopenfilename(
            title="Select a file...",
            filetypes=(("Comma separated value file", "*.csv"),)
        )
        if filename:
            self.data_collector.load(filename)
            self.num_examples_info.configure(text=self._TOTAL_EXAMPLES_TEXT.format(self.data_collector.num_collected))

    def save_dataset(self):
        filename = asksaveasfilename(
            title="Save file as...",
            filetypes=(("Comma separated value file", "*.csv"),),
            defaultextension=".csv"
        )
        if filename:
            self.data_collector.save(filename)

    def start_drawing_heatmap(self):
        self.stop_training()
        self._draw_heatmap = True

    def stop_drawing_heatmap(self):
        self._draw_heatmap = False

    def clear_heatmap(self):
        self.heatmap = None
        self.bg.delete("all")

    def start_training(self):
        self._is_training = True
        self.stop_drawing_heatmap()
        self.clear_heatmap()

    def stop_training(self):
        self._is_training = False
        self._is_pressed = False

    def _press_to_collect(self, event):
        if self._is_training:
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
        self.num_examples_info.configure(text=self._TOTAL_EXAMPLES_TEXT.format(self.data_collector.num_collected))

    def _draw_in_canvas(self, canvas, array):
        self._canvas_contents[canvas] = PhotoImage(image=fromarray(array))
        canvas.create_image(0, 0, image=self._canvas_contents[canvas], anchor=tk.NW)

    def update(self):
        # Get a frame from the video source
        color_frame, depth_frame = self.video_capture.get_frame()
        if color_frame is None:
            self.after(self.delay, self.update)
            return

        rois, landmarks, gazes, head_poses = self.frame_processor.process(color_frame)
        draw_detections(color_frame, (rois, landmarks, gazes))
        processed_face_frame = resize_image(color_frame, (self.width // 5, self.height // 4), keep_aspect_ratio=True)
        self._draw_in_canvas(self.cam_canvas, processed_face_frame)

        bg_im = numpy.full(shape=(self.height, self.width), fill_value=255, dtype="uint8")
        self._draw_in_canvas(self.bg, bg_im)

        if self._is_training and self._is_pressed and rois:
            x, y = self.get_mouse_position()
            dist = get_distance(depth_frame, rois[0].position, rois[0].size)
            self.bg.create_text(x - 30, y - 25, anchor=tk.NW, text="CLICK!", fill="#004D40", font="Georgia, 16")

            for roi, eyes, gaze, head_pose in zip(rois, landmarks, gazes, head_poses):
                self.collect_data(
                    roi=roi,
                    eyes=eyes,
                    head_pose=head_pose,
                    gaze=gaze,
                    dist=dist,
                    screen_point=(x, y),
                )

        self.after(self.delay, self.update)


if __name__ == "__main__":
    app = App("Gaze tracker")
