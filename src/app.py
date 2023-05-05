import tkinter as tk
import tkinter.ttk as ttk
from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import cv2
import PIL.Image
import PIL.ImageTk
from omegaconf import DictConfig

from .data_collector import DataCollector
from .estimators import FrameProcessor
from .video_capture import VideoCapture
from .utils import draw_detections, resize_image


class App(tk.Tk):
    def __init__(self, args: DictConfig, title, update_delay=15):
        super().__init__()
        self.data_collector = DataCollector()
        self.frame_processor = FrameProcessor(args)
        self.video_capture = MyVideoCapture()

        self.train_mse = 0
        self.val_mse = 0

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.geometry("%dx%d" % (self.width, self.height))
        self.title(title)
        # self.bg = tk.Canvas(self)
        # self.bg.pack(fill=tk.BOTH, expand=True)

        self._build_camera_frame(self)
        self._build_control_menu(self)

        self.delay = update_delay
        self.update()

    def reset_dataset(self):
        self.data_collector.reset()

    def load_dataset(self):
        filename = askopenfilename(
            title="Select a file...",
            filetypes=(("Comma separated value file", "*.csv"),)
        )
        if filename:
            self.data_collector.load(filename)

    def save_dataset(self):
        filename = asksaveasfilename(
            title="Save file as...",
            filetypes=(("Comma separated value file", "*.csv"),),
            defaultextension=".csv"
        )
        if filename:
            self.data_collector.save(filename)

    def start_drawing_heatmap(self):
        self.draw_heatmap = True

    def stop_drawing_heatmap(self):
        self.draw_heatmap = False

    def clear_heatmap(self):
        self.heatmap = None

    def start_training(self):
        self.stop_drawing_heatmap()
        self.clear_heatmap()
        self.is_training = True

    def _style_init(self):
        style = ttk.Style()
        style.theme_use("winxpblue")
        style.configure('TLabel', font=("Georgia", 16))
        style.configure('TButton', font=("Georgia", 14))

    def _build_camera_frame(self, container, side=tk.LEFT, anchor=tk.N):
        self.canvas = tk.Canvas(container, width=int(self.width // 4), height=int(self.height // 4))
        self.canvas.pack(side=side, anchor=anchor)

    def _build_control_menu(self, container, side=tk.RIGHT, anchor=tk.NE):
        bg_color = "#00FFFF"
        pad = 10
        frame = tk.Frame(container, bg=bg_color)

        info_frame = tk.Frame(frame, bg=bg_color)
        ttk.Label(info_frame, text=f"Total examples: {self.data_collector.num_collected}")
        ttk.Label(info_frame, text=f"Train MSE: {self.train_mse}")
        ttk.Label(info_frame, text=f"Val MSE: {self.val_mse}")

        for widget in info_frame.winfo_children():
            widget.configure(background=bg_color)
            widget.pack(side=tk.LEFT, pady=pad, padx=pad)

        buttons_frame = tk.Frame(frame, background=bg_color)
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

    def update(self):
        # Get a frame from the video source
        color_frame, depth_frame = self.video_capture.get_frame()
        rois, landmarks, gazes, head_poses = self.frame_processor.process(color_frame)
        processed_face_frame = draw_detections(color_frame, (rois, landmarks, gazes))
        processed_face_frame = resize_image(color_frame, (self.width // 4, self.height // 4), keep_aspect_ratio=True)

        if color_frame is not None:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(processed_face_frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        # if self.is_training:
        #     for roi, eyes_position, gaze, head_pose in zip(rois, landmarks, gazes, head_poses):
        #         self.data_collector.add(face_position=roi.)

        self.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), None
            else:
                return None, None
        else:
            return None, None

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    app = App("Gaze tracker")
    app.mainloop()