import time
import tkinter as tk
import tkinter.ttk as ttk
from ttkthemes import ThemedTk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import cv2
import pandas as pd
import PIL.Image
import PIL.ImageTk


class App(ThemedTk):
    # Total examples, train/val MSE
    # Load dataset, Save dataset, Reset dataset
    # Start training, Draw heatmap, Stop drawing heatmap, Clear Heatmap

    def __init__(self, title, update_delay=15):
        super().__init__()
        self._style_init()
        self.data = []
        self.train_mse = 0
        self.val_mse = 0

        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry("%dx%d" % (width, height))
        self.title(title)

        # Create a canvas that can fit the above video source size
        self.vid = MyVideoCapture(0)
        self.canvas = tk.Canvas(self, width=self.vid.width, height=self.vid.height)
        self.canvas.pack(side=tk.LEFT, anchor=tk.N)
        self._build_control_menu(self)

        self.delay = update_delay
        self.update()

    def _style_init(self):
        style = ttk.Style()
        style.theme_use("winxpblue")
        style.configure('TLabel', font=("Georgia", 16))
        style.configure('TButton', font=("Georgia", 14))

    def reset_dataset(self):
        self.data = []

    def load_dataset(self):
        filename = askopenfilename(
            title="Select a file...",
            filetypes=(("Comma separated value file", "*.csv"),)
        )
        df = pd.read_csv(filename)
        self.data = df.values.tolist()

    def save_dataset(self):
        filename = asksaveasfilename(
            title="Save file as...",
            filetypes=(("Comma separated value file", "*.csv"),),
            defaultextension=".csv"
        )
        pd.DataFrame(self.data).to_csv(filename)

    def _build_control_menu(self, container, side=tk.RIGHT, anchor=tk.NE):
        bg_color = "#00FFFF"
        frame = tk.Frame(container, bg=bg_color)
        pad = 10

        info_frame = tk.Frame(frame, bg=bg_color)
        ttk.Label(info_frame, text=f"Total examples: {len(self.data)}", background=bg_color)
        ttk.Label(info_frame, text=f"Train MSE: {self.train_mse}", background=bg_color)
        ttk.Label(info_frame, text=f"Val MSE: {self.val_mse}", background=bg_color)

        for widget in info_frame.winfo_children():
            widget.pack(side=tk.LEFT, pady=pad, padx=pad)

        buttons_frame = tk.Frame(frame, background=bg_color)
        ttk.Button(buttons_frame, text='Load dataset', command=self.load_dataset).grid(column=0, row=0)
        ttk.Button(buttons_frame, text='Save dataset', command=self.save_dataset).grid(column=0, row=1)
        ttk.Button(buttons_frame, text='Reset dataset', command=self.reset_dataset).grid(column=0, row=2)
        ttk.Button(buttons_frame, text='Draw heatmap').grid(column=1, row=0)
        ttk.Button(buttons_frame, text='Stop drawing heatmap').grid(column=1, row=1)
        ttk.Button(buttons_frame, text='Clear Heatmap').grid(column=1, row=2)
        ttk.Button(buttons_frame, text='Start training').grid(column=0, row=3, columnspan=2)

        for widget in buttons_frame.winfo_children():
            widget.grid(padx=pad, pady=pad, sticky="ew")

        for widget in frame.winfo_children():
            widget.pack(padx=pad, pady=pad)
        frame.pack(side=side, anchor=anchor)


    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

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
                return (ret, cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (400, 300)))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


if __name__ == "__main__":
    app = App("Gaze tracker")
    app.mainloop()