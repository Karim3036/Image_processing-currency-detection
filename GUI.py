from tkinter import filedialog
from PIL import Image, ImageTk
#import skimage.io as io
import tkinter as tk
import numpy as np
import pickle

with open("model.p", "rb") as model_file:
    model = pickle.load(model_file)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry("600x500")
        self.title("Image Upload and Process")

        self.btn_upload = tk.Button(self, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack(pady=10)

        self.img_label = tk.Label(self)
        self.img_label.pack()

        self.result_label = tk.Label(self, text="")
        self.result_label.pack()

    def upload_image(self):
        self.filename = filedialog.askopenfilename(initialdir="./", title="Select Image", filetypes=(("png files", "*.png"), ("jpg files", "*.jpg"), ("jpeg files", "*.jpeg"), ("all files", "*.*")))
        if self.filename:
            self.image = Image.open(self.filename)
            self.image = self.image.resize((400, 400), Image.ANTIALIAS)
            self.photo = ImageTk.PhotoImage(self.image)
            self.img_label.config(image=self.photo)
            self.img_label.image = self.photo

            image_array = preprocess_image(self.filename)
            result = model.predict(image_array)
            result_str = f"The result is: {result[0]}"
            self.result_label.config(text=result_str)

if __name__ == "__main__":
    app = App()
    app.mainloop()