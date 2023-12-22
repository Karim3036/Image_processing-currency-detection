import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.feature import hog
import pickle

classes=['1EGP', '5EGP','10EGP','20EGP','50EGP','100EGP','200EGP']
with open("model_EGP.p", "rb") as model_file:
    model = pickle.load(model_file)

def preprocess_image(img):
    # img = Image.open(image_path)
    img = img.resize((224, 224))
    img_array = np.array(img)
    return img_array

def HoG(image):
  # Resize the image to 130x276
  resized_img = cv2.resize(image, (64, 128))
  # print(resized_img.shape)

  # Convert the original image to gray scale
  resized_img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

  # Specify the parameters for our HOG descriptor
  win_size = (64, 128)  # You need to set a proper window size
  block_size = (16, 16)
  block_stride = (8, 8)
  cell_size = (8, 8)
  num_bins = 9

  # Set the parameters of the HOG descriptor using the variables defined above
  hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

  # Compute the HOG Descriptor for the gray scale image
  hog_descriptor = hog.compute(resized_img_gray)
  return hog_descriptor

class App(tk.Tk):
  def __init__(self):
    super().__init__()

    # Load the background image
    bg_image = Image.open(r"D:\College\Senior 2\Image Processing\Project\Old-world-map.jpg")
    bg_image = bg_image.resize((650, 500), Image.BICUBIC)
    self.bg_photo = ImageTk.PhotoImage(bg_image)

    # Create a label with the background image and place it at the bottom
    bg_label = tk.Label(self, image=self.bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    self.geometry("600x500")
    self.title("Currency Recognition")

    self.btn_upload = tk.Button(self, text="Upload Image", command=self.upload_image, font=("Times New Roman", 24), padx=20, pady=10, bg="gray")
    self.btn_upload.pack(pady=10)
    self.btn_upload.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

  def upload_image(self):
    self.filename = filedialog.askopenfilename(initialdir="./", title="Select Image", filetypes=(("jpg files", "*.jpg"),("png files", "*.png"),  ("jpeg files", "*.jpeg"), ("all files", "*.*")))
    if self.filename:
      self.image = Image.open(self.filename)
      self.image = self.image.resize((450, 300), Image.BICUBIC)
      self.photo = ImageTk.PhotoImage(self.image)

      self.img_label = tk.Label(self, text="Image successfully uploaded!")
      self.img_label.pack()

      self.img_label.config(image=self.photo)
      self.img_label.image = self.photo

      self.img_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)  # Adjust the rely value to position the image label

      self.start_button = tk.Button(self, text="Recognise", command=self.start_processing, bg="gray", font=("Times New Roman", 24), padx=20, pady=10)
      self.start_button.pack()
      self.start_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label


  def start_processing(self):
    for widget in self.winfo_children():
      widget.destroy()
    # Load the background image
    bg_image = Image.open(r"D:\College\Senior 2\Image Processing\Project\Old-world-map.jpg")
    bg_image = bg_image.resize((650, 500), Image.BICUBIC)
    self.bg_photo = ImageTk.PhotoImage(bg_image)

    # Create a label with the background image and place it at the bottom
    bg_label = tk.Label(self, image=self.bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    self.image = self.image.resize((224, 224))
    img_array = np.array(self.image)
    hog_descriptor = HoG(img_array)
    feat2 = hog_descriptor.reshape(1, -1)
    # Make predictions
    predictions = model.predict(feat2)
    # Convert predictions variable to int
    predictions = int(predictions)

    # Create a red box frame
    self.frame = tk.Frame(self, bg="gray", width=200, height=100)
    self.frame.pack()
    self.frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Print the predicted class on the screen inside the red box
    self.prediction_label = tk.Label(self.frame, text=classes[predictions], font=("Times New Roman", 24), pady=20, bg="gray")
    self.prediction_label.pack()
    self.prediction_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    # Add "Upload Another Image" button
    self.btn_upload_another = tk.Button(self, text="Upload Another Image", command=self.reset_app)
    self.btn_upload_another.pack(pady=10)

  def reset_app(self):
    self.destroy()  # Destroy the current app window
    app = App()  # Create a new instance of the App class
    app.mainloop()  # Start the new app

if __name__ == "__main__":
  app = App()
  app.mainloop()



if __name__ == "__main__":
  app = App()
  app.mainloop()
