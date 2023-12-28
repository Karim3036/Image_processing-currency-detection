import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.feature import hog
import pickle
from PIL import ImageTk
import requests
import re

classes=['1EGP','1SAR', '5EGP','5SAR','10EGP','10SAR','20EGP','50EGP','50SAR','100EGP','100SAR','200EGP','500SAR']
Egyptian=[0,2,4,6,7,9,11]
classes_EGP=['1EGP', '5EGP','10EGP','20EGP','50EGP','100EGP','200EGP']
Saudi=[1,3,5,8,10,12]
classes_SAR=['1SAR','5SAR','10SAR','50SAR','100SAR','500SAR']

#get pre trained model
with open("Currency_model_prob.p", "rb") as model_file:
    trained_model = pickle.load(model_file)

with open("Saudi_model.p", "rb") as model_file:
    SAR_model = pickle.load(model_file)

with open("Egypt_model.p", "rb") as model_file:
    EGP_model = pickle.load(model_file)

# #PROCESSING FUNCTIONS##
def auto_rotate_image(image, counter=1):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and aid contour detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_with_contour = image.copy()

    # Find the contour with the maximum area
    max_contour = max(contours, key=cv2.contourArea)

    # Fit a rotated bounding box to the contour
    rect = cv2.minAreaRect(max_contour)

    # Unpack the rectangle information
    (dummy, (width_contour, height_contour), angle) = rect

    # Ensure width is always the larger side
    if width_contour < height_contour:
        width_contour, height_contour = height_contour, width_contour
        angle -= 90

    height, width = image.shape[:2]
    width_ratio = float(width_contour / width)

    #Counter is for the second call of this function, in this call I am interested in getting the new coordinates of the bounding box but no further rotation needed.
    if counter == 0 and width_ratio > 0.65:
        # Rotate the image to the detected angle
        rotated_image = rotating_image(image_with_contour, angle)
        return rotated_image, max_contour
    else:
        return image_with_contour, max_contour

def rotating_image(image, angle):
    # Get the image dimensions
    height, width = image.shape[:2]

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Determine the new bounding box after rotation
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])

    new_width = int((width * cos_theta) + (height * sin_theta))
    new_height = int((width * sin_theta) + (height * cos_theta))

    # Adjust the rotation matrix for translation to keep the entire rotated image
    rotation_matrix[0, 2] += (new_width - width) / 2
    rotation_matrix[1, 2] += (new_height - height) / 2

    # Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return rotated_image

def crop_image(image, box):
    x, y, w, h = cv2.boundingRect(box)
    if w < h:
        w, h = h, w
    return image[y:y + h, x:x + w]

def Background_Removal(img):
    if img is not None:

      new_image, contour1 = auto_rotate_image(img, 0)
      rect1 = cv2.minAreaRect(contour1)
      box1 = cv2.boxPoints(rect1)
      box1 = np.int0(box1)

      # Unpack the rectangle information
      (dummy, (width_contour1, height_contour1), angle1) = rect1 #dummy variable as the rect returns 3 variables

      # Ensure width is always the larger side
      if width_contour1 < height_contour1:
          width_contour1, height_contour1 = height_contour1, width_contour1
          angle1 -= 90

      rotated_image, contour2 = auto_rotate_image(new_image)
      rect2 = cv2.minAreaRect(contour2)
      box2 = cv2.boxPoints(rect2)
      box2 = np.int0(box2)

      (dummy2, (width_contour2, height_contour2), angle2) = rect2

      # Ensure width is always the larger side
      if width_contour2 < height_contour2:
          width_contour2, height_contour2 = height_contour2, width_contour2
          angle2 -= 90

      if -1 < angle2 < 1:
          cropped_image = crop_image(rotated_image, box2)
      elif -1 < angle1 < 1:
          new_height, new_width = rotated_image.shape[:2]
          # Resize the image to avoid features cropping
          resized_image = cv2.resize(new_image, (new_width, new_height))
          cropped_image = crop_image(resized_image, box1)
      else:
          cropped_image = rotated_image

      return cropped_image

    else:
        print(f"Cannot open {img}")
        pass

def img_pre_processing(image):
    # Apply Gaussian filtering to smooth out the image and remove any noise that may affect the accuracy of the HOG feature extractor
    #we need to try different kernel sizes for the best!
    gaussian_image = cv2.GaussianBlur(image, (3, 3), 0)

    # Convert the image to floating point
    gaussian_image_float = gaussian_image.astype(np.float32)

    # Calculate the histogram
    hist = cv2.calcHist([image], [0], None, [256], [0,256])

    # Calculate the mean of the histogram
    mean_hist = np.mean(hist)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Extract the S channel
    s_channel = hsv_image[:, :, 1]

    # Calculate the average saturation
    average_saturation = np.mean(s_channel)

    # Decide the gamma value
    if (average_saturation > 128) and (mean_hist > 128):
        gamma = 1.2  # or any value greater than 1
    else:
        gamma = 0.8  # or any value less than 1

    # Apply gamma correction
    gamma_corrected_image = cv2.pow(gaussian_image_float/ 255, gamma)


    final_image_before_HOG=gamma_corrected_image*255


    return final_image_before_HOG

def HoG(image):
  # Resize the image to 130x276
  resized_img = cv2.resize(image, (64, 128))
  # print(resized_img.shape)

  # Convert the original image to gray scale
  resized_img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

  # Specify the parameters for our HOG descriptor
  win_size = (64, 128)  # You need to set a proper window size
  block_size = (32, 32)
  block_stride = (16, 16)
  cell_size = (16, 16)
  num_bins = 18

  # Set the parameters of the HOG descriptor using the variables defined above
  hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

  # Compute the HOG Descriptor for the gray scale image
  hog_descriptor = hog.compute(resized_img_gray)
  return hog_descriptor



class App(tk.Tk):
  def __init__(self):
    super().__init__()
    self.Start_GUI()
  def Background(self):
    # Load the background image
    bg_image = Image.open("Assets/Background_Pic.jpg")
    bg_image = bg_image.resize((650, 500), Image.BICUBIC)
    self.bg_photo = ImageTk.PhotoImage(bg_image)
    # Reload the background image
    bg_label = tk.Label(self, image=self.bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

  def Start_GUI(self):
    self.Background()

    # Create a label with the background image and place it at the bottom
    bg_label = tk.Label(self, image=self.bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    self.geometry("600x500")
    self.title("Currency Recognition")

    self.btn_upload = tk.Button(self, text="Upload Image With Black Background", command=lambda:self.upload_image('yes'), font=("Times New Roman", 16), padx=20, pady=10, bg="gray")
    self.btn_upload.pack(pady=10)
    self.btn_upload.place(relx=0.5, rely=0.3, anchor=tk.CENTER)

    self.message_label = tk.Label(self, text="Images with backgroud may give inaccurate results!", font=("Times New Roman", 16))
    self.message_label.pack(pady=10)
    self.message_label.place(relx=0.5, rely=0.4, anchor=tk.CENTER)

    self.btn_upload = tk.Button(self, text="Upload Image Without Any Background", command=lambda:self.upload_image("no"), font=("Times New Roman", 16), padx=20, pady=10, bg="gray")
    self.btn_upload.pack(pady=10)
    self.btn_upload.place(relx=0.5, rely=0.6, anchor=tk.CENTER)


  def upload_image(self,background):
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

      self.start_button = tk.Button(self, text="Recognise", command=lambda:self.start_processing(background), bg="gray", font=("Times New Roman", 24), padx=20, pady=10)
      self.start_button.pack()
      self.start_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label


  def start_processing(self,backgroud):
    for widget in self.winfo_children():
      widget.destroy()
    self.Background()
    print(backgroud)

    if backgroud =='no':
        # Create a label with the background image and place it at the bottom
      bg_label = tk.Label(self, image=self.bg_photo)
      bg_label.place(x=0, y=0, relwidth=1, relheight=1)
      #if width > height rotate 90 degrees counterclockwise
      if self.image.size[0] < self.image.size[1]:
        orientation = "vertical"
        self.image = self.image.transpose(Image.ROTATE_90)
      self.image = self.image.resize((224, 224))
      img_array = np.array(self.image)
      hog_image=img_pre_processing(img_array)
      hog_image=hog_image.astype(np.uint8)

      hog_descriptor = HoG(hog_image)
      feat2 = hog_descriptor.reshape(1, -1)
      #make predictions
      predictions = trained_model.predict(feat2)
      predictions = int(predictions)
      per=trained_model.predict_proba(feat2)
      #write code to get max value of per
      #maxval=np.amax(per,axis=1)
      max_pred=(np.max(per))
      #make max_pred into an float
      max_pred=float(max_pred)
      if (max_pred*100 >= 70):
        # predictions=int(predictions)
        self.Print_Prediction(predictions)
      elif (max_pred*100 < 70  and max_pred*100 >= 40):
        if(predictions in Egyptian):
            predect= EGP_model.predict(feat2)
            per2=EGP_model.predict_proba(feat2)
            max_pred2=(np.max(per2))
            if (max_pred2*100 >= 75):
                predect = int(predect)
                self.Print_Prediction(predect,'EGP')
            else:
                self.Wrong_Prediction()
        elif(predictions in Saudi):
          predect= SAR_model.predict(feat2)
          per2=SAR_model.predict_proba(feat2)
          max_pred2=(np.max(per2))
          if (max_pred2*100 >= 75):
              predect = int(predect)
              self.Print_Prediction(predect,'SAR')
          else:
              self.Wrong_Prediction()
        else:
            self.Wrong_Prediction()
      else:
        # Create a label with the background image and place it at the bottom
        bg_label = tk.Label(self, image=self.bg_photo)
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)
        self.image = self.image.resize((224, 224))
        img_array = np.array(self.image)
        self.image = Background_Removal(img_array)
        hog_descriptor = HoG(img_array)
        feat2 = hog_descriptor.reshape(1, -1)
        #make predictions
        predictions = trained_model.predict(feat2)
        predictions = int(predictions)
        per=trained_model.predict_proba(feat2)
        #write code to get max value of per
        #maxval=np.amax(per,axis=1)
        max_pred=(np.max(per))
        #make max_pred into an float
        max_pred=float(max_pred)
        if backgroud =='no':
          if (max_pred*100 >= 70):
            # predictions=int(predictions)
            self.Print_Prediction(predictions)
        elif (max_pred*100 < 70  and max_pred*100 >= 40):
          if(predictions in Egyptian):
              predect= EGP_model.predict(feat2)
              per2=EGP_model.predict_proba(feat2)
              max_pred2=(np.max(per2))
              if (max_pred2*100 >= 75):
                  predect = int(predect)
                  self.Print_Prediction(predect,'EGP')
              else:
                  self.Wrong_Prediction()
          elif(predictions in Saudi):
            predect= SAR_model.predict(feat2)
            per2=SAR_model.predict_proba(feat2)
            max_pred2=(np.max(per2))
            if (max_pred2*100 >= 75):
                predect = int(predect)
                self.Print_Prediction(predect,'SAR')
            else:
                self.Wrong_Prediction()
          else:
              self.Wrong_Prediction()

  def Print_Prediction(self,predictions,classe='None'):
      # Create a red box frame
      print(predictions)
      self.frame = tk.Frame(self, bg="gray", width=200, height=100)
      self.frame.pack()
      self.frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
      if classe=='None':
          # Print the predicted class on the screen inside the red box
          self.prediction_label = tk.Label(self.frame, text=classes[predictions], font=("Times New Roman", 24), pady=20, bg="gray")
          self.prediction_label.pack()
          self.prediction_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
      elif classe=='EGP':
          # Print the predicted class on the screen inside the red box
          self.prediction_label = tk.Label(self.frame, text=classes_EGP[predictions], font=("Times New Roman", 24), pady=20, bg="gray")
          self.prediction_label.pack()
          self.prediction_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
      elif classe=='SAR':
          # Print the predicted class on the screen inside the red box
          self.prediction_label = tk.Label(self.frame, text=classes_SAR[predictions], font=("Times New Roman", 24), pady=20, bg="gray")
          self.prediction_label.pack()
          self.prediction_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
      else:
         pass
      # Add "Upload Another Image" button
      self.btn_upload_another = tk.Button(self, text="Upload Another Image", command=self.reset_app,font=("Times New Roman", 16))
      self.btn_upload_another.pack(pady=10)
      self.btn_upload_another.place(relx=0.3, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label
      self.btn_convert_currency = tk.Button(self, text="Convert Currency", command=lambda: self.convert_currency(predictions,classe), font=("Times New Roman", 16))
      self.btn_convert_currency.pack(pady=10)
      self.btn_convert_currency.place(relx=0.7, rely=0.8, anchor=tk.CENTER)  # Place the button next to the "Upload Another Image" button
      pass

  def Wrong_Prediction(self):

      # print("BEEEP")
        self.message_label = tk.Label(self, text="ERROR! Please enter another image", font=("Times New Roman", 24))
        self.message_label.pack(pady=10)
        self.message_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
      # Add "Upload Another Image" button
        self.btn_upload_another = tk.Button(self, text="Upload Another Image", command=self.reset_app,font=("Times New Roman", 16))
        self.btn_upload_another.pack(pady=10)
        self.btn_upload_another.place(relx=0.5, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label
        pass

  def convert_currency(self,predictions,classe='None'):
    for widget in self.winfo_children():
      widget.destroy()
    self.Background()
    # print(classes[predictions])
    # Load and resize the flag images
    flag1 = Image.open("Assets/USA.png").resize((100, 100))
    flag2 = Image.open("Assets/EU.png").resize((100, 100))
    flag3 = Image.open("Assets/SA.png").resize((100, 100))
    flag4 = Image.open("Assets/EG.png").resize((100, 100))

    # Convert the flag images to Tkinter PhotoImage
    flag1_photo = ImageTk.PhotoImage(flag1)
    flag2_photo = ImageTk.PhotoImage(flag2)
    flag3_photo = ImageTk.PhotoImage(flag3)
    flag4_photo = ImageTk.PhotoImage(flag4)

    # Create buttons for the flag images and place them one below the other
    flag1_button = tk.Button(self, image=flag1_photo, command=lambda:self.flag1_clicked(predictions))
    flag1_button.image = flag1_photo
    flag1_button.pack(pady=30, anchor=tk.CENTER)

    flag2_button = tk.Button(self, image=flag2_photo, command=lambda:self.flag2_clicked(predictions))
    flag2_button.image = flag2_photo
    flag2_button.pack(pady=30, anchor=tk.CENTER)

    if "EGP" in str(classes[predictions]) or classe=='EGP':
      flag3_button = tk.Button(self, image=flag3_photo, command=lambda:self.flag3_clicked(predictions))
      flag3_button.image = flag3_photo
      flag3_button.pack(pady=30, anchor=tk.CENTER)
    else:
      flag4_button = tk.Button(self, image=flag4_photo, command=lambda:self.flag4_clicked(predictions))
      flag4_button.image = flag4_photo
      flag4_button.pack(pady=30, anchor=tk.CENTER)
    # Add "Upload Another Image" button
    self.btn_upload_another = tk.Button(self, text="Upload Another Image", command=self.reset_app,font=("Times New Roman", 16))
    self.btn_upload_another.pack(pady=10)
    self.btn_upload_another.place(relx=0.2, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label


  def flag1_clicked(self,predictions):
    for widget in self.winfo_children():
      widget.destroy()
    self.Background()
    currency = classes[predictions]
    number = re.findall(r'\d+', currency)
    number = int(number[0])  # Convert number to integer
    print(number)
    if "EGP" in str(currency):
      try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/' + 'EGP')
        data = response.json()

      # Get the exchange rate from the data
        rate = data['rates']['USD']
      except requests.exceptions.RequestException as e:
        print("Connection error")
        rate = 0.032  # or return to Convert Currency depending on your program structure

      # Convert the amount using the exchange rate
      converted_amount = number * rate
      print(converted_amount)
      # Create a label to display the converted amount
      self.converted_amount_label = tk.Label(self, text=str(converted_amount) + " USD", font=("Times New Roman", 24))
      self.converted_amount_label.pack(pady=10)
      self.converted_amount_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    else:
      try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/' + 'SAR')
        data = response.json()

      # Get the exchange rate from the data
        rate = data['rates']['USD']
      except requests.exceptions.RequestException as e:
        print("Connection error")
        rate = 0.27  # or return to Convert Currency depending on your program structure

      # Convert the amount using the exchange rate
      converted_amount = number * rate
      print(converted_amount)
      # Create a label to display the converted amount
      self.converted_amount_label = tk.Label(self, text=str(converted_amount) + " USD", font=("Times New Roman", 24))
      self.converted_amount_label.pack(pady=10)
      self.converted_amount_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Add "Upload Another Image" button
    self.btn_upload_another = tk.Button(self, text="Upload Another Image", command=self.reset_app,font=("Times New Roman", 16))
    self.btn_upload_another.pack(pady=10)
    self.btn_upload_another.place(relx=0.3, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label
    # Create a button for currency conversion
    self.btn_convert_currency = tk.Button(self, text="Convert to another Currency", command=lambda: self.convert_currency(predictions), font=("Times New Roman", 16))
    self.btn_convert_currency.pack(pady=10)
    self.btn_convert_currency.place(relx=0.7, rely=0.8, anchor=tk.CENTER)  # Place the button next to the "Upload Another Image" button




  def flag2_clicked(self,predictions):
    for widget in self.winfo_children():
      widget.destroy()
    self.Background()
    currency = classes[predictions]
    number = re.findall(r'\d+', currency)
    number = int(number[0])  # Convert number to integer
    print(number)
    if "EGP" in str(currency):
      try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/' + 'EGP')
        data = response.json()

      # Get the exchange rate from the data
        rate = data['rates']['EUR']
      except requests.exceptions.RequestException as e:
        print("Connection error")
        rate = 0.029  # or return to Convert Currency depending on your program structure

      # Convert the amount using the exchange rate
      converted_amount = number * rate
      print(converted_amount)
      # Create a label to display the converted amount
      self.converted_amount_label = tk.Label(self, text=str(converted_amount) + " Euro", font=("Times New Roman", 24))
      self.converted_amount_label.pack(pady=10)
      self.converted_amount_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    else:
      try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/' + 'SAR')
        data = response.json()

      # Get the exchange rate from the data
        rate = data['rates']['EUR']
      except requests.exceptions.RequestException as e:
        print("Connection error")
        rate = 0.24  # or return to Convert Currency depending on your program structure

      # Convert the amount using the exchange rate
      converted_amount = number * rate
      print(converted_amount)
      # Create a label to display the converted amount
      self.converted_amount_label = tk.Label(self, text=str(converted_amount) + " Euro", font=("Times New Roman", 24))
      self.converted_amount_label.pack(pady=10)
      self.converted_amount_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Add "Upload Another Image" button
    self.btn_upload_another = tk.Button(self, text="Upload Another Image", command=self.reset_app,font=("Times New Roman", 16))
    self.btn_upload_another.pack(pady=10)
    self.btn_upload_another.place(relx=0.3, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label
    # Create a button for currency conversion
    self.btn_convert_currency = tk.Button(self, text="Convert to another Currency", command=lambda: self.convert_currency(predictions), font=("Times New Roman", 16))
    self.btn_convert_currency.pack(pady=10)
    self.btn_convert_currency.place(relx=0.7, rely=0.8, anchor=tk.CENTER)  # Place the button next to the "Upload Another Image" button


  def flag3_clicked(self,predictions):
    for widget in self.winfo_children():
      widget.destroy()
    self.Background()
    currency = classes[predictions]
    number = re.findall(r'\d+', currency)
    number = int(number[0])  # Convert number to integer
    print(number)

    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/' + 'EGP')
        data = response.json()

      # Get the exchange rate from the data
        rate = data['rates']['SAR']
    except requests.exceptions.RequestException as e:
        print("Connection error")
        rate = 0.12  # or return to Convert Currency depending on your program structure

    # Convert the amount using the exchange rate
    converted_amount = number * rate
    print(converted_amount)
    # Create a label to display the converted amount
    self.converted_amount_label = tk.Label(self, text=str(converted_amount) + " SAR", font=("Times New Roman", 24))
    self.converted_amount_label.pack(pady=10)
    self.converted_amount_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Add "Upload Another Image" button
    self.btn_upload_another = tk.Button(self, text="Upload Another Image", command=self.reset_app,font=("Times New Roman", 16))
    self.btn_upload_another.pack(pady=10)
    self.btn_upload_another.place(relx=0.3, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label
    # Create a button for currency conversion
    self.btn_convert_currency = tk.Button(self, text="Convert to another Currency", command=lambda: self.convert_currency(predictions), font=("Times New Roman", 16))
    self.btn_convert_currency.pack(pady=10)
    self.btn_convert_currency.place(relx=0.7, rely=0.8, anchor=tk.CENTER)  # Place the button next to the "Upload Another Image" button


  def flag4_clicked(self,predictions):
    for widget in self.winfo_children():
      widget.destroy()
    self.Background()
    currency = classes[predictions]
    number = re.findall(r'\d+', currency)
    number = int(number[0])  # Convert number to integer
    print(number)

    try:
        response = requests.get('https://api.exchangerate-api.com/v4/latest/' + 'SAR')
        data = response.json()

      # Get the exchange rate from the data
        rate = data['rates']['EGP']
    except requests.exceptions.RequestException as e:
        print("Connection error")
        rate = 8.21 # or return to Convert Currency depending on your program structure

    # Convert the amount using the exchange rate
    converted_amount = number * rate
    print(converted_amount)
    # Create a label to display the converted amount
    self.converted_amount_label = tk.Label(self, text=str(converted_amount) + " EGP", font=("Times New Roman", 24))
    self.converted_amount_label.pack(pady=10)
    self.converted_amount_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    # Add "Upload Another Image" button
    self.btn_upload_another = tk.Button(self, text="Upload Another Image", command=self.reset_app,font=("Times New Roman", 16))
    self.btn_upload_another.pack(pady=10)
    self.btn_upload_another.place(relx=0.3, rely=0.8, anchor=tk.CENTER)  # Place the button below the image label
    # Create a button for currency conversion
    self.btn_convert_currency = tk.Button(self, text="Convert to another Currency", command=lambda: self.convert_currency(predictions), font=("Times New Roman", 16))
    self.btn_convert_currency.pack(pady=10)
    self.btn_convert_currency.place(relx=0.7, rely=0.8, anchor=tk.CENTER)  # Place the button next to the "Upload Another Image" button


  # Rest of the code...
  def reset_app(self):
    self.Start_GUI()

if __name__ == "__main__":
  app = App()
  app.mainloop()


