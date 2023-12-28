# Image_processing-currency-detection
## Breif 
The project aims to help identify a given bank note in Egyptian or Saudi Arabian currencies  using a Graphical User Interface **GUI** and prints to the user the value of the given note/bill then can also convert it's value to Dollars, Euros or Saudi Arabian for the Egyptian bills and vice versa for Saudi Arabian bills.

## How to run full model
1. Download the **GUI.py** and run this file
2. the following home screen will appear:    
![WhatsApp Image 2023-12-28 at 12 58 44 AM](https://github.com/Karim3036/Image_processing-currency-detection/assets/98956384/813fd653-57aa-4f7f-b49b-1beaf7f06fc4)  
Choose the desired option for uploading you image
3. After the image is uploaded the user sees the chosen image and has the option to recognise the image:
   ![WhatsApp Image 2023-12-28 at 12 58 50 AM](https://github.com/Karim3036/Image_processing-currency-detection/assets/98956384/12ae63f4-8c7f-4501-a43c-f48597d50b0b)
4. The model recognises the image (bill value) and displays it  
![WhatsApp Image 2023-12-28 at 12 58 57 AM](https://github.com/Karim3036/Image_processing-currency-detection/assets/98956384/7f153192-b8c5-4e02-a7fa-07747625f305)
5. The user can now choose to upload another image or to convert the current value to other currencies  
![WhatsApp Image 2023-12-28 at 12 59 02 AM](https://github.com/Karim3036/Image_processing-currency-detection/assets/98956384/7639558e-4438-4123-b394-048035fd09e3)
6. The result of the conversion will now be displayed and again the user has the option to choose their next move either to repeat a previous procedure or to close the GUI   
![WhatsApp Image 2023-12-28 at 12 59 08 AM](https://github.com/Karim3036/Image_processing-currency-detection/assets/98956384/683b19ca-1612-425c-9987-db9799bb7989)  

**Note:** to run in code check Testing_Full.ipynb in [trained_models](https://github.com/Karim3036/Image_processing-currency-detection/blob/main/trained_models)

**Pre-Requesits:** "needed to run most files so make sure they exist before running the program"
* commonfunctions.py
* Pre_Processing_Functions.py
* os
* pickle
* cv2
* skimage
* sklearn
* tkinter
* PIL
* numpy
* requests
* re  



## SVM results 
**full dataset**     
#SIFT alone = 32%  
#HOG alone = 77%  
#HOG+H+S+HS= 86%  
#H+S+HS+HOG+SIFT= 35 or 41%  
#H+HS+S+HOG+LBP=31%  
#HOG+H+S+HS(with process_image)=80.2%  

**33 imgs in dataset**     
H,S,HS,HOG (with pre-processing)=77.2%  
H,S,HS,HOG (NO pre-processing)=74.2%  
HS+HOG (with pre-processing)= 80.3%  
HS+HOG (No processing)=78.78%  
BOW (No pre)=33.3%  
BOW (pre)=37.88%  
H (No Pre)= 54.54%  
H (Pre)= 56.06%  
S (No Pre)= 27.27%  
S (Pre)= 34.85%  
HS (No Pre)= 56.06%  
HS (Pre)= 53.03%  
HOG (No Pre)= 77.27%  
HOG (Pre)= 80.30%  
GLCM (No pre)= 10.606%  
