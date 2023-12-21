# Image_processing-currency-detection

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
GLCM (No pre, Co pilot)= 10.606%
