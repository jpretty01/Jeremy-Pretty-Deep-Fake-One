import cv2
import numpy as np

import os

# Get the current working directory of where the image is
Selfie_01 = os.path.join(os.path.dirname(__file__), 'Jeremy1.jpeg')
Web_01 = os.path.join(os.path.dirname(__file__), 'web_img.jpg')


# Load the selfie and web image
selfie = cv2.imread(Selfie_01)
web_img = cv2.imread(Web_01)

# Convert the images to grayscale
selfie_gray = cv2.cvtColor(selfie, cv2.COLOR_BGR2GRAY)
web_img_gray = cv2.cvtColor(web_img, cv2.COLOR_BGR2GRAY)

# Detect the faces in the images
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
selfie_faces = face_cascade.detectMultiScale(selfie_gray, scaleFactor=1.1, minNeighbors=5)
web_img_faces = face_cascade.detectMultiScale(web_img_gray, scaleFactor=1.1, minNeighbors=5)

# Get the largest face from each image
selfie_face = max(selfie_faces, key=lambda x: x[2] * x[3])
web_img_face = max(web_img_faces, key=lambda x: x[2] * x[3])

# Extract the face regions from the images
selfie_face_region = selfie[selfie_face[1]:selfie_face[1] + selfie_face[3], selfie_face[0]:selfie_face[0] + selfie_face[2]]
web_img_face_region = web_img[web_img_face[1]:web_img_face[1] + web_img_face[3], web_img_face[0]:web_img_face[0] + web_img_face[2]]

# Resize the selfie face region to match the size of the web image face region
selfie_face_region_resized = cv2.resize(selfie_face_region, (web_img_face_region.shape[1], web_img_face_region.shape[0]))

# Replace the web image face region with the selfie face region
web_img[web_img_face[1]:web_img_face[1] + web_img_face[3], web_img_face[0]:web_img_face[0] + web_img_face[2]] = selfie_face_region_resized

# Save the deepfake image
cv2.imwrite("deepfake.jpg", web_img)
