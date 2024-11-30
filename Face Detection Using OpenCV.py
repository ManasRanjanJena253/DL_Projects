# Importing Dependencies
import cv2 as cv
import numpy as np

# Loading an image into variable
img = cv.imread("C:/Users/mranj/OneDrive/Desktop/WhatsApp Image 2024-11-23 at 19.16.50_4f098116.jpg")
resized_img = cv.resize(src = img, dsize = (600, 600), fx = 200, fy = 200, interpolation = cv.INTER_AREA)
cv.imshow('Person', img)  # Showing the image.
cv.imshow('Resized Image', resized_img)


# Haarcascade is a classifier model that already exist and help us in detecting various faces.
# Haarcascade don't detect any type of color in the image and try to find objects similar to the human face based on its edges.

gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)  # Converting the image into gray scale as our classifier don't need an image.
cv.imshow('Gray', gray)

haar_cascade = cv.CascadeClassifier('haarcascad for face detection')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor  = 1.1, minNeighbors = 3)

# faces_rect will have a list dtype and will store the faces found in the image. We can get the number of faces found in the image by printing the len of the face_rect.
print('No. of faces found = ', len(faces_rect))
print(f"The contents inside the faces_rect variable :: {faces_rect}")

for (x, y, w, h) in faces_rect:
    cv.rectangle(resized_img, (x,y), (x+w, y+h), (0,255,0), thickness = 3)

cv.imshow('Detected Faces', resized_img)

# Haarcascade face detection method also detects a lot of noise due to which it may make wrong predictions or may identify wrong objects as images. We can decrease this by increasing minNeighbours.
# But increasing minNeighbours also affects the faces found in a large group of people. Thus, minNeighbours should be higher for images with less crowd of people and lower if more crowd of people.
# The less the value of minNeighbours is , the more prone to noise our model is.
# For all these reasons we don't use harrcascade for advanced projects.
# We can do the same face detection in case of videos also by applying this classification for each frame.
