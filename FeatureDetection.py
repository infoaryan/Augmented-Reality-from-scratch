#Code written by : Aryan Verma (infoaryan)
#Full explanation video link : https://youtu.be/lU4zgDe1x6Y 

import cv2
import numpy as np

#Getting the Image ready for feature detection
input_image = cv2.imread('meluha.jpg')
input_image = cv2.resize(input_image, (400,550),interpolation=cv2.INTER_AREA)
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# Initiate ORB object
orb = cv2.ORB_create(nfeatures=1000)

# find the keypoints with ORB
keypoints, descriptors = orb.detectAndCompute(gray_image, None)

# draw only the location of the keypoints without size or
final_keypoints = cv2.drawKeypoints(gray_image, keypoints,input_image,(0,255,0))

cv2.imshow('ORB keypoints', final_keypoints)
cv2.waitKey()
