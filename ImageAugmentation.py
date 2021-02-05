# Code written by : Aryan Verma
# Video Explanation Link : https://youtu.be/GVdH9BI_PLQ

import cv2
import numpy as np


MIN_MATCHES = 20
detector = cv2.ORB_create(nfeatures=5000)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=100)
flann = cv2.FlannBasedMatcher(index_params,search_params)


def load_input():
	input_image = cv2.imread('meluha_from_webcam.jpg')
	augment_image = cv2.imread('mask.jpg')

	input_image = cv2.resize(input_image, (300,400),interpolation=cv2.INTER_AREA)
	augment_image = cv2.resize(augment_image, (300,400))
	gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
	# find the keypoints with ORB
	keypoints, descriptors = detector.detectAndCompute(gray_image, None)

	return gray_image,augment_image,keypoints, descriptors


def compute_matches(descriptors_input, descriptors_output):
	# Match descriptors
	if(len(descriptors_output)!=0 and len(descriptors_input)!=0):
		matches = flann.knnMatch(np.asarray(descriptors_input,np.float32),np.asarray(descriptors_output,np.float32),k=2)
		good = []
		for m,n in matches:
			if m.distance < 0.69*n.distance:
				good.append(m)
		return good
	else:
		return None


if __name__=='__main__':

	#Getting Information form the Input image
	input_image, aug_image, input_keypoints, input_descriptors = load_input()

	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()

	while(ret):
		ret, frame = cap.read()
		if(len(input_keypoints)<MIN_MATCHES):
			continue
		frame = cv2.resize(frame, (600,450))
		frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		output_keypoints, output_descriptors = detector.detectAndCompute(frame_bw, None)
		matches = compute_matches(input_descriptors, output_descriptors)
		if(matches!=None):
			if(len(matches)>10):
				src_pts = np.float32([ input_keypoints[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
				dst_pts = np.float32([ output_keypoints[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

				#Finally find the homography matrix
				M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
				#matchesMask = mask.ravel().tolist()
				pts = np.float32([ [0,0],[0,399],[299,399],[299,0] ]).reshape(-1,1,2)
				dst = cv2.perspectiveTransform(pts,M)
				M_aug = cv2.warpPerspective(aug_image, M, (600,450))

				#getting the frame ready for addition operation with Mask Image
				frameb = cv2.fillConvexPoly(frame,dst.astype(int),0)
				Final = frameb+M_aug
				
				#output_final = cv2.polylines(frame,[np.int32(dst)],True,255,3, cv2.LINE_AA)
				cv2.imshow('Final Output', Final)
				#cv2.imshow('Finallli', Final)
			else:
				cv2.imshow('Final Output', frame)
		else:
			cv2.imshow('Final Output', frame)
		key = cv2.waitKey(15)
		if(key==27):
			break
