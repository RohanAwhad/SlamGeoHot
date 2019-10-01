#!/usr/bin/env python3

import os
import cv2
from display import Display
import time
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
import g2o
from pointmap import Map, Point



# camera intrinsics
W = 3840 // 4
H = 2160 // 4
F = 270			# focal length

K = np.array([[F, 0, W//2],[0,F,H//2], [0,0,1]])
Kinv = np.linalg.inv(K)


# main classes
mapp = Map()
disp = Display(W, H) if os.getenv("D2D") is not None else None



def triangulate(pose1, pose2, pts1, pts2):
	return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T

def process_frame(img):
	img = cv2.resize(img, (W,H))
	frame = Frame(img, mapp, K)
	
	if frame.id == 0:
		return

	f1 = mapp.frames[-1]
	f2 = mapp.frames[-2]

	idx1, idx2, Rt = match_frames(f1, f2)
	f1.pose = np.dot(Rt, f2.pose)
	
	pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
	
	#homogeneous 3-Dcoords
	pts4d /= pts4d[:, 3:]

	#rejecting pts without good parallax
	#reject pts behind cam
	good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)
	
	#print(sum(good_pts4d), len(good_pts4d))
	
	for i, p in enumerate(pts4d):
		if not good_pts4d[i]: 
			continue
		pt = Point(mapp, p)
		pt.add_observation(f1, idx1[i])
		pt.add_observation(f2, idx2[i])


	#print(f1.pose)
	#print(pts4d)

	for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
		#u1, v1 = map(lambda x: int(round(x)), pt1)
		#u2, v2 = map(lambda x: int(round(x)), pt2)

		u1, v1 = denormalize(K, pt1)
		u2, v2 = denormalize(K, pt2)

		cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
		cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

	#print(img.shape)
	
	# 2-D display
	if disp is not None: disp.paint(img)

	#3-D display
	mapp.display()


if __name__=='__main__':

	cap = cv2.VideoCapture('test_countryroad.mp4')

	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True: 
			process_frame(frame)
		else:
			break