import cv2
from display import Display
import time
import numpy as np
from extractor import Extractor

W = 3840 // 4
H = 2160 // 4

disp = Display(W, H)
F = 270			# focal length
K = np.array([[F, 0, W//2],[0,F,H//2], [0,0,1]])
fe = Extractor(K)

def process_frame(img):
	img = cv2.resize(img, (W,H))
	matches, pose = fe.extract(img)

	for pt1, pt2 in matches:
		#u1, v1 = map(lambda x: int(round(x)), pt1)
		#u2, v2 = map(lambda x: int(round(x)), pt2)

		u1, v1 = fe.denormalize(pt1)
		u2, v2 = fe.denormalize(pt2)



		cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
		cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

	#print(img.shape)
	disp.paint(img)


if __name__=='__main__':

	cap = cv2.VideoCapture('test_countryroad.mp4')

	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True: 
			process_frame(frame)
		else:
			break