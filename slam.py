import cv2
from display import Display

W = 3840 // 4
H = 2160 // 4



disp = Display(W, H)

def process_frame(img):
	img = cv2.resize(img, (W,H))
	print(img.shape)
	disp.paint(img)


if __name__=='__main__':

	cap = cv2.VideoCapture('test.mp4')

	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True: 
			process_frame(frame)
		else:
			break