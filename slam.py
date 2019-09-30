import cv2
from display import Display
import time
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
import g2o
from multiprocessing import Process, Queue
import OpenGL.GL as gl
import pangolin


# camera intrinsics
W = 3840 // 4
H = 2160 // 4
F = 270			# focal length

K = np.array([[F, 0, W//2],[0,F,H//2], [0,0,1]])


#global map
class Map(object):
	def __init__(self):
		self.frames = []
		self.points = []
		#self.viewer_init()
		self.state=None

		self.q = Queue()
		viewer = Process(target=self.viewer_thread, args=(self.q, ))
		viewer.daemon = True
		viewer.start()


	def viewer_thread(self, q):
		self.viewer_init()
		while 1: 
			self.viewer_refresh(q)


	def viewer_init(self,):

		pangolin.CreateWindowAndBind('Main', 640, 480)
		gl.glEnable(gl.GL_DEPTH_TEST)

		# Define Projection and initial ModelView matrix
		self.scam = pangolin.OpenGlRenderState(
			pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100),
			pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
		self.handler = pangolin.Handler3D(self.scam)

		# Create Interactive Window 
		self.dcam = pangolin.CreateDisplay()
		self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
		self.dcam.SetHandler(self.handler)

		self.state = None

	def viewer_refresh(self, q):
		if self.state is None or not q.empty():
			self.state = q.get()
		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glClearColor(1.0, 1.0, 1.0, 1.0)
		self.dcam.Activate(self.scam)

		gl.glPointSize(10)
		gl.glColor3f(0.0, 1.0, 0.0)
		pangolin.DrawPoints(np.array([d[:3, 3] for d in self.state[0]]))
		
		gl.glPointSize(2)
		gl.glColor3f(1.0, 0.0, 0.0)
		pangolin.DrawPoints(np.array(self.state[1]))

		pangolin.FinishFrame()


	def display(self):
		poses, pts = [], []
		for f in self.frames:
			poses.append(f.pose)
		for p in self.points:
			pts.append(p.pt)
		self.q.put((poses, pts))

# main classes
disp = Display(W, H)
mapp = Map()

class Point(object):
	# A point is a 3-D point in world
	# Each point is observed in multiple frames

	def __init__(self, mapp, loc):
		self.pt = loc
		self.frames = []
		self.idxs = []
		self.id = len(mapp.points)
		mapp.points.append(self)


	def add_observation(self, frame, idx):
		self.frames.append(frame)
		self.idxs.append(idx)



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


	print(f1.pose)
	print(pts4d)

	for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
		#u1, v1 = map(lambda x: int(round(x)), pt1)
		#u2, v2 = map(lambda x: int(round(x)), pt2)

		u1, v1 = denormalize(K, pt1)
		u2, v2 = denormalize(K, pt2)

		cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
		cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

	#print(img.shape)
	disp.paint(img)

	mapp.display()


if __name__=='__main__':

	cap = cv2.VideoCapture('test_countryroad.mp4')

	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True: 
			process_frame(frame)
		else:
			break