from multiprocessing import Process, Queue
import OpenGL.GL as gl
import pangolin
import numpy as np

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
		self.viewer_init(1024, 768)
		while 1: 
			self.viewer_refresh(q)


	def viewer_init(self, w, h):

		pangolin.CreateWindowAndBind('Main', w, h)
		gl.glEnable(gl.GL_DEPTH_TEST)

		# Define Projection and initial ModelView matrix
		self.scam = pangolin.OpenGlRenderState(
			pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 1000),
			pangolin.ModelViewLookAt(0, -10, -8,
									0, 0, 0,
									0, -1, 0))
		self.handler = pangolin.Handler3D(self.scam)

		# Create Interactive Window 
		self.dcam = pangolin.CreateDisplay()
		self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -w/h)
		self.dcam.SetHandler(self.handler)

		self.state = None

	def viewer_refresh(self, q):
		if self.state is None or not q.empty():
			self.state = q.get()

		# turn state into points
		#ppts = np.array([d[:3, 3] for d in self.state[0]])
		spts = np.array(self.state[1])

		gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
		gl.glClearColor(1.0, 1.0, 1.0, 1.0)
		self.dcam.Activate(self.scam)

		# draw poses
		gl.glColor3f(0.0, 1.0, 0.0)
		pangolin.DrawCameras(self.state[0])

		#draw keypoints
		gl.glPointSize(2)
		gl.glColor3f(1.0, 0.0, 0.0)
		pangolin.DrawPoints(spts)

		pangolin.FinishFrame()


	def display(self):
		poses, pts = [], []
		for f in self.frames:
			poses.append(f.pose)
		for p in self.points:
			pts.append(p.pt)
		self.q.put((np.array(poses), np.array(pts)))
