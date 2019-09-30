import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform


IRt = np.eye(4)

def add_ones(x):
	#print(x.shape)
	ret = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
	#print(ret)
	return ret

def extractRt(E):
	W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
	U, d, Vt = np.linalg.svd(E)
	if np.linalg.det(U) < 0:
		U *= -1.0
	if np.linalg.det(Vt) < 0:
		Vt *= -1.0
	
	R = np.dot(np.dot(U, W), Vt)
	if np.sum(R.diagonal()) < 0:
		R = np.dot(np.dot(U, W.T), Vt)

	t = U[:, 2]
	ret = np.eye(4)
	ret[:3, :3] = R
	ret[:3, 3] = t
	print(ret)
	return ret

		

def extract(img):
	orb = cv2.ORB_create()
	#detection
	pts = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
	
	#extraction 
	kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
	kps, des = orb.compute(img,kps)

	return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def normalize(Kinv, pts):
	return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]

def denormalize(K, pt):
	ret = np.dot(K, np.array([pt[0], pt[1], 1.0]))
	ret /= ret[2]
	return int(round(ret[0])), int(round(ret[1]))


def match_frames(f1, f2):
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(f1.des, f2.des, k=2)

	# Lowe's ratio Test
	ret = []
	idx1, idx2 = [], []

	for m, n in matches:
		if m.distance < 0.75*n.distance:
			idx1.append(m.queryIdx)
			idx2.append(m.trainIdx)

			kp1 = f1.pts[m.queryIdx]
			kp2 = f2.pts[m.trainIdx]
			ret.append((kp1, kp2))

	# filter
	if len(ret) >= 8:
		ret = np.array(ret)
		idx1 = np.array(idx1)
		idx2 = np.array(idx2)
		

		#fit matrix
		model, inliers = ransac((ret[:, 0], ret[:, 1]),
								#FundamentalMatrixTransform,
								EssentialMatrixTransform,
								min_samples=8,
								#residual_threshold=1,
								residual_threshold=.005,
								max_trials=200)


		# ignore outliers
		#ret = ret[inliers]


		Rt = extractRt(model.params)
		

	#return
	return idx1[inliers], idx2[inliers], Rt



class Frame(object):
	def __init__(self, img, K):
		self.pts, self.des = extract(img)
		self.K = K
		self.Kinv = np.linalg.inv(self.K)
		self.pose = IRt
		self.pts = normalize(self.Kinv, self.pts)

		

