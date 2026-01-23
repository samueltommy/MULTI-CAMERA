import numpy as np
import cv2

def compute_homography(src_pts, dst_pts):
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, status = cv2.findHomography(src, dst, method=0)
    return H, status

def project_point(H, pt):
    if H is None: return None
    src = np.array([[pt[0], pt[1], 1.0]], dtype=np.float32).T
    dst = H.dot(src)
    if dst[2,0] == 0: return (float(dst[0,0]), float(dst[1,0]))
    return (float(dst[0,0]/dst[2,0]), float(dst[1,0]/dst[2,0]))

def compute_distance(p1, p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
