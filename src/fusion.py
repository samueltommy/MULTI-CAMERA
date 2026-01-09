import cv2
import numpy as np
from typing import List, Tuple, Dict

def compute_homography(src_pts: List[Tuple[float, float]], dst_pts: List[Tuple[float, float]]):
    """Compute homography H that maps src_pts -> dst_pts. Points are list of (x,y)."""
    src = np.array(src_pts, dtype=np.float32)
    dst = np.array(dst_pts, dtype=np.float32)
    H, status = cv2.findHomography(src, dst, method=0)
    return H, status

def project_point(H: np.ndarray, pt: Tuple[float, float]) -> Tuple[float, float]:
    """Project a single (x,y) point using homography H."""
    if H is None:
        return None
    src = np.array([ [pt[0], pt[1], 1.0] ], dtype=np.float32).T
    dst = H.dot(src)
    if dst[2,0] == 0:
        return (float(dst[0,0]), float(dst[1,0]))
    x = float(dst[0,0] / dst[2,0])
    y = float(dst[1,0] / dst[2,0])
    return (x, y)

def match_detections(dets_top: List[Dict], dets_side: List[Dict], H: np.ndarray, max_dist_px: float=60.0):
    """Match detections from top view to side view using homography.

    dets_top: list of dicts containing at least 'center':(x,y)
    dets_side: list of dicts containing at least 'bottom_center':(x,y)
    Returns list of tuples (i_top, i_side, dist)
    """
    matches = []
    if H is None or not dets_top or not dets_side:
        return matches

    # precompute projected top centers into side image
    proj = []
    proj = []
    for d in dets_top:
        c = d.get('bottom_center') # Ensure you are using bottom_center!
        if c is None:
            proj.append(None)
        else:
            proj.append(project_point(H, (c[0], c[1])))

    for i_top, p in enumerate(proj):
        if p is None:
            continue
            
        # Get the class of the object in Camera 1
        cls_top = dets_top[i_top].get('cls')

        best_j = None
        best_d = None
        
        for j, sd in enumerate(dets_side):
            # --- NEW CHECK: STRICT CLASS MATCHING ---
            cls_side = sd.get('cls')
            if cls_top != cls_side:
                continue  # Skip if one is a Person and the other is a Chair
            # ----------------------------------------

            bc = sd.get('bottom_center')
            if bc is None:
                continue
            
            dx = float(p[0] - bc[0])
            dy = float(p[1] - bc[1])
            dist = (dx*dx + dy*dy) ** 0.5
            
            if best_d is None or dist < best_d:
                best_d = dist
                best_j = j
                
        if best_j is not None and best_d is not None and best_d <= max_dist_px:
            matches.append((i_top, best_j, best_d))

    return matches
