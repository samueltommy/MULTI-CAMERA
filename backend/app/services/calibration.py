import cv2
import numpy as np
import time
import json
from app.services.camera import camera_manager
from app.database.session import SessionLocal
from app.database.models import Calibration
from app.services.fusion import fusion_service

class CalibrationService:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.src_pts = []
        self.dst_pts = []
        self.captured_ids = set()

    def reset(self):
        self.src_pts = []
        self.dst_pts = []
        self.captured_ids = set()

    def get_marker_centers(self, corners, ids):
        centers = {}
        if ids is None:
            return centers
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            c = corners[i][0]
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            centers[marker_id] = (cx, cy)
        return centers

    def capture_points(self):
        frame1, _ = camera_manager.get_raw_frame(0)
        frame2, _ = camera_manager.get_raw_frame(1)
        
        if frame1 is None or frame2 is None:
            return False, "Cameras not ready"

        corners1, ids1, _ = self.detector.detectMarkers(frame1)
        corners2, ids2, _ = self.detector.detectMarkers(frame2)

        centers1 = self.get_marker_centers(corners1, ids1)
        centers2 = self.get_marker_centers(corners2, ids2)
        
        common_ids = set(centers1.keys()) & set(centers2.keys())
        
        if not common_ids:
            return False, "No common markers found in both cameras"

        new_count = 0
        for cid in common_ids:
            self.src_pts.append(centers1[cid])
            self.dst_pts.append(centers2[cid])
            self.captured_ids.add(cid)
            new_count += 1
            
        return True, f"Captured {new_count} points. Total: {len(self.src_pts)}"

    def compute_and_save(self, name=None):
        if len(self.src_pts) < 4:
            return False, "At least 4 points required"

        src = np.array(self.src_pts, dtype=np.float32)
        dst = np.array(self.dst_pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        
        if H is not None:
            if not name:
                name = f"Web Calibration {time.strftime('%Y-%m-%d %H:%M:%S')}"
                
            # Update fusion service (which also saves to DB)
            fusion_service.set_homography(H, name=name)
            
            return True, "Calibration successful and saved"
        
        return False, "Homography computation failed"

calibration_service = CalibrationService()
