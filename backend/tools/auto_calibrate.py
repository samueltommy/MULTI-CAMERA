import cv2
import numpy as np
import time
import os
import sys

# Add parent dir to path to allow importing app config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.core.config import settings
    from app.database.session import SessionLocal
    from app.database.models import Calibration
    import json
    RTSP_1 = settings.RTSP_URL_1
    RTSP_2 = settings.RTSP_URL_2
except ImportError:
    print("Could not import settings or DB models. DB saving will be disabled.")
    RTSP_1 = os.environ.get('RTSP_URL_1', 'rtsp://admin:admin123456@192.168.1.105:554/ch=1?subtype=1')
    RTSP_2 = os.environ.get('RTSP_URL_2', 'rtsp://admin:admin123456@192.168.1.110:554/ch=1?subtype=1')
    SessionLocal = None

# Use the standard dictionary (Ids 0-49 are available)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

def get_marker_centers(corners, ids):
    """
    Returns a dictionary {marker_id: (center_x, center_y)}
    """
    centers = {}
    if ids is None:
        return centers
    
    # Flatten ids list
    ids = ids.flatten()
    
    for i, marker_id in enumerate(ids):
        # corners[i][0] is the list of 4 points for this marker
        c = corners[i][0]
        cx = int(np.mean(c[:, 0]))
        cy = int(np.mean(c[:, 1]))
        centers[marker_id] = (cx, cy)
    return centers

def auto_calibrate(required_points=5, headless=False):
    print(f"Connecting to cameras:\nCam 1: {RTSP_1}\nCam 2: {RTSP_2}")
    cap1 = cv2.VideoCapture(RTSP_1)
    cap2 = cv2.VideoCapture(RTSP_2)

    # Buffers to store valid points
    all_src_pts = [] # Camera 1
    all_dst_pts = [] # Camera 2
    
    # Track which markers we've already captured to get unique positions if possible
    captured_ids = set()

    print("\n--- AUTO CALIBRATION ---")
    print(f"Goal: Capture at least {required_points} matching points.")
    
    if not headless:
        print("1. Place ArUco markers (ID 0-49) on the floor.")
        print("2. Ensure the SAME markers are visible in BOTH cameras.")
        print("3. Press 'SPACE' to capture or wait for auto-capture.")
        print("4. Press 'q' to quit.")

    last_capture_time = 0
    
    try:
        while len(all_src_pts) < required_points:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                time.sleep(0.5)
                continue

            # Detect markers
            corners1, ids1, _ = detector.detectMarkers(frame1)
            corners2, ids2, _ = detector.detectMarkers(frame2)

            centers1 = get_marker_centers(corners1, ids1)
            centers2 = get_marker_centers(corners2, ids2)
            common_ids = set(centers1.keys()) & set(centers2.keys())

            # Auto-capture logic: if we see new markers or some time has passed
            current_time = time.time()
            if common_ids and (current_time - last_capture_time > 2.0):
                new_ids = common_ids - captured_ids
                if new_ids or (current_time - last_capture_time > 5.0):
                    for cid in common_ids:
                        all_src_pts.append(centers1[cid])
                        all_dst_pts.append(centers2[cid])
                        captured_ids.add(cid)
                    last_capture_time = current_time
                    print(f"Auto-captured {len(common_ids)} points. Total: {len(all_src_pts)}/{required_points}")

            if not headless:
                # Visualization for manual mode
                frame1_disp = cv2.resize(frame1, (640, 480))
                frame2_disp = cv2.resize(frame2, (640, 480))
                cv2.aruco.drawDetectedMarkers(frame1_disp, corners1, ids1)
                cv2.aruco.drawDetectedMarkers(frame2_disp, corners2, ids2)
                
                status_text = f"Captured: {len(all_src_pts)}/{required_points}"
                cv2.putText(frame1_disp, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                combined = np.hstack((frame1_disp, frame2_disp))
                cv2.imshow("Calibration", combined)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): return False
                if key == ord(' '):
                    for cid in common_ids:
                        all_src_pts.append(centers1[cid])
                        all_dst_pts.append(centers2[cid])
                    print(f"Manual captured. Total: {len(all_src_pts)}")

        # Compute and Save
        src = np.array(all_src_pts, dtype=np.float32)
        dst = np.array(all_dst_pts, dtype=np.float32)
        H, _ = cv2.findHomography(src, dst, method=0)
        
        if H is not None:
            if SessionLocal:
                db = SessionLocal()
                db.query(Calibration).update({Calibration.is_active: False})
                new_cal = Calibration(
                    name=f"Auto Calibration {time.strftime('%Y-%m-%d %H:%M:%S')}",
                    matrix_json=json.dumps(H.tolist()),
                    is_active=True
                )
                db.add(new_cal)
                db.commit()
                print(f"Successfully saved calibration (ID: {new_cal.id})")
                db.close()
                return True
    finally:
        cap1.release()
        cap2.release()
        if not headless: cv2.destroyAllWindows()
    return False

def main():
    required = int(os.environ.get('CALIBRATION_POINTS', 5))
    auto_calibrate(required_points=required, headless=False)

if __name__ == "__main__":
    main()
