import cv2
import numpy as np
import time
import os
import sys

# Add parent dir to path to allow importing app config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.core.config import settings
    RTSP_1 = settings.RTSP_URL_1
    RTSP_2 = settings.RTSP_URL_2
    OUTPUT_FILE = settings.HOMOGRAPHY_PATH
except ImportError:
    print("Could not import settings from app.core.config. Using defaults/env vars.")
    RTSP_1 = os.environ.get('RTSP_URL_1', 'rtsp://admin:admin123456@192.168.1.105:554/ch=1?subtype=1')
    RTSP_2 = os.environ.get('RTSP_URL_2', 'rtsp://admin:admin123456@192.168.1.110:554/ch=1?subtype=1')
    OUTPUT_FILE = os.environ.get('HOMOGRAPHY_PATH', '../homography_top_to_side.npy')

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

def main():
    print(f"Connecting to cameras:\nCam 1: {RTSP_1}\nCam 2: {RTSP_2}")
    cap1 = cv2.VideoCapture(RTSP_1)
    cap2 = cv2.VideoCapture(RTSP_2)

    # Buffers to store valid points
    all_src_pts = [] # Camera 1
    all_dst_pts = [] # Camera 2

    print("\n--- INSTRUCTIONS ---")
    print("1. Place ANY ArUco markers (ID 0-49) on the floor.")
    print("2. Ensure the SAME markers are visible in BOTH cameras.")
    print("3. Press 'SPACE' to capture all matching markers currently visible.")
    print("4. Press 'c' to calculate and save.")
    print("5. Press 'r' to reset points and start over.")
    print("6. Press 'q' to quit.\n")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Failed to read from one or both cameras. Retrying...")
            time.sleep(0.5)
            continue

        # Resize for display if too large
        frame1_disp = cv2.resize(frame1, (640, 480))
        frame2_disp = cv2.resize(frame2, (640, 480))

        # Detect markers in both frames
        corners1, ids1, _ = detector.detectMarkers(frame1)
        corners2, ids2, _ = detector.detectMarkers(frame2)

        # Draw detected markers
        cv2.aruco.drawDetectedMarkers(frame1_disp, corners1, ids1)
        cv2.aruco.drawDetectedMarkers(frame2_disp, corners2, ids2)

        # Display status
        cv2.putText(frame1_disp, f"Captured Points: {len(all_src_pts)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame2_disp, "Press SPACE to capture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Retrieve centers
        centers1 = get_marker_centers(corners1, ids1)
        centers2 = get_marker_centers(corners2, ids2)

        # Find common IDs currently visible
        common_ids = set(centers1.keys()) & set(centers2.keys())
        
        # Draw circles for common markers to indicate readiness
        for cid in common_ids:
            # Scale back to display coordinates roughly for visualization (imprecise but helpful)
            # Actually detecting on full frame, displaying resized. Detections are full frame coords.
            # We won't draw on display frame precisely unless we scale coordinates.
            pass

        combined = np.hstack((frame1_disp, frame2_disp))
        cv2.imshow("Auto Calibrate (Left: Cam1, Right: Cam2)", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '): # Spacebar to capture
            if not common_ids:
                print("No common markers visible!")
            else:
                count = 0
                for cid in common_ids:
                    p1 = centers1[cid]
                    p2 = centers2[cid]
                    all_src_pts.append(p1)
                    all_dst_pts.append(p2)
                    count += 1
                print(f"Captured {count} matches. Total: {len(all_src_pts)}")
        elif key == ord('r'):
            all_src_pts = []
            all_dst_pts = []
            print("Reset points.")
        elif key == ord('c'):
            if len(all_src_pts) < 4:
                print("Need at least 4 points to compute homography.")
            else:
                src = np.array(all_src_pts, dtype=np.float32)
                dst = np.array(all_dst_pts, dtype=np.float32)
                
                H, status = cv2.findHomography(src, dst, method=0)
                if H is not None:
                    print(f"\nHomography computed:\n{H}")
                    np.save(OUTPUT_FILE, H)
                    print(f"Saved to {OUTPUT_FILE}")
                else:
                    print("Homography computation failed.")

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
