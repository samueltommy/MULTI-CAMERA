import cv2
import numpy as np
import time
import os

# --- CONFIGURATION ---
RTSP_1 = 'rtsp://admin:admin123456@192.168.1.8:554/ch=1?subtype=0' # Top/Camera 1
RTSP_2 = 'rtsp://admin:admin123456@192.168.1.11:554/ch=1?subtype=0' # Side/Camera 2
OUTPUT_FILE = 'homography_top_to_side.npy'

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
    print("Connecting to cameras...")
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
            print("Waiting for streams...")
            time.sleep(0.1)
            continue

        # Resize for display (optional, keeps windows manageable)
        display1 = cv2.resize(frame1, (640, 480))
        display2 = cv2.resize(frame2, (640, 480))

        # Detect in both frames
        corners1, ids1, _ = detector.detectMarkers(frame1)
        corners2, ids2, _ = detector.detectMarkers(frame2)

        # Draw all detected markers for visual feedback
        if ids1 is not None:
            cv2.aruco.drawDetectedMarkers(display1, corners1, ids1)
        if ids2 is not None:
            cv2.aruco.drawDetectedMarkers(display2, corners2, ids2)

        # Find Common Markers (Visible in BOTH)
        common_ids = []
        if ids1 is not None and ids2 is not None:
            # Intersection of IDs found in both cameras
            common_ids = np.intersect1d(ids1.flatten(), ids2.flatten())

        # Visual feedback: Draw Green Circle on Common Markers
        centers1 = get_marker_centers(corners1, ids1)
        centers2 = get_marker_centers(corners2, ids2)

        # Scale factors if we resized display
        sx1 = display1.shape[1] / frame1.shape[1]
        sy1 = display1.shape[0] / frame1.shape[0]
        sx2 = display2.shape[1] / frame2.shape[1]
        sy2 = display2.shape[0] / frame2.shape[0]

        for cid in common_ids:
            # Draw on display 1
            if cid in centers1:
                c = centers1[cid]
                # Scale to display coords
                dc = (int(c[0]*sx1), int(c[1]*sy1))
                cv2.circle(display1, dc, 8, (0, 255, 0), -1) # Green Dot
            
            # Draw on display 2
            if cid in centers2:
                c = centers2[cid]
                dc = (int(c[0]*sx2), int(c[1]*sy2))
                cv2.circle(display2, dc, 8, (0, 255, 0), -1)

        # Text Info
        status_text = f"Matches Now: {len(common_ids)}"
        total_text = f"Total Collected: {len(all_src_pts)}"
        cv2.putText(display1, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display1, total_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Camera 1 (Top)", display1)
        cv2.imshow("Camera 2 (Side)", display2)

        key = cv2.waitKey(1) & 0xFF

        # CAPTURE
        if key == 32: # Space
            if len(common_ids) > 0:
                added_count = 0
                for cid in common_ids:
                    # Get full resolution coordinates
                    pt1 = centers1[cid]
                    pt2 = centers2[cid]
                    
                    # Add to list
                    all_src_pts.append(list(pt1))
                    all_dst_pts.append(list(pt2))
                    added_count += 1
                
                print(f"Captured {added_count} new points! Total: {len(all_src_pts)}")
            else:
                print("No common markers visible to capture.")

        # RESET
        elif key == ord('r'):
            all_src_pts = []
            all_dst_pts = []
            print("Reset all points.")

        # COMPUTE
        elif key == ord('c'):
            if len(all_src_pts) < 4:
                print(f"Need at least 4 points. You have {len(all_src_pts)}.")
            else:
                print("Calculating Homography...")
                src = np.array(all_src_pts)
                dst = np.array(all_dst_pts)
                
                H, status = cv2.findHomography(src, dst)
                print("Matrix H:\n", H)
                np.save(OUTPUT_FILE, H)
                print(f"Saved to {OUTPUT_FILE}. Ready to run app.")
                break

        elif key == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()