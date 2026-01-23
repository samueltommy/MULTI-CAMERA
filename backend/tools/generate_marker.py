import cv2
import numpy as np

def generate_marker(id=0, size=200):
    # Load the dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Generate Marker
    img = cv2.aruco.generateImageMarker(aruco_dict, id, size)
    
    filename = f"marker_id{id}.png"
    cv2.imwrite(filename, img)
    print(f"Saved {filename} ({size}x{size}px). Print this and stick it on a flat surface!")

if __name__ == "__main__":
    generate_marker(0)
