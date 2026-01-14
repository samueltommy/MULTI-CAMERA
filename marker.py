import cv2
import numpy as np

# Load the dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Generate Marker ID 0, 200x200 pixels
img = cv2.aruco.generateImageMarker(aruco_dict, 0, 200)

cv2.imwrite("marker_id0.png", img)
print("Saved marker_id0.png. Print this and stick it on a flat cardboard!")