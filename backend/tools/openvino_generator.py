from ultralytics import YOLO

# Load the new YOLO26 small segmentation model
model = YOLO("yolo26s-seg.pt")

# Export to OpenVINO format (as used in your config)
# This will create the folder: yolo26s-seg_openvino_model/
model.export(format="openvino")