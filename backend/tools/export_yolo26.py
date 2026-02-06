from ultralytics import YOLO
import os

def export_model():
    # This script assumes you have placed 'yolo26s-seg.pt' in the models/ directory
    model_name = "yolo26s-seg"
    pt_path = os.path.join("..", "models", f"{model_name}.pt")
    
    if not os.path.exists(pt_path):
        print(f"Error: {pt_path} not found. Please place the file there first.")
        return

    print(f"Loading {pt_path}...")
    model = YOLO(pt_path)
    
    print(f"Exporting to OpenVINO...")
    # Export to OpenVINO format
    # The output will be in a folder named 'yolo26s-seg_openvino_model'
    model.export(format="openvino", imgsz=480, dynamic=False)
    
    print(f"Export complete. The model is ready at: ../models/{model_name}_openvino_model/")

if __name__ == "__main__":
    export_model()
