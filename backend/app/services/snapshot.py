import os
import cv2
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from app.core.config import settings
from app.database.session import SessionLocal
from app.database.models import FusedObject
from datetime import datetime

class SnapshotService:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        os.makedirs(settings.SNAPSHOT_DIR, exist_ok=True)

    def _save_file(self, path, img):
        cv2.imwrite(path, img)

    def save_fusion_snapshot(self, track_obj, frame_top, frame_side=None):
        if track_obj.get('has_snapshot'):
            return
        
        runtime_id = track_obj['id']
        ts = int(time.time())
        
        # 1. NEW: Create a dedicated subfolder for this ID
        id_folder_rel = f"id_{runtime_id}"
        id_folder_abs = os.path.join(settings.SNAPSHOT_DIR, id_folder_rel)
        os.makedirs(id_folder_abs, exist_ok=True) #
        
        # 2. Handle Top Camera
        top_crop = self._crop(frame_top, track_obj['top']['box']) if track_obj.get('top') else None
        top_path_rel = os.path.join(id_folder_rel, f"top_{ts}.jpg") #
        
        if top_crop is not None:
            top_path_abs = os.path.join(settings.SNAPSHOT_DIR, top_path_rel)
            self.executor.submit(self._save_file, top_path_abs, top_crop) #

        # 3. Handle Side Camera
        side_path_rel = None
        if track_obj.get('is_fused') and track_obj.get('side') and frame_side is not None:
            side_crop = self._crop(frame_side, track_obj['side']['box'])
            if side_crop is not None:
                side_path_rel = os.path.join(id_folder_rel, f"side_{ts}.jpg") #
                side_path_abs = os.path.join(settings.SNAPSHOT_DIR, side_path_rel)
                self.executor.submit(self._save_file, side_path_abs, side_crop) #

        # 4. Update Database with the new relative paths
        self.executor.submit(self._update_db, runtime_id, top_path_rel, side_path_rel, track_obj) #
        
        track_obj['has_snapshot'] = True
        track_obj['snapshot_paths'] = (top_path_rel, side_path_rel) #

    def _crop(self, frame, box):
        if frame is None or not box: return None
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1c, y1c = max(0, min(w-1, x1)), max(0, min(h-1, y1))
        x2c, y2c = max(0, min(w-1, x2)), max(0, min(h-1, y2))
        if y2c > y1c and x2c > x1c:
            return frame[y1c:y2c, x1c:x2c].copy()
        return None

    def _update_db(self, runtime_id, top_path, side_path, track_data):
        db = SessionLocal()
        try:
            obj = db.query(FusedObject).filter(FusedObject.track_id == runtime_id).first()
            if not obj:
                obj = FusedObject(track_id=runtime_id)
                db.add(obj)
            
            obj.snapshot_top = top_path
            obj.snapshot_side = side_path
            
            # Save Metadata (Dengan konversi ke float standar Python)
            if track_data.get('top') and track_data['top'].get('center'):
                obj.top_center_x = float(track_data['top']['center'][0])
                obj.top_center_y = float(track_data['top']['center'][1])
                
            if track_data.get('side') and track_data['side'].get('bottom_center'):
                obj.side_center_x = float(track_data['side']['bottom_center'][0])
                obj.side_center_y = float(track_data['side']['bottom_center'][1])

            # Save Weight and Fusion Status (FIX: Convert numpy float to python float)
            est_weight = track_data.get('estimated_weight')
            if est_weight is not None:
                obj.estimated_weight = float(est_weight)
            
            obj.is_fused = track_data.get('is_fused', False)
                
            db.commit()
        except Exception as e:
            print(f"DB Error: {e}")
            db.rollback()
        finally:
            db.close()

snapshot_service = SnapshotService()