import os
import cv2
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
        
        # 1. Handle Top Camera (Always exists)
        top_crop = self._crop(frame_top, track_obj['top']['box']) if track_obj.get('top') else None
        top_path_rel = f"id{runtime_id}_top_{ts}.jpg" if top_crop is not None else None
        
        if top_crop is not None:
            top_path_abs = os.path.join(settings.SNAPSHOT_DIR, top_path_rel)
            self.executor.submit(self._save_file, top_path_abs, top_crop)

        # 2. Handle Side Camera (Might be None if not fused)
        side_crop = None
        side_path_rel = None
        if track_obj.get('is_fused') and track_obj.get('side') and frame_side is not None:
            side_crop = self._crop(frame_side, track_obj['side']['box'])
            if side_crop is not None:
                side_path_rel = f"id{runtime_id}_side_{ts}.jpg"
                side_path_abs = os.path.join(settings.SNAPSHOT_DIR, side_path_rel)
                self.executor.submit(self._save_file, side_path_abs, side_crop)

        # 3. Update Database
        self.executor.submit(self._update_db, runtime_id, top_path_rel, side_path_rel, track_obj)
        
        track_obj['has_snapshot'] = True
        track_obj['snapshot_paths'] = (top_path_rel, side_path_rel)

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
            
            # Save Metadata
            if track_data.get('top') and track_data['top'].get('center'):
                obj.top_center_x = track_data['top']['center'][0]
                obj.top_center_y = track_data['top']['center'][1]
                
            if track_data.get('side') and track_data['side'].get('bottom_center'):
                obj.side_center_x = track_data['side']['bottom_center'][0]
                obj.side_center_y = track_data['side']['bottom_center'][1]

            # Save Weight and Fusion Status
            obj.estimated_weight = track_data.get('estimated_weight')
            obj.is_fused = track_data.get('is_fused', False)
                
            db.commit()
        except Exception as e:
            print(f"DB Error: {e}")
            db.rollback()
        finally:
            db.close()

import time # added missing import due to usage in save_fusion_snapshot
snapshot_service = SnapshotService()
