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

    def save_fusion_snapshot(self, fused_track, frame_top, frame_side):
        # fused_track is the dict from FusionService
        if fused_track.get('has_snapshot'):
            return

        # Check in DB to be sure (optional, but good for consistency)
        # But for performance we rely on the runtime flag 'has_snapshot'
        
        runtime_id = fused_track['id']
        
        # Prepare crops
        top_crop = self._crop(frame_top, fused_track['top']['box'])
        side_crop = self._crop(frame_side, fused_track['side']['box'])

        if top_crop is None and side_crop is None:
            return

        ts = int(time.time())
        top_path_rel = f"id{runtime_id}_top_{ts}.jpg"
        side_path_rel = f"id{runtime_id}_side_{ts}.jpg"
        
        top_path_abs = os.path.join(settings.SNAPSHOT_DIR, top_path_rel)
        side_path_abs = os.path.join(settings.SNAPSHOT_DIR, side_path_rel)

        if top_crop is not None:
            self.executor.submit(self._save_file, top_path_abs, top_crop)
        else:
            top_path_rel = None
            
        if side_crop is not None:
            self.executor.submit(self._save_file, side_path_abs, side_crop)
        else:
            side_path_rel = None

        # Update DB
        self.executor.submit(self._update_db, runtime_id, top_path_rel, side_path_rel, fused_track)
        
        fused_track['has_snapshot'] = True
        fused_track['snapshot_paths'] = (top_path_rel, side_path_rel)

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
            # Check if exists
            obj = db.query(FusedObject).filter(FusedObject.track_id == runtime_id).first()
            if not obj:
                obj = FusedObject(track_id=runtime_id)
                db.add(obj)
            
            obj.snapshot_top = top_path
            obj.snapshot_side = side_path
            
            # Save metadata from the moment of snapshot
            top_c = track_data['top'].get('center')
            side_c = track_data['side'].get('bottom_center')
            if top_c:
                obj.top_center_x = top_c[0]
                obj.top_center_y = top_c[1]
            if side_c:
                obj.side_center_x = side_c[0]
                obj.side_center_y = side_c[1]
                
            db.commit()
        except Exception as e:
            print(f"DB Error: {e}")
            db.rollback()
        finally:
            db.close()

import time # added missing import due to usage in save_fusion_snapshot
snapshot_service = SnapshotService()
