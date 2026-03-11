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
        try:
            cv2.imwrite(path, img)
        except Exception as e:
            print(f"Error saving snapshot: {e}")

    def save_fusion_snapshot(self, track_obj, frame_top, frame_side=None, session_id=None):
        if track_obj.get('has_snapshot'):
            return
        
        runtime_id = track_obj['id']
        ts = int(time.time())
        
        # --- STRUKTUR FOLDER BARU ---
        # Format: snapshots / SESSION_ID / id_100 / top_...jpg
        if session_id:
            # Jika ada session_id, buat folder sesi
            base_folder = os.path.join(settings.SNAPSHOT_DIR, session_id, f"id_{runtime_id}")
        else:
            # Fallback jika tidak ada sesi (misal debug), langsung folder ID
            base_folder = os.path.join(settings.SNAPSHOT_DIR, f"id_{runtime_id}")
            
        os.makedirs(base_folder, exist_ok=True)
        # ----------------------------

        # 1. Handle Top Camera
        top_crop = self._crop(frame_top, track_obj['top']['box']) if track_obj.get('top') else None
        
        # Simpan path relatif untuk DB (opsional: bisa simpan full path atau relatif terhadap snapshot dir)
        # Kita simpan relatif agar fleksibel
        db_path_top = None
        
        if top_crop is not None:
            filename_top = f"top_{ts}.jpg"
            abs_path_top = os.path.join(base_folder, filename_top)
            
            # Path relatif untuk disimpan di DB (misal: "20260304_120000/id_1/top_...jpg")
            rel_folder = os.path.relpath(base_folder, settings.SNAPSHOT_DIR)
            db_path_top = os.path.join(rel_folder, filename_top).replace("\\", "/") # Force forward slash for consistency
            
            self.executor.submit(self._save_file, abs_path_top, top_crop)

        # 2. Handle Side Camera
        side_crop = None
        db_path_side = None
        
        if track_obj.get('is_fused') and track_obj.get('side') and frame_side is not None:
            side_crop = self._crop(frame_side, track_obj['side']['box'])
            if side_crop is not None:
                filename_side = f"side_{ts}.jpg"
                abs_path_side = os.path.join(base_folder, filename_side)
                
                rel_folder = os.path.relpath(base_folder, settings.SNAPSHOT_DIR)
                db_path_side = os.path.join(rel_folder, filename_side).replace("\\", "/")
                
                self.executor.submit(self._save_file, abs_path_side, side_crop)

        # 3. Update Database
        self.executor.submit(self._update_db, runtime_id, db_path_top, db_path_side, track_obj, session_id)
        
        track_obj['has_snapshot'] = True
        track_obj['snapshot_paths'] = (db_path_top, db_path_side)

    def _crop(self, frame, box):
        if frame is None or not box: return None
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        x1c, y1c = max(0, min(w-1, x1)), max(0, min(h-1, y1))
        x2c, y2c = max(0, min(w-1, x2)), max(0, min(h-1, y2))
        if y2c > y1c and x2c > x1c:
            return frame[y1c:y2c, x1c:x2c].copy()
        return None

    def _update_db(self, runtime_id, top_path, side_path, track_data, session_id=None):
        db = SessionLocal()
        try:
            # 1. Hitung Umur Ayam
            from app.database.models import FarmSettings
            import datetime as dt
            
            settings = db.query(FarmSettings).first()
            current_age = 0
            if settings:
                if settings.manual_age_override:
                    current_age = settings.manual_age_override
                elif settings.chick_in_date:
                    delta = dt.date.today() - settings.chick_in_date
                    current_age = max(0, delta.days)

            # 2. Query Objek
            query = db.query(FusedObject).filter(FusedObject.track_id == runtime_id)
            if session_id:
                query = query.filter(FusedObject.session_id == session_id)
            else:
                query = query.filter(FusedObject.session_id.is_(None))
                
            obj = query.first()
            if not obj:
                obj = FusedObject(track_id=runtime_id, session_id=session_id)
                db.add(obj)
            
            # 3. Simpan Path & Metadata Standar
            if top_path: obj.snapshot_top = top_path
            if side_path: obj.snapshot_side = side_path
            
            if track_data.get('top') and track_data['top'].get('center'):
                obj.top_center_x = float(track_data['top']['center'][0])
                obj.top_center_y = float(track_data['top']['center'][1])
                obj.score = float(track_data['top'].get('score', 0.98))
                
            if track_data.get('side') and track_data['side'].get('bottom_center'):
                obj.side_center_x = float(track_data['side']['bottom_center'][0])
                obj.side_center_y = float(track_data['side']['bottom_center'][1])

            obj.estimated_weight = float(track_data.get('estimated_weight', 0))
            obj.is_fused = track_data.get('is_fused', False)
            obj.age_days = current_age
            
            # --- 4. SIMPAN RAW FEATURES UNTUK MACHINE LEARNING ---
            if track_data.get('top'):
                top_box = track_data['top'].get('box')
                if top_box:
                    obj.bbox_width_px = float(top_box[2] - top_box[0])
                obj.mask_area_px = float(track_data['top'].get('mask_area', 0.0))
                
            if track_data.get('side'):
                side_box = track_data['side'].get('box')
                if side_box:
                    obj.bbox_height_px = float(side_box[3] - side_box[1])

            # Ambil skala kalibrasi ArUco saat ini
            from app.services.weight_predictor import weight_predictor
            obj.cm_per_pixel = weight_predictor.cm_per_pixel
            # -----------------------------------------------------
                
            db.commit()
        except Exception as e:
            print(f"DB Error: {e}")
            db.rollback()
        finally:
            db.close()

snapshot_service = SnapshotService()