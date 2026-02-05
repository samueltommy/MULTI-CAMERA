import numpy as np
import time
import math
import json
from app.core.config import settings
from app.utils.geometry import project_point
from app.services.snapshot import snapshot_service
from app.database.session import SessionLocal
from app.database.models import Calibration

class FusionService:
    def __init__(self):
        self.H = None
        self.load_homography()
        self.fused_tracks = []
        self.fused_id_counter = 1
        
    def load_homography(self):
        # Try loading from DB first
        try:
            db = SessionLocal()
            cal = db.query(Calibration).filter(Calibration.is_active == 1).order_by(Calibration.created_at.desc()).first()
            if cal:
                self.H = np.array(json.loads(cal.matrix_json))
                print(f"Loaded homography from DB (ID: {cal.id})")
                db.close()
                return
            db.close()
        except Exception as e:
            print(f"Failed to load homography from DB: {e}")

        # Fallback to file
        try:
           self.H = np.load(settings.HOMOGRAPHY_PATH)
           print(f"Loaded homography fallback from {settings.HOMOGRAPHY_PATH}")
        except Exception:
            self.H = None
            print("Homography not loaded.")

    def set_homography(self, H, name="Manual Calibration"):
        self.H = H
        # Update file fallback
        try:
            np.save(settings.HOMOGRAPHY_PATH, H)
        except: pass
        
        # Save to DB
        try:
            db = SessionLocal()
            # Deactivate old ones? (Optional but keeps history clean)
            db.query(Calibration).update({Calibration.is_active: 0})
            
            new_cal = Calibration(
                name=name,
                matrix_json=json.dumps(H.tolist()),
                is_active=1
            )
            db.add(new_cal)
            db.commit()
            print(f"Saved new homography to DB (ID: {new_cal.id})")
            db.close()
        except Exception as e:
            print(f"Failed to save homography to DB: {e}")

    def match_detections(self, dets_top, dets_side):
        matches = []
        if self.H is None or not dets_top or not dets_side:
            return matches

        # Project top points
        proj = []
        for d in dets_top:
            c = d.get('center') # Note: original used bottom_center for top projection in one place? 
            # In app.py: matches = fusion.match_detections(...)
            # In src/fusion.py: 
            #   c = d.get('bottom_center') 
            #   proj.append(project_point(H, (c[0], c[1])))
            
            # Wait, usually for Top view, center is better. But I must follow the user's "work perfectly the same".
            # Let's check src/fusion.py content again from context.
            # It said: c = d.get('bottom_center')
            
            # But in app.py logic when constructing `dets_top` (cam 0), it calculated 'center' and 'bottom_center'.
            # Usually Top view -> Center is the footprint. 
            # The existing code uses 'bottom_center'. Use that.
            c = d.get('bottom_center')
            if c:
                proj.append(project_point(self.H, c))
            else:
                proj.append(None)

        for i_top, p in enumerate(proj):
            if p is None: continue
            cls_top = dets_top[i_top].get('cls')
            
            best_j = None
            best_d = float('inf')
            
            for j, sd in enumerate(dets_side):
                cls_side = sd.get('cls')
                if cls_top != cls_side: continue
                
                bc = sd.get('bottom_center')
                if bc is None: continue
                
                dx = p[0] - bc[0]
                dy = p[1] - bc[1]
                dist = (dx*dx+dy*dy)**0.5
                
                if dist < best_d:
                    best_d = dist
                    best_j = j
            
            if best_j is not None and best_d <= settings.MATCH_MAX_DIST_PX:
                matches.append((i_top, best_j, best_d))
        return matches

    def process_frame(self, dets0, dets1, ts0, ts1, frame0, frame1):
        if self.H is None: return []
        
        # Sync check
        if abs(ts0 - ts1) > (settings.SYNC_TOL_MS / 1000.0):
            return []

        matches = self.match_detections(dets0, dets1)
        now_ts = time.time()
        
        # Prune
        self.fused_tracks = [t for t in self.fused_tracks if (now_ts - t['last_seen']) <= settings.FUSE_TIME_WINDOW]
        
        current_fused_objects = []

        for mi, mj, dist in matches:
            top = dets0[mi]
            side = dets1[mj]
            
            # Association
            best_ft = None
            best_score = float('inf')
            
            side_bc = side.get('bottom_center')
            top_c = top.get('center')
            
            for ft in self.fused_tracks:
                prev_side = ft.get('side_center')
                prev_top = ft.get('top_center')
                
                ds = 1e6
                if prev_side and side_bc:
                    ds = ((prev_side[0]-side_bc[0])**2 + (prev_side[1]-side_bc[1])**2)**0.5
                
                dt = 1e6
                if prev_top and top_c:
                     dt = ((prev_top[0]-top_c[0])**2 + (prev_top[1]-top_c[1])**2)**0.5
                
                spatial = settings.ASSOC_WEIGHT_SIDE * ds + settings.ASSOC_WEIGHT_TOP * dt
                last = ft.get('last_seen', now_ts)
                time_penalty = math.exp(-(now_ts - last)/ settings.ASSOC_TIME_DECAY)
                score = spatial / max(1e-6, time_penalty)
                
                if score < best_score:
                    best_score = score
                    best_ft = ft
            
            attached = None
            if best_ft is not None:
                # thresholds logic from app.py
                # They used side_dist primarily for thresholding
                s_dist = float('inf')
                if best_ft.get('side_center') and side_bc:
                     s_dist = ((best_ft['side_center'][0]-side_bc[0])**2 + (best_ft['side_center'][1]-side_bc[1])**2)**0.5
                
                if s_dist <= settings.MATCH_MAX_DIST_PX:
                    attached = best_ft

            if attached is None:
                fid = self.fused_id_counter
                self.fused_id_counter += 1
                attached = {
                    'id': fid,
                    'first_seen': now_ts,
                    'last_seen': now_ts,
                    'top_center': top_c,
                    'side_center': side_bc,
                    'top': top, # store current detection for snapshotting
                    'side': side,
                    'has_snapshot': False
                }
                self.fused_tracks.append(attached)
            else:
                attached['last_seen'] = now_ts
                attached['top_center'] = top_c
                attached['side_center'] = side_bc
                attached['top'] = top
                attached['side'] = side
            
            # Snapshot logic
            snapshot_service.save_fusion_snapshot(attached, frame0, frame1)

            current_fused_objects.append({
                'fid': attached['id'],
                'top': top,
                'side': side,
                'dist': dist
            })
            
        return current_fused_objects

fusion_service = FusionService()
