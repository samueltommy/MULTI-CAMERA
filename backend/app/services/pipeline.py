import time
import cv2
import threading
import numpy as np
import traceback
from collections import deque
from app.core.config import settings
from app.services.camera import camera_manager
from app.services.inference import InferenceManager
from app.services.fusion import fusion_service
from app.services.video_recorder import video_recorder
from app.utils.latency_tracker import LatencyTracker

class PipelineService:
    def __init__(self):
        self.inference_manager = None 
        self.running = False
        self.thread = None
        self.inference_enabled = False
        self.started_on_demand = False
        self.raw_detections = [[], []]
        self.raw_detection_ts = [0.0, 0.0]
        self.session_active = False
        self.session_end_time = 0.0
        self.current_session_id = None 
        
        self.best_session_result = {
            'count': 0, 'detections': [[], []], 'frames': [None, None], 'timestamp': 0.0
        }
        
        self.object_stats = {} 
        self.alpha = 0.05
        self.move_threshold = 5.0
        self.jump_threshold = 0.30

        self._consecutive_zero = [0, 0]
        self._last_nonzero_ts = [0.0, 0.0]
        self._last_draw_n = [0, 0]
        self._inference_skip_frames = getattr(settings, 'INFERENCE_SKIP_FRAMES', 1)
        self._frame_counter = [0, 0]
        
        # Variabel Tracker Latensi
        self.latency_logger = None
        self.latency_frame_counter = 0

    def _get_smart_weight(self, obj_id, current_center, raw_weight):
        # (Logika smart weight sama seperti sebelumnya, tidak berubah)
        if obj_id not in self.object_stats:
            self.object_stats[obj_id] = {
                'centers': deque(maxlen=5),
                'stable_weight': 0.0
            }
        stats = self.object_stats[obj_id]
        if current_center:
            stats['centers'].append(current_center)
        last_stable = stats['stable_weight']
        if raw_weight < 0.05:
            return last_stable
        is_moving = False
        if len(stats['centers']) >= 2:
            curr = stats['centers'][-1]
            prev = stats['centers'][-2]
            dist = ((curr[0]-prev[0])**2 + (curr[1]-prev[1])**2)**0.5
            if dist > self.move_threshold:
                is_moving = True
        if is_moving and last_stable > 0:
            return last_stable
        if last_stable > 0:
            diff_pct = abs(raw_weight - last_stable) / last_stable
            if diff_pct > self.jump_threshold:
                return last_stable
        if last_stable == 0.0:
            new_weight = raw_weight
        else:
            new_weight = (self.alpha * raw_weight) + ((1 - self.alpha) * last_stable)
        stats['stable_weight'] = new_weight
        return new_weight

    def start_session(self, duration=60):
        print(f"Starting triggered session for {duration} seconds")
        self.current_session_id = time.strftime('%Y%m%d_%H%M%S')
        
        self.best_session_result = {
            'count': 0, 'detections': [[], []], 'frames': [None, None], 'timestamp': 0.0
        }
        self.session_end_time = time.time() + duration
        self.session_active = True
        
        # -------------------------------------------------------------
        # INISIASI LATENCY TRACKER
        self.latency_logger = LatencyTracker(self.current_session_id)
        self.latency_frame_counter = 0
        # -------------------------------------------------------------
        
        from app.database.session import SessionLocal
        from app.database.models import FarmSettings
        from datetime import date
        from app.services.weight_predictor import weight_predictor
        
        db = SessionLocal()
        try:
            settings_db = db.query(FarmSettings).first()
            age_days = 0
            if settings_db:
                if settings_db.manual_age_override:
                    age_days = settings_db.manual_age_override
                elif settings_db.chick_in_date:
                    delta = date.today() - settings_db.chick_in_date
                    age_days = max(0, delta.days)
            
            weight_predictor.update_age_limits(age_days)
        except Exception as e:
            print(f"[Pipeline] Error setting dynamic outlier: {e}")
        finally:
            db.close()

        video_recorder.start_session(fps=settings.MOTION_HIGH_FPS)

    def stop_session(self):
        self.session_active = False
        print("Session force stopped")
        video_recorder.stop_session()
        
        # Tutup file CSV Latensi jika diberhentikan paksa
        if self.latency_logger:
            self.latency_logger.close()
            self.latency_logger = None

        if self.current_session_id:
            from app.services.statistics import ml_statistics_service
            threading.Thread(
                target=ml_statistics_service.process_session, 
                args=(self.current_session_id,), 
                daemon=True
            ).start()

    def start(self, inference_enabled=False):
        self.enable_inference(inference_enabled)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join()
        if self.inference_manager:
            try: self.inference_manager.stop()
            except Exception: pass
        self.inference_manager = None

    def mark_started_on_demand(self, v=True):
        self.started_on_demand = bool(v)

    def enable_inference(self, enable=True):
        if enable and not self.inference_enabled:
            if self.inference_manager is None:
                self.inference_manager = InferenceManager(settings)
            self.inference_manager.start()
            self.inference_enabled = True
            print("[pipeline] inference ENABLED")
        elif not enable and self.inference_enabled:
            if self.inference_manager:
                self.inference_manager.stop()
            self.inference_enabled = False
            self.raw_detections = [[], []]
            self.raw_detection_ts = [0.0, 0.0]
            print("[pipeline] inference DISABLED")

    def _draw_detections(self, frame, detections):
        if frame is None: return None
        out = frame.copy()
        h, w = out.shape[:2]
        thickness = max(2, int(min(w, h) / 200))
        for d in detections:
            box = d.get('box')
            if not box: continue
            x1, y1, x2, y2 = [int(v) for v in box]
            obj_id = d.get('track_id', '?')
            weight = d.get('weight', 0.0) 
            color = (0, 220, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = f"ID:{obj_id} | {weight:.3f}kg"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            pad = 6
            lx1, ly1 = x1, max(0, y1 - th - pad)
            lx2, ly2 = x1 + tw + pad, y1
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(out, label, (lx1 + 3, ly2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _run(self):
        print("Starting pipeline loop")
        prev_small = [None, None]
        
        while self.running:
            try:
                # ==========================================
                # TIMER: TOTAL PIPELINE
                t0 = time.time()
                t_pipeline_start = time.perf_counter()
                
                # ==========================================
                # TIMER: RTSP CAPTURE
                t_cap_start = time.perf_counter()
                f0, ts0 = camera_manager.get_raw_frame(0)
                f1, ts1 = camera_manager.get_raw_frame(1)
                t_capture = time.perf_counter() - t_cap_start
                
                if f0 is None and f1 is None:
                    time.sleep(0.01)
                    continue

                # 2. Motion Detection
                motion_present = False
                for idx, frame in enumerate([f0, f1]):
                    if frame is None: continue
                    small_motion = frame[::4, ::4, :]
                    gray = cv2.cvtColor(small_motion, cv2.COLOR_BGR2GRAY)
                    if prev_small[idx] is not None and prev_small[idx].shape == gray.shape:
                        diff = cv2.absdiff(gray, prev_small[idx])
                        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                        motion_score = np.count_nonzero(th) / th.size
                        if motion_score >= settings.MOTION_THRESHOLD:
                            motion_present = True
                    prev_small[idx] = gray

                target_fps = settings.MOTION_HIGH_FPS if (motion_present or self.session_active) else settings.MOTION_LOW_FPS
                tick_interval = 1.0 / target_fps
                
                # ==========================================
                # TIMER: INFERENCE (KIRIM + AMBIL HASIL YOLO)
                t_infer_start = time.perf_counter()
                
                # 3. Inference Sending
                if self.inference_enabled and self.inference_manager:
                    for cam_idx, frame in enumerate([f0, f1]):
                        if frame is not None:
                            self._frame_counter[cam_idx] += 1
                            if self._frame_counter[cam_idx] % (self._inference_skip_frames + 1) == 0:
                                self.inference_manager.send_frame(cam_idx, frame)
                                print(f"[pipeline.infer] cam={cam_idx} sent frame #{self._frame_counter[cam_idx]} for inference")
                            if self._frame_counter[cam_idx] > 100000:
                                self._frame_counter[cam_idx] = 0

                # 4. Process Results
                if self.inference_enabled and self.inference_manager:
                    results = self.inference_manager.get_results()
                else:
                    results = []

                # Trigger Auto-Save
                try:
                    if self.inference_enabled and results:
                        for res in results:
                            if 'detections' in res:
                                dets = res.get('detections', [])
                                if len(dets) > 0 and not self.session_active:
                                    duration = getattr(settings, 'AUTO_SAVE_ON_DETECT_SECONDS', 10)
                                    print(f"[pipeline] detection triggered auto-save session for {duration}s")
                                    self.start_session(duration=duration)
                except Exception:
                    pass

                # Handle Inference Results
                worker_annotated_frames = [None, None]
                if self.inference_enabled:
                    for res in results:
                        if 'error' in res: 
                            print(f"Inference error: {res['error']}")
                            continue
                        if 'metric_infer_ms' in res: continue
                        cam = res.get('cam')
                        if cam is None: continue
                        dets = res.get('detections', [])
                        orig_frame = f0 if cam == 0 else f1
                        if orig_frame is None: continue

                        shape = res.get('shape') 
                        scaled_dets = []
                        if shape:
                            ih, iw = shape[:2]
                            oh, ow = orig_frame.shape[:2]
                            sx, sy = (ow/iw, oh/ih)
                            for d in dets:
                                sd = d.copy()
                                b = d['box']
                                sd['box'] = [int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]
                                if d['center']: sd['center'] = (d['center'][0]*sx, d['center'][1]*sy)
                                if d['bottom_center']: sd['bottom_center'] = (d['bottom_center'][0]*sx, d['bottom_center'][1]*sy)
                                if 'mask_area' in d:
                                    sd['mask_area'] = d['mask_area'] * (sx * sy)
                                scaled_dets.append(sd)
                            try:
                                if getattr(settings, 'USE_WORKER_ANNOTATED', False):
                                    wshape = res.get('shape')
                                    if wshape:
                                        worker_ann = self.inference_manager.resolve_shm_image(cam, wshape)
                                        if worker_ann is not None:
                                            ah, aw = worker_ann.shape[:2]
                                            oh, ow = orig_frame.shape[:2]
                                            if (ah, aw) != (oh, ow):
                                                worker_ann = cv2.resize(worker_ann, (ow, oh))
                                            worker_annotated_frames[cam] = worker_ann
                            except Exception:
                                pass

                        try:
                            hold_secs = getattr(settings, 'DETECTION_HOLD_SECONDS', 1.0)
                            now = time.time()
                            if len(scaled_dets) == 0 and self._last_nonzero_ts[cam] > 0 and (now - self._last_nonzero_ts[cam]) <= hold_secs:
                                scaled_dets = list(self.raw_detections[cam]) if self.raw_detections[cam] else []
                            elif len(scaled_dets) > 0:
                                self._last_nonzero_ts[cam] = now
                                self._consecutive_zero[cam] = 0
                        except Exception:
                            pass

                        self.raw_detections[cam] = scaled_dets
                        self.raw_detection_ts[cam] = res.get('ts', 0)
                        try:
                            if len(scaled_dets) == 0:
                                self._consecutive_zero[cam] += 1
                            else:
                                if len(scaled_dets) > 0:
                                    self._consecutive_zero[cam] = 0
                                    self._last_nonzero_ts[cam] = time.time()
                        except Exception:
                            pass
                            
                t_infer = time.perf_counter() - t_infer_start

                # ==========================================
                # TIMER: FUSION & GEOMETRY
                t_fuse_start = time.perf_counter()

                current_fused = []
                fused_top_centers = [] 
                
                from app.services.weight_predictor import weight_predictor
                from app.services.snapshot import snapshot_service
                
                # A. FUSION
                if self.raw_detections[0] and self.raw_detections[1] and fusion_service.H is not None:
                    current_fused = fusion_service.process_frame(
                        self.raw_detections[0], self.raw_detections[1],
                        self.raw_detection_ts[0], self.raw_detection_ts[1],
                        f0, f1
                    )
                
                frame_shape = f0.shape if f0 is not None else None

                # B. PROCESS FUSED (3D Volume)
                for fo in current_fused:
                    obj_id = fo['top'].get('track_id', '?')
                    top_box = fo['top'].get('box')
                    side_box = fo['side'].get('box')
                    top_center = fo['top'].get('center')
                    
                    raw_weight = weight_predictor.predict_from_volume(fo['top'], fo['side'], 'chicken', frame_shape)
                    final_weight = self._get_smart_weight(obj_id, top_center, raw_weight)

                    if final_weight >= 0:
                        track_obj = {
                            'id': obj_id,
                            'top': fo['top'],
                            'side': fo['side'],
                            'is_fused': True,
                            'estimated_weight': final_weight,
                            'has_snapshot': False,
                            'class_name': fo['top'].get('class_name', 'chicken'),
                            'score': fo['top'].get('score', 0.0)
                        }
                        snapshot_service.save_fusion_snapshot(track_obj, f0, f1, session_id=self.current_session_id)
                        
                        fo['top']['weight'] = final_weight
                        if fo.get('side'):
                            fo['side']['track_id'] = obj_id 
                            fo['side']['weight'] = final_weight 
                        
                    if top_center:
                        fused_top_centers.append(top_center)

                # C. PROCESS UNFUSED (2D Area)
                if self.raw_detections[0]:
                    if not hasattr(self, 'unfused_id_counter'):
                        self.unfused_id_counter = 10000 
                        
                    for det_top in self.raw_detections[0]:
                        obj_id = det_top.get('track_id', '?')
                        top_center = det_top.get('center')
                        
                        is_already_fused = False
                        for fc in fused_top_centers:
                            if fc and top_center and abs(fc[0]-top_center[0]) < 5 and abs(fc[1]-top_center[1]) < 5:
                                is_already_fused = True
                                break
                                
                        if not is_already_fused:
                            top_box = det_top.get('box')
                            raw_weight = weight_predictor.predict_from_area(det_top, 'chicken', frame_shape)
                            final_weight = self._get_smart_weight(obj_id, top_center, raw_weight)
                            
                            if final_weight >= 0:
                                track_obj = {
                                    'id': self.unfused_id_counter, 
                                    'top': det_top,
                                    'side': None,
                                    'is_fused': False,
                                    'estimated_weight': final_weight,
                                    'has_snapshot': False,
                                    'class_name': det_top.get('class_name', 'chicken'),
                                    'score': det_top.get('score', 0.0)
                                }
                                snapshot_service.save_fusion_snapshot(track_obj, f0, None, session_id=self.current_session_id)
                                det_top['weight'] = final_weight

                t_fuse = time.perf_counter() - t_fuse_start

                # ==============================================================================
                # 5. GENERATE ANNOTATED FRAMES
                # ==============================================================================
                annotated_frames = [None, None]
                raw_frames = [f0, f1]

                for i in range(2):
                    if raw_frames[i] is not None:
                        frame_ann = None
                        try:
                            if getattr(settings, 'USE_WORKER_ANNOTATED', False) and worker_annotated_frames[i] is not None:
                                frame_ann = worker_annotated_frames[i]
                                try:
                                    if frame_ann.shape[:2] != raw_frames[i].shape[:2]:
                                        frame_ann = cv2.resize(frame_ann, (raw_frames[i].shape[1], raw_frames[i].shape[0]))
                                except Exception:
                                    pass
                            else:
                                frame_ann = self._draw_detections(raw_frames[i], self.raw_detections[i])
                        except Exception:
                            frame_ann = self._draw_detections(raw_frames[i], self.raw_detections[i])

                        annotated_frames[i] = frame_ann
                        
                        if frame_ann is not None:
                            if current_fused:
                                if i == 0:
                                    for fo in current_fused:
                                        tc = fo['top'].get('center')
                                        if tc: cv2.circle(frame_ann, (int(tc[0]), int(tc[1])), 10, (0,0,255), -1)
                                if i == 1:
                                    if fusion_service.H is not None:
                                         from app.utils.geometry import project_point
                                         for d in self.raw_detections[0]:
                                             c = d.get('center')
                                             if c:
                                                 pp = project_point(fusion_service.H, c)
                                                 if pp: cv2.circle(frame_ann, (int(pp[0]), int(pp[1])), 8, (255,0,0), 2)
                                    for fo in current_fused:
                                        sc = fo['side'].get('bottom_center')
                                        if sc: cv2.circle(frame_ann, (int(sc[0]), int(sc[1])), 10, (0,0,255), -1)
                        
                        camera_manager.set_annotated_frame(i, frame_ann)

                # 8. Session & Cleanup
                if self.session_active:
                    if time.time() > self.session_end_time:
                        self.session_active = False
                        print(f"Session finished. Best count: {self.best_session_result['count']}")
                        video_recorder.stop_session()
                        
                        # Tutup dan Flush CSV Latensi ketika sesi habis
                        if self.latency_logger:
                            self.latency_logger.close()
                            self.latency_logger = None

                        if self.current_session_id:
                            from app.services.statistics import ml_statistics_service
                            from tools.cloud_sync import trigger_cloud_sync 
                            
                            threading.Thread(
                                target=ml_statistics_service.process_session, 
                                args=(self.current_session_id,), 
                                daemon=True
                            ).start()

                            threading.Thread(
                                target=trigger_cloud_sync, 
                                daemon=True
                            ).start()

                        if self.inference_enabled:
                            try:
                                print("[pipeline] session finished; disabling inference")
                                self.enable_inference(False)
                            except Exception:
                                pass
                    else:
                        fused_count = len(current_fused)
                        if fused_count > self.best_session_result['count']:
                            self.best_session_result = {
                                'count': fused_count,
                                'detections': [list(self.raw_detections[0]), list(self.raw_detections[1])],
                                'frames': [f0.copy() if f0 is not None else None, f1.copy() if f1 is not None else None],
                                'timestamp': time.time(),
                                'fused': current_fused
                            }
                    video_recorder.write_frame(0, f0, annotated_frames[0])
                    video_recorder.write_frame(1, f1, annotated_frames[1])

                camera_manager.increment_tick()

                if self.inference_enabled and self._frame_counter[0] % 100 == 0:
                    active_ids = {d.get('track_id') for d in self.raw_detections[0]} | \
                                {d.get('track_id') for d in self.raw_detections[1]}
                    self.object_stats = {tid: s for tid, s in self.object_stats.items() if tid in active_ids}
                
                # ==========================================
                # REKAM DATA LATENSI KE CSV JIKA SESI AKTIF
                t_pipeline_total = time.perf_counter() - t_pipeline_start
                
                if self.session_active and self.latency_logger:
                    self.latency_frame_counter += 1
                    self.latency_logger.record_frame(
                        self.latency_frame_counter, 
                        t_capture, 
                        t_infer, 
                        t_fuse, 
                        t_pipeline_total
                    )
                # ==========================================

                elapsed = time.perf_counter() - t_pipeline_start
                to_sleep = max(0.0, tick_interval - elapsed)
                time.sleep(to_sleep)
            except Exception as e:
                print(f"Pipeline error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

    def _finalize_session(self, current_session_id):
        from app.services.statistics import ml_statistics_service
        from app.services.cloud_telemetry import cloud_telemetry
        import threading
        
        stats, detailed_records = ml_statistics_service.get_full_session_data(current_session_id)
        
        if stats:
            threading.Thread(
                target=cloud_telemetry.push_full_backup, 
                args=(
                    current_session_id, 
                    stats['age_days'], 
                    stats['average_weight'], 
                    stats['total_chickens'],
                    detailed_records
                ), 
                daemon=True
            ).start()

pipeline_service = PipelineService()