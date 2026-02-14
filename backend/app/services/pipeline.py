import time
import cv2
import threading
import numpy as np
import traceback
from app.core.config import settings
from app.services.camera import camera_manager
from app.services.inference import InferenceManager
from app.services.fusion import fusion_service
from app.services.video_recorder import video_recorder

class PipelineService:
    def __init__(self):
        self.inference_manager = InferenceManager(settings)
        self.running = False
        self.thread = None
        # If True, pipeline will shut itself down after a triggered session finishes
        self.started_on_demand = False
        self.raw_detections = [[], []]
        self.raw_detection_ts = [0.0, 0.0]
        self.session_active = False
        self.session_end_time = 0.0
        self.best_session_result = {
            'count': 0,
            'detections': [[], []],
            'frames': [None, None],
            'timestamp': 0.0
        }
        # Debug counters for unexpected zero-detections
        self._consecutive_zero = [0, 0]
        self._last_nonzero_ts = [0.0, 0.0]
        # Simple per-camera trackers used as fallback when YOLO misses frames
        self._trackers = [[], []]  # each entry: list of {'tracker': obj, 'cls': int, 'score': float, 'last_update': ts}
        self._tracker_expiry = getattr(settings, 'DETECTION_HOLD_SECONDS', 1.0)
        # Debug: remember last drawn detection count to surface changes
        self._last_draw_n = [0, 0]
        # Frame skip for inference (send every Nth frame to worker to reduce YOLO CPU load)
        self._inference_skip_frames = getattr(settings, 'INFERENCE_SKIP_FRAMES', 5)
        self._frame_counter = [0, 0]  # Count frames per camera

    def start_session(self, duration=60):
        print(f"Starting triggered session for {duration} seconds")
        self.best_session_result = {
            'count': 0,
            'detections': [[], []],
            'frames': [None, None],
            'timestamp': 0.0
        }
        self.session_end_time = time.time() + duration
        self.session_active = True
        # Start video recording
        video_recorder.start_session(fps=settings.MOTION_HIGH_FPS)

    def stop_session(self):
        self.session_active = False
        print("Session force stopped")
        # Stop video recording
        video_recorder.stop_session()

    def start(self):
        self.inference_manager.start()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.inference_manager.stop()

    def mark_started_on_demand(self, v=True):
        self.started_on_demand = bool(v)

    def _draw_detections(self, frame, detections):
        """Helper to draw detections on a frame locally (high-visibility annotations)."""
        if frame is None:
            return None
        out = frame.copy()
        h, w = out.shape[:2]
        thickness = max(2, int(min(w, h) / 200))

        for d in detections:
            box = d.get('box')
            if not box:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            score = d.get('score', 0.0)
            cls_id = d.get('cls', '?')

            # Box (thicker for visibility)
            color = (0, 220, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

            # Label with filled background for contrast
            label = f"{cls_id}:{score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            pad = 6
            lx1, ly1 = x1, max(0, y1 - th - pad)
            lx2, ly2 = x1 + tw + pad, y1
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(out, label, (lx1 + 3, ly2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        # Header when detections exist so it's obvious in saved videos
        try:
            if detections:
                header = f"YOLO detections: {len(detections)}"
                cv2.putText(out, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        except Exception:
            pass

        return out

    def _run(self):
        print("Starting pipeline loop")
        # Initialize previous frames for motion (simplified motion logic)
        prev_small = [None, None]
        
        while self.running:
            try:
                t0 = time.time()

                # 1. Get Frames
                f0, ts0 = camera_manager.get_raw_frame(0)
                f1, ts1 = camera_manager.get_raw_frame(1)
                
                if f0 is None and f1 is None:
                    time.sleep(0.01)
                    continue

                # 2. Motion Detection / Adaptive FPS Control
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
                
                # 3. Send to Inference (Skip frames to reduce load: send every Nth frame)
                for cam_idx, frame in enumerate([f0, f1]):
                    if frame is not None:
                        self._frame_counter[cam_idx] += 1
                        # Send frame if counter is multiple of skip value
                        if self._frame_counter[cam_idx] % (self._inference_skip_frames + 1) == 0:
                            self.inference_manager.send_frame(cam_idx, frame)
                            print(f"[pipeline.infer] cam={cam_idx} sent frame #{self._frame_counter[cam_idx]} for inference")
                        # Reset counter periodically to avoid overflow
                        if self._frame_counter[cam_idx] > 100000:
                            self._frame_counter[cam_idx] = 0

                # 4. Process Results (Update cached detections)
                # This retrieves inference results that have been processed by the worker
                results = self.inference_manager.get_results()
                
                # Log when results arrive for debugging
                try:
                    if results:
                        print(f"[pipeline.results] received {len(results)} result(s) at t={now:.2f}s")
                        for res in results:
                            cam = res.get('cam')
                            if 'detections' in res:
                                print(f"[pipeline.results] -- cam={cam}: {len(res['detections'])} detections")
                except Exception:
                    pass

                # If available, the worker can provide its own annotated image via SHM
                worker_annotated_frames = [None, None]
                
                for res in results:
                    if 'error' in res: 
                        print(f"Inference error: {res['error']}")
                        continue
                    if 'metric_infer_ms' in res: continue
                        
                    cam = res.get('cam')
                    if cam is None: continue
                    
                    # We only need the detections, we will draw them on the FRESH frame
                    # This decouples inference FPS from Display FPS
                    dets = res.get('detections', [])
                    
                    # We need to scale detections back to original resolution?
                    # The worker receives a resized image (inference_width).
                    # 'detections' from worker are relative to that small image.
                    # We need to scale them to the CURRENT f0/f1 size.
                    # But f0/f1 might have changed size? Assume stream size is constant.
                    
                    orig_frame = f0 if cam == 0 else f1
                    if orig_frame is None: continue

                    # Calculate scale
                    # The worker was sent a frame resized to settings.INFERENCE_WIDTH
                    # We can infer the scale from the 'shape' returned or just re-calculate based on config
                    # But simpler: The worker returns detections.
                    # We need to know the size of the image the worker PROCESSED.
                    # The worker sends back 'shape'.
                    
                    shape = res.get('shape') # (h, w, 3) of the processed image
                    if shape:
                        ih, iw = shape[:2]
                        oh, ow = orig_frame.shape[:2]
                        sx, sy = (ow/iw, oh/ih)
                        
                        scaled_dets = []
                        for d in dets:
                            sd = d.copy()
                            b = d['box']
                            sd['box'] = [int(b[0]*sx), int(b[1]*sy), int(b[2]*sx), int(b[3]*sy)]
                            if d['center']: sd['center'] = (d['center'][0]*sx, d['center'][1]*sy)
                            if d['bottom_center']: sd['bottom_center'] = (d['bottom_center'][0]*sx, d['bottom_center'][1]*sy)
                            scaled_dets.append(sd)

                        # Try to read the worker-produced annotated image (original YOLO rendering)
                        try:
                            if getattr(settings, 'USE_WORKER_ANNOTATED', False):
                                wshape = res.get('shape')
                                if wshape:
                                    try:
                                        worker_ann = self.inference_manager.resolve_shm_image(cam, wshape)
                                        if worker_ann is not None:
                                            # scale worker annotated image to match the original frame size
                                            ah, aw = worker_ann.shape[:2]
                                            oh, ow = orig_frame.shape[:2]
                                            if (ah, aw) != (oh, ow):
                                                worker_ann = cv2.resize(worker_ann, (ow, oh))
                                            # store so pipeline can prefer it when rendering/recording
                                            worker_annotated_frames[cam] = worker_ann
                                            print(f"[pipeline] worker annotated available for cam={cam}")
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                        # Hold previous non-empty detections for brief dropouts (temporal smoothing)
                        try:
                            hold_secs = getattr(settings, 'DETECTION_HOLD_SECONDS', 1.0)
                            now = time.time()
                            
                            if len(scaled_dets) == 0 and self._last_nonzero_ts[cam] > 0 and (now - self._last_nonzero_ts[cam]) <= hold_secs:
                                # reuse previous detections (smooth temporary gaps)
                                scaled_dets = list(self.raw_detections[cam]) if self.raw_detections[cam] else []
                                time_held = now - self._last_nonzero_ts[cam]
                                print(f"[pipeline.hold] cam={cam} holding detections for {time_held:.1f}s (hold_window={hold_secs:.1f}s)")
                            elif len(scaled_dets) > 0:
                                # update timestamp on any non-empty detection result
                                self._last_nonzero_ts[cam] = now
                                self._consecutive_zero[cam] = 0
                                print(f"[pipeline.hold] cam={cam} got fresh detections: {len(scaled_dets)} objects")
                        except Exception as e:
                            print(f"[pipeline.hold] error in hold logic: {e}")

                        prev_count = len(self.raw_detections[cam]) if self.raw_detections[cam] else 0
                        self.raw_detections[cam] = scaled_dets
                        self.raw_detection_ts[cam] = res.get('ts', 0)

                        # Debug: track consecutive zero-detection runs and save frames for inspection
                        try:
                            if len(scaled_dets) == 0:
                                self._consecutive_zero[cam] += 1
                                if self._consecutive_zero[cam] >= getattr(settings, 'DEBUG_SAVE_ZERO_DET_THRESHOLD', 3):
                                    # Save raw + annotated frames for debugging
                                    from pathlib import Path
                                    import time as _time
                                    dbg_dir = Path('snapshots') / 'debug_zero_det'
                                    dbg_dir.mkdir(parents=True, exist_ok=True)
                                    ts = int(_time.time())
                                    raw_frame = f0 if cam == 0 else f1
                                    ann = None
                                    try:
                                        ann = self._draw_detections(raw_frame, scaled_dets)
                                    except Exception:
                                        ann = None
                                    try:
                                        if raw_frame is not None:
                                            Path(dbg_dir / f"cam{cam}_raw_{ts}.jpg").write_bytes(cv2.imencode('.jpg', raw_frame)[1].tobytes())
                                        if ann is not None:
                                            Path(dbg_dir / f"cam{cam}_ann_{ts}.jpg").write_bytes(cv2.imencode('.jpg', ann)[1].tobytes())
                                        print(f"[pipeline.debug] saved debug_zero_det cam={cam} count=0 consecutive={self._consecutive_zero[cam]} dir={dbg_dir}")
                                    except Exception as e:
                                        print(f"[pipeline.debug] failed to save debug frames: {e}")
                            else:
                                # reset counter and update last nonzero timestamp
                                if len(scaled_dets) > 0:
                                    self._consecutive_zero[cam] = 0
                                    self._last_nonzero_ts[cam] = time.time()
                        except Exception:
                            pass

                        # If we detected objects, start a short auto-save session so the annotated video is stored
                        try:
                            if len(scaled_dets) > 0 and not self.session_active:
                                duration = getattr(settings, 'AUTO_SAVE_ON_DETECT_SECONDS', 10)
                                print(f"[pipeline] detection triggered auto-save session for {duration}s (cam={cam} count={len(scaled_dets)})")
                                self.start_session(duration=duration)
                        except Exception:
                            pass

                # 5. GENERATE ANNOTATED FRAMES (Smooth Rendering)
                # Instead of using the worker's stale image, we draw the latest detections
                # on the CURRENT fresh frame.
                
                annotated_frames = [None, None]
                raw_frames = [f0, f1]

                for i in range(2):
                    if raw_frames[i] is not None:
                        # Prefer the worker's original annotated frame if configured and available
                        frame_ann = None
                        try:
                            if getattr(settings, 'USE_WORKER_ANNOTATED', False) and worker_annotated_frames[i] is not None:
                                frame_ann = worker_annotated_frames[i]
                                # ensure shape matches raw frame
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

                        # Log when detection counts change so we can trace dynamic updates
                        try:
                            cur_n = len(self.raw_detections[i]) if self.raw_detections[i] else 0
                            if cur_n != self._last_draw_n[i]:
                                print(f"[pipeline.draw] cam={i} detections={cur_n}")
                                # Save a quick snapshot for visual verification when detections appear/disappear
                                try:
                                    from pathlib import Path
                                    snap_dir = Path('snapshots')
                                    snap_dir.mkdir(parents=True, exist_ok=True)
                                    if cur_n > 0 and frame_ann is not None:
                                        Path(snap_dir / f"last_detection_cam{i}.jpg").write_bytes(cv2.imencode('.jpg', frame_ann)[1].tobytes())
                                except Exception:
                                    pass
                                self._last_draw_n[i] = cur_n
                        except Exception:
                            pass
                
                # 6. Fusion (Using cached detections)
                current_fused = []
                if self.raw_detections[0] and self.raw_detections[1] and fusion_service.H is not None:
                    # Check if detections are fresh enough? (Optional)
                    # For now we fuse whatever we have
                    current_fused = fusion_service.process_frame(
                        self.raw_detections[0], self.raw_detections[1],
                        self.raw_detection_ts[0], self.raw_detection_ts[1],
                        f0, f1
                    )

                # 7. Draw Fusion Markers & Update Camera Manager
                for i in range(2):
                    frame_out = annotated_frames[i]
                    if frame_out is not None:
                        # Add fusion dots
                        if current_fused:
                            if i == 0:
                                for fo in current_fused:
                                    tc = fo['top'].get('center')
                                    if tc: cv2.circle(frame_out, (int(tc[0]), int(tc[1])), 10, (0,0,255), -1)
                            if i == 1:
                                if fusion_service.H is not None:
                                     from app.utils.geometry import project_point
                                     for d in self.raw_detections[0]:
                                         c = d.get('center')
                                         if c:
                                             pp = project_point(fusion_service.H, c)
                                             if pp: cv2.circle(frame_out, (int(pp[0]), int(pp[1])), 8, (255,0,0), 2)
                                for fo in current_fused:
                                    sc = fo['side'].get('bottom_center')
                                    if sc: cv2.circle(frame_out, (int(sc[0]), int(sc[1])), 10, (0,0,255), -1)
                        
                        # Update display
                        camera_manager.set_annotated_frame(i, frame_out)
                    else:
                        # Fallback if raw frame was None (rare)
                        pass

                # 8. Session Logic & Recording
                if self.session_active:
                    if time.time() > self.session_end_time:
                        self.session_active = False
                        print(f"Session finished. Best count: {self.best_session_result['count']}")
                        video_recorder.stop_session()
                        if self.started_on_demand:
                            try:
                                print("[pipeline] session finished; shutting down on-demand pipeline")
                                self.inference_manager.stop()
                            except Exception:
                                pass
                            self.running = False
                            break
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
                    
                    # RECORD FRAMES
                    # Now we record the SMOOTH annotated frame, not the STALE one
                    video_recorder.write_frame(0, f0, annotated_frames[0])
                    video_recorder.write_frame(1, f1, annotated_frames[1])

                camera_manager.increment_tick()
                
                elapsed = time.time() - t0
                to_sleep = max(0.0, tick_interval - elapsed)
                time.sleep(to_sleep)
            except Exception as e:
                print(f"Pipeline error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

pipeline_service = PipelineService()