import time
import cv2
import threading
import numpy as np
import traceback
from app.core.config import settings
from app.services.camera import camera_manager
from app.services.inference import InferenceManager
from app.services.fusion import fusion_service

class PipelineService:
    def __init__(self):
        self.inference_manager = InferenceManager(settings)
        self.running = False
        self.thread = None
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

    def stop_session(self):
        self.session_active = False
        print("Session force stopped")

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
                
                # 3. Send to Inference
                if f0 is not None: self.inference_manager.send_frame(0, f0)
                if f1 is not None: self.inference_manager.send_frame(1, f1)

                # 4. Process Results
                results = self.inference_manager.get_results()
                temp_outs = [None, None]
                
                for res in results:
                    if 'error' in res: 
                        print(f"Inference error: {res['error']}")
                        continue
                    if 'metric_infer_ms' in res: continue
                        
                    cam = res.get('cam')
                    if cam is None: continue
                    
                    orig_frame = f0 if cam == 0 else f1
                    if orig_frame is None: continue
                    
                    ann_small = None
                    if 'shape' in res:
                        ann_small = self.inference_manager.resolve_shm_image(cam, res['shape'])
                    elif 'jpeg' in res:
                        arr = np.frombuffer(res['jpeg'], dtype=np.uint8)
                        ann_small = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    
                    if ann_small is not None:
                        annotated_orig = cv2.resize(ann_small, (orig_frame.shape[1], orig_frame.shape[0]))
                        temp_outs[cam] = annotated_orig
                        
                        dets = res.get('detections', [])
                        ih, iw = ann_small.shape[:2]
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
                        
                        self.raw_detections[cam] = scaled_dets
                        self.raw_detection_ts[cam] = res.get('ts', 0)

                # Ensure WebRTC always has a frame to show
                for i in range(2):
                    if temp_outs[i] is not None:
                        camera_manager.set_annotated_frame(i, temp_outs[i])
                    else:
                        if camera_manager.get_annotated_frame(i) is None:
                            camera_manager.set_annotated_frame(i, f0 if i == 0 else f1)

                # 5. Fusion
                current_fused = []
                if self.raw_detections[0] and self.raw_detections[1] and fusion_service.H is not None:
                    current_fused = fusion_service.process_frame(
                        self.raw_detections[0], self.raw_detections[1],
                        self.raw_detection_ts[0], self.raw_detection_ts[1],
                        f0, f1
                    )

                # 6. Session Logic
                if self.session_active:
                    if time.time() > self.session_end_time:
                        self.session_active = False
                        print(f"Session finished. Best count: {self.best_session_result['count']}")
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

                # 7. Annotate Fusion & Increment Tick
                for i in range(2):
                    frame_out = camera_manager.get_annotated_frame(i)
                    if frame_out is not None and current_fused:
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

                camera_manager.increment_tick()
                
                elapsed = time.time() - t0
                to_sleep = max(0.0, tick_interval - elapsed)
                time.sleep(to_sleep)
            except Exception as e:
                print(f"Pipeline error: {e}")
                traceback.print_exc()
                time.sleep(0.1)

pipeline_service = PipelineService()
