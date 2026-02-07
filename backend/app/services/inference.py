import multiprocessing
from multiprocessing import shared_memory
import numpy as np
import time
import cv2
import traceback
import torch
from ultralytics import YOLO

def worker_process_func(in_q, out_q, model_path, device_name, use_half, enable_masks, alpha, inference_width, shm_in_names_arg, shm_out_names_arg, shm_max_w_arg, shm_max_h_arg, conf_1, conf_2):
    try:
        dev = device_name
        model = YOLO(model_path)
        conf_thresholds = [conf_1, conf_2]
        try:
            model.to(dev)
            if dev != 'cpu' and use_half:
                 model.model.half()
        except Exception:
            pass

        # open shared memory segments by name
        try:
            in_shms = [shared_memory.SharedMemory(name=n) for n in shm_in_names_arg]
            out_shms = [shared_memory.SharedMemory(name=n) for n in shm_out_names_arg]
            max_w = shm_max_w_arg
            max_h = shm_max_h_arg
        except Exception as e:
            out_q.put({'error': f'worker shm open failed: {e}'})
            return

        while True:
            msg = in_q.get()
            if msg is None:
                break
            try:
                cam = int(msg.get('cam', 0))
                shape = msg.get('shape')
                if shape is None:
                    continue
                h, w, c = shape
                buf = in_shms[cam].buf
                arr = np.ndarray((max_h, max_w, 3), dtype=np.uint8, buffer=buf)
                small = arr[:h, :w, :].copy()
                
                if small is None:
                    continue

                t0 = time.time()
                # Run inference
                try:
                    conf = conf_thresholds[cam] if cam < len(conf_thresholds) else 0.25
                    is_yolo26 = "yolo26" in model_path.lower()
                    
                    if use_half and dev != 'cpu':
                        results = model.track(small, device=dev, half=True, persist=True, 
                                            tracker="bytetrack.yaml", verbose=False, conf=conf,
                                            end2end=is_yolo26)
                    else:
                        results = model.track(small, device=dev, half=False, persist=True, 
                                            tracker="bytetrack.yaml", verbose=False, conf=conf,
                                            end2end=is_yolo26)
                    infer_ms = (time.time() - t0) * 1000.0
                    out_q.put({'metric_infer_ms': infer_ms})
                except Exception:
                    conf = conf_thresholds[cam] if cam < len(conf_thresholds) else 0.25
                    is_yolo26 = "yolo26" in model_path.lower()
                    results = model(small, device=dev, half=False, conf=conf, end2end=is_yolo26)

                annotated = small.copy()
                boxes, scores, class_ids, track_ids = [], [], [], []
                
                try:
                    res = results[0]
                    boxes = res.boxes.xyxy.cpu().numpy()
                    if res.boxes.id is not None:
                        track_ids = res.boxes.id.int().cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    class_ids = res.boxes.cls.cpu().numpy()
                    if res.boxes.id is not None:
                         track_ids = res.boxes.id.int().cpu().numpy()
                    else:
                         track_ids = [-1] * len(boxes)
                except Exception:
                    pass

                detections = []
                for i, (box, score, cls, tid) in enumerate(zip(boxes, scores, class_ids, track_ids)):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    detections.append({
                        'box': [x1, y1, x2, y2], 
                        'score': float(score), 
                        'cls': int(cls), 
                        'track_id': int(tid),
                        'center': (cx, cy), 
                        'bottom_center': ((x1 + x2) / 2.0, float(y2))
                    })
                
                # Draw masks and boxes (simplified for brevity, similar to original)
                if enable_masks and getattr(results[0], 'masks', None) is not None:
                     # ... mask drawing logic ...
                     # For brevity in this refactor, I will just copy the logic if essential or assume standard drawing
                     # The user wants "perfectly the same". I should copy the mask drawing logic.
                    masks = results[0].masks.xy
                    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                    for i, mask in enumerate(masks):
                        try:
                            # Alpha blending for transparency
                            color = palette[int(class_ids[i]) % len(palette)]
                            mask_pts = (mask).astype(np.int32)
                            
                            # Create a mask image
                            mask_img = np.zeros(annotated.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(mask_img, [mask_pts], 255)
                            
                            # Create colored overlay
                            colored_overlay = np.zeros_like(annotated, dtype=np.uint8)
                            colored_overlay[:] = color
                            
                            # Blend where the mask is present
                            # alpha comes from args (defaults to 0.35 in config)
                            # We only blend pixels inside the mask
                            region = (mask_img == 255)
                            annotated[region] = cv2.addWeighted(annotated[region], 1.0 - alpha, colored_overlay[region], alpha, 0)
                            
                            # Draw contour for better visibility
                            cv2.polylines(annotated, [mask_pts], True, color, 1)
                        except: pass

                for i, (box, score, cls) in enumerate(zip(boxes, scores, class_ids)):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{int(cls)}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

                # Write output
                out_buf = out_shms[cam].buf
                h2, w2 = annotated.shape[:2]
                if h2 <= max_h and w2 <= max_w:
                    dest = np.ndarray((max_h, max_w, 3), dtype=np.uint8, buffer=out_buf)
                    dest[:h2, :w2, :] = annotated
                    out_q.put({'cam': cam, 'shape': (h2, w2, 3), 'detections': detections, 'ts': time.time()})
                else:
                    ok, buf = cv2.imencode('.jpg', annotated)
                    if ok:
                        out_q.put({'cam': cam, 'jpeg': buf.tobytes(), 'detections': detections, 'ts': time.time()})

            except Exception as e:
                out_q.put({'error': f'worker processing exception: {e}'})

    except Exception as e:
        out_q.put({'error': f'worker init failed: {e}'})


class InferenceManager:
    def __init__(self, config):
        self.config = config
        self.worker_proc = None
        self.worker_in_q = None
        self.worker_out_q = None
        self.shm_in_names = [None, None]
        self.shm_out_names = [None, None]
        self.shms = [] 
        self.shm_max_w = config.INFERENCE_WIDTH
        self.shm_max_h = max(16, int(config.INFERENCE_WIDTH * 0.75))

    def start(self):
        # Create shared memory
        try:
            for i in range(2):
                shm_in = shared_memory.SharedMemory(create=True, size=self.shm_max_w * self.shm_max_h * 3)
                shm_out = shared_memory.SharedMemory(create=True, size=self.shm_max_w * self.shm_max_h * 3)
                self.shms.append(shm_in)
                self.shms.append(shm_out)
                self.shm_in_names[i] = shm_in.name
                self.shm_out_names[i] = shm_out.name
        except Exception as e:
            print("Error creating SHM:", e)

        self.worker_in_q = multiprocessing.Queue(maxsize=8)
        self.worker_out_q = multiprocessing.Queue(maxsize=16)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.worker_proc = multiprocessing.Process(
            target=worker_process_func,
            args=(
                self.worker_in_q, 
                self.worker_out_q, 
                self.config.MODEL_PATH, 
                device, 
                self.config.USE_HALF, 
                self.config.ENABLE_MASKS, 
                self.config.ALPHA, 
                self.config.INFERENCE_WIDTH, 
                self.shm_in_names, 
                self.shm_out_names, 
                self.shm_max_w, 
                self.shm_max_h,
                self.config.CONF_THRESHOLD_1,
                self.config.CONF_THRESHOLD_2
            ),
            daemon=True
        )
        self.worker_proc.start()
        print(f"Inference worker started pid={self.worker_proc.pid}")

    def stop(self):
        if self.worker_in_q:
            self.worker_in_q.put(None)
        if self.worker_proc:
            self.worker_proc.terminate()
        for s in self.shms:
            try:
                s.close()
                s.unlink()
            except: pass

    def send_frame(self, cam_idx, frame):
        # Resize and put in Shm
        if frame is None: return
        
        h, w = frame.shape[:2]
        small = frame
        if w > self.shm_max_w:
             scale = self.shm_max_w / w
             small = cv2.resize(frame, (0,0), fx=scale, fy=scale)
        
        # ensure fits
        sh, sw = small.shape[:2]
        if sh > self.shm_max_h or sw > self.shm_max_w:
            small = cv2.resize(small, (self.shm_max_w, self.shm_max_h))

        try:
             # Use pre-opened shm to avoid handle leaks
             shm = self.shms[cam_idx * 2]
             dest = np.ndarray((self.shm_max_h, self.shm_max_w, 3), dtype=np.uint8, buffer=shm.buf)
             dest[:small.shape[0], :small.shape[1], :] = small
             try:
                self.worker_in_q.put_nowait({'cam': cam_idx, 'shape': small.shape})
             except: pass
        except Exception as e:
            print("Error sending frame to worker:", e)

    def get_results(self):
        results = []
        if not self.worker_out_q: return results
        while True:
            try:
                res = self.worker_out_q.get_nowait()
                results.append(res)
            except:
                break
        return results

    def resolve_shm_image(self, cam_idx, shape):
        # Read back image from SHM
        try:
            # Use pre-opened shm to avoid handle leaks
            shm = self.shms[cam_idx * 2 + 1]
            h, w, c = shape
            arr = np.ndarray((self.shm_max_h, self.shm_max_w, 3), dtype=np.uint8, buffer=shm.buf)
            return arr[:h, :w, :].copy()
        except Exception:
            return None
