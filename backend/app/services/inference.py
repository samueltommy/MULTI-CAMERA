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
        print(f"[inference.worker] starting worker on device={dev} model={model_path}")
        
        # 1. Muat 2 model terpisah agar state tracking Cam 0 & Cam 1 tidak bentrok
        models = [YOLO(model_path), YOLO(model_path)]
        conf_thresholds = [conf_1, conf_2]
        
        try:
            for m in models:
                m.to(dev)
                if dev != 'cpu' and use_half:
                     m.model.half()
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
                # IMPORTANT: Read immediately to minimize race condition window
                arr = np.ndarray((max_h, max_w, 3), dtype=np.uint8, buffer=buf)
                small = arr[:h, :w, :].copy()
                
                if small is None:
                    continue

                t0 = time.time()
                try:
                    conf = conf_thresholds[cam] if cam < len(conf_thresholds) else 0.25
                    use_half_local = (use_half and dev != 'cpu')

                    # 2. Gunakan .track() dengan persist=True pada model spesifik kamera tersebut
                    results = models[cam].track(
                        small, 
                        device=dev, 
                        half=use_half_local, 
                        conf=conf, 
                        persist=True,      # Wajib True agar ID diingat antar frame
                        tracker="bytetrack.yaml", 
                        verbose=False
                    )

                    infer_ms = (time.time() - t0) * 1000.0
                    out_q.put({'metric_infer_ms': infer_ms})
                except Exception as e:
                    out_q.put({'error': f'inference failed: {e}'})
                    continue

                # We extract detections to send back to pipeline for smooth drawing
                boxes, scores, class_ids, track_ids = [], [], [], []
                
                try:
                    res = results[0]
                    boxes = res.boxes.xyxy.cpu().numpy()
                    scores = res.boxes.conf.cpu().numpy()
                    class_ids = res.boxes.cls.cpu().numpy()
                    
                    # 3. Ambil ID dari hasil tracking
                    if res.boxes.id is not None:
                         track_ids = res.boxes.id.int().cpu().numpy()
                    else:
                         track_ids = [-1] * len(boxes)

                    masks_data = None
                    # Cek apakah model mendeteksi mask segmentasi
                    if hasattr(res, 'masks') and res.masks is not None:
                        masks_data = res.masks.data.cpu().numpy() # Bentuk array 0 dan 1

                except Exception:
                    pass

                detections = []
                for i, (box, score, cls, tid) in enumerate(zip(boxes, scores, class_ids, track_ids)):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    mask_area = 0.0
                    if masks_data is not None and i < len(masks_data):
                        # Hitung total piksel yang benar-benar merupakan tubuh ayam
                        mask_area = np.sum(masks_data[i])
                    else:
                        # Fallback jika mask gagal/tidak ada: pakai luas kotak
                        mask_area = float((x2 - x1) * (y2 - y1))
                    detections.append({
                        'box': [x1, y1, x2, y2], 
                        'score': float(score), 
                        'cls': int(cls), 
                        'track_id': int(tid), 
                        'center': (cx, cy), 
                        'bottom_center': ((x1 + x2) / 2.0, float(y2)),
                        'mask_area': float(mask_area)
                    })

                # --- PERBAIKAN: DEFINISIKAN ANNOTATED ---
                # Kita perlu membuat gambar 'annotated' agar bisa dikirim ukurannya (shape) ke pipeline
                # dan juga untuk tampilan debug di Shared Memory.
                annotated = small.copy()
                
                # (Opsional) Gambar kotak pada annotated untuk debug view via SHM
                # Pipeline utama akan menggambar ulang sendiri, tapi ini berguna jika kita melihat raw stream worker
                for i, (box, score, cls, tid) in enumerate(zip(boxes, scores, class_ids, track_ids)):
                    x1, y1, x2, y2 = [int(v) for v in box]
                    # Kotak hijau tipis untuk debug
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    label = f"{int(tid)}" if tid != -1 else f"{int(cls)}"
                    cv2.putText(annotated, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # ----------------------------------------

                # Log detection count occasionally
                try:
                    if not hasattr(worker_process_func, '_frame_log_counter'):
                        worker_process_func._frame_log_counter = 0
                    worker_process_func._frame_log_counter += 1
                    if worker_process_func._frame_log_counter % 30 == 0:
                        print(f"[inference.worker] cam={cam} detections={len(detections)} infer_ms={infer_ms:.1f}ms")
                except Exception:
                    pass
                
                # Write output to SHM (Debug View)
                try:
                    out_buf = out_shms[cam].buf
                    h2, w2 = annotated.shape[:2]
                    if h2 <= max_h and w2 <= max_w:
                        dest = np.ndarray((max_h, max_w, 3), dtype=np.uint8, buffer=out_buf)
                        dest[:h2, :w2, :] = annotated
                except Exception:
                    pass 

                # Kirim hasil ke Pipeline (Shape wajib ada agar pipeline bisa scaling koordinat)
                out_q.put({
                    'cam': cam, 
                    'shape': annotated.shape[:3], # (h, w, c) - Variabel annotated sekarang sudah ada
                    'detections': detections, 
                    'ts': time.time()
                })

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
        # Track whether a frame for each camera is currently in-flight in the worker.
        # This prevents overwriting the per-camera SHM buffer while the worker
        # is still processing that camera's message.
        self._inflight = [False, False]
        # Timestamp when an in-flight message was queued (for simple staleness checks)
        self._inflight_ts = [0.0, 0.0]

    def start(self):
        # Clean up any stale SHM handles left from a previous run
        try:
            for s in list(self.shms):
                try:
                    s.close(); s.unlink()
                except Exception:
                    pass
        finally:
            self.shms = []

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

        # Reduced maxsize to prevent queue buildup that causes SHM overwrite race conditions
        self.worker_in_q = multiprocessing.Queue(maxsize=2)
        self.worker_out_q = multiprocessing.Queue(maxsize=16)

        # reset inflight trackers
        self._inflight = [False, False]
        self._inflight_ts = [0.0, 0.0]

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Prefer a plain .pt model in the repository (fresh reset) if available.
        # This keeps the public InferenceManager API unchanged while ensuring
        # the worker runs the requested `yolo26s-seg.pt` model by default.
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[3]
        pt_candidate = repo_root / "models" / "yolo26-ayam.pt"
        model_path_to_use = str(pt_candidate) if pt_candidate.exists() else self.config.MODEL_PATH
        if pt_candidate.exists():
            print(f"[inference] using PT model override: {model_path_to_use}")
        else:
            print(f"[inference] using configured model path: {model_path_to_use}")

        self.worker_proc = multiprocessing.Process(
            target=worker_process_func,
            args=(
                self.worker_in_q,
                self.worker_out_q,
                model_path_to_use,
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
        print(f"Inference worker started pid={self.worker_proc.pid} (model={model_path_to_use})")

    def stop(self):
        # Signal worker to exit gracefully, then force-terminate if still alive.
        try:
            if self.worker_in_q:
                try:
                    self.worker_in_q.put_nowait(None)
                except Exception:
                    pass
            if self.worker_proc:
                try:
                    # give worker a short chance to exit
                    self.worker_proc.join(timeout=0.5)
                except Exception:
                    pass
                try:
                    if self.worker_proc.is_alive():
                        self.worker_proc.terminate()
                        self.worker_proc.join(timeout=0.5)
                except Exception:
                    pass
        finally:
            self.worker_proc = None
            self.worker_in_q = None
            self.worker_out_q = None

        # Close and unlink any shared memory segments and reset trackers
        try:
            for s in list(self.shms):
                try:
                    s.close()
                    s.unlink()
                except Exception:
                    pass
        finally:
            self.shms = []
            self.shm_in_names = [None, None]
            self.shm_out_names = [None, None]
            self._inflight = [False, False]
            self._inflight_ts = [0.0, 0.0]

    def send_frame(self, cam_idx, frame):
        if frame is None: return

        # If worker not running, try to restart it so frames don't silently stop being processed
        if not self.worker_proc or not getattr(self.worker_proc, 'is_alive', lambda: False)():
            print("[inference] worker not alive — restarting from send_frame")
            try:
                self.start()
            except Exception as e:
                print(f"[inference] failed to restart worker: {e}")
                return

        # Prevent overwriting the same camera SHM while that camera's frame is still being processed
        if self._inflight[cam_idx]:
            # stale inflight -> try to clear if too old
            import time as _t
            if self._inflight_ts[cam_idx] and (_t.time() - self._inflight_ts[cam_idx]) > 5.0:
                print(f"[inference] inflight for cam={cam_idx} stale, clearing")
                self._inflight[cam_idx] = False
                self._inflight_ts[cam_idx] = 0.0
            else:
                return

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
             
             # Try put. If full, ignore.
             try:
                self.worker_in_q.put_nowait({'cam': cam_idx, 'shape': small.shape})
                # mark this camera as in-flight until a worker result for this cam is received
                import time as _time
                self._inflight[cam_idx] = True
                self._inflight_ts[cam_idx] = _time.time()
             except Exception:
                pass # Queue full, drop frame safely
        except Exception as e:
            print("Error sending frame to worker:", e)

    def get_results(self):
        results = []
        if not self.worker_out_q: return results
        while True:
            try:
                res = self.worker_out_q.get_nowait()
                # If worker returned a result for a camera, mark that camera as no longer in-flight
                try:
                    cam = res.get('cam')
                    if cam is not None and 0 <= cam < len(self._inflight):
                        self._inflight[cam] = False
                        self._inflight_ts[cam] = 0.0
                except Exception:
                    pass
                results.append(res)
            except Exception:
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
