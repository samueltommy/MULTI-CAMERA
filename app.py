from flask import Flask, render_template
import cv2
from ultralytics import YOLO
import threading
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch
import numpy as np
import asyncio
import json
import av
from fractions import Fraction
import traceback
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
import subprocess
import os
import shlex
import multiprocessing
from multiprocessing import shared_memory
import atexit
from statistics import mean

app = Flask(__name__)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO model
# Configurable runtime options (environment overrides for edge tuning)
MODEL_PATH = os.environ.get('MODEL_PATH', 'yolov8s-seg.pt')
TARGET_FPS = int(os.environ.get('TARGET_FPS', '12'))
INFERENCE_WIDTH = int(os.environ.get('INFERENCE_WIDTH', '480'))
ENABLE_MASKS = os.environ.get('ENABLE_MASKS', '1').lower() in ('1', 'true', 'yes')
ALPHA = float(os.environ.get('ALPHA', '0.35'))
FFMPEG_FPS = int(os.environ.get('FFMPEG_FPS', str(TARGET_FPS)))
USE_HALF = os.environ.get('USE_HALF', '1').lower() in ('1', 'true', 'yes')
# control whether to start ffmpeg relays (can consume CPU/network)
START_FFMPEG_RELAY = os.environ.get('START_FFMPEG_RELAY', '0').lower() in ('1', 'true', 'yes')
# Display and motion-adaptive inference settings
DISPLAY_FPS = int(os.environ.get('DISPLAY_FPS', '15'))
MOTION_DETECTION_WIDTH = int(os.environ.get('MOTION_DETECTION_WIDTH', '320'))
MOTION_THRESHOLD = float(os.environ.get('MOTION_THRESHOLD', '0.02'))  # fraction of changed pixels
MOTION_HIGH_FPS = int(os.environ.get('MOTION_HIGH_FPS', '8'))
MOTION_LOW_FPS = int(os.environ.get('MOTION_LOW_FPS', '1'))
# limit torch threads on CPU edge devices
TORCH_THREADS = int(os.environ.get('TORCH_THREADS', '1'))
try:
    torch.set_num_threads(TORCH_THREADS)
except Exception:
    pass

print(f"Using model: {MODEL_PATH}, target_fps={TARGET_FPS}, inference_width={INFERENCE_WIDTH}, masks={ENABLE_MASKS}, half={USE_HALF}")
model = None

# RTSP URLs
rtsp1 = 'rtsp://admin:admin123456@192.168.1.8:554/ch=1?subtype=0'
rtsp2 = 'rtsp://admin:admin123456@192.168.1.11:554/ch=1?subtype=0'

# Global variables for annotated (streamed) frames
frames1 = [None]
frames2 = [None]
lock1 = threading.Lock()
lock2 = threading.Lock()
# display synchronization
display_frame_id = 0
display_lock = threading.Lock()
display_frame_ids = [0, 0]
# display ticker (decoupled from inference) to provide smooth playback
display_tick = 0
display_tick_lock = threading.Lock()

# Raw frames and timestamps from RTSP readers (low-latency buffers)
raw_frames = [None, None]
raw_timestamps = [0.0, 0.0]
raw_locks = [threading.Lock(), threading.Lock()]

# Multiprocessing queues for worker communication (set in __main__)
worker_in_q = None
worker_out_q = None
# shared memory segments (created in __main__)
shm_in_names = [None, None]
shm_out_names = [None, None]
shm_max_w = None
shm_max_h = None
worker_proc = None

# metrics
metrics = {
    'last_inference_ms': 0.0,
    'inference_samples_ms': [],
    'worker_queue_len': 0,
    'processed_frames': 0,
}
metrics_lock = threading.Lock()

# Trackers
tracker1 = DeepSort(max_age=30, n_init=3, nn_budget=100)
tracker2 = DeepSort(max_age=30, n_init=3, nn_budget=100)

def get_color(class_id):
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 128, 128),# Gray
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
    ]
    return colors[class_id % len(colors)]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def rtsp_reader(rtsp_url, idx):
    """Fast reader that only updates the latest raw frame and timestamp."""
    print(f"RTSP reader starting for {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Failed to open {rtsp_url} in reader")
    else:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    while True:
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(rtsp_url)
            time.sleep(1.0)
            continue
        ret, frame = cap.read()
        if not ret:
            # try reopen
            cap.release()
            time.sleep(0.5)
            cap = cv2.VideoCapture(rtsp_url)
            continue
        with raw_locks[idx]:
            raw_frames[idx] = frame
            raw_timestamps[idx] = time.time()


def worker_process(in_q, out_q, model_path, device_name, use_half, enable_masks, alpha, inference_width, shm_in_names_arg, shm_out_names_arg, shm_max_w_arg, shm_max_h_arg):
    """Background inference worker running in a separate process.
    Receives dicts: {'cam': idx, 'jpeg': bytes, 'orig_size': (w,h)}
    Returns dicts: {'cam': idx, 'jpeg': bytes}
    """
    try:
        import torch as _torch
        from ultralytics import YOLO as _YOLO
        import cv2 as _cv2
        import numpy as _np
    except Exception as e:
        out_q.put({'error': f'worker import failed: {e}'})
        return

    dev = device_name
    try:
        model = _YOLO(model_path).to(dev)
    except Exception:
        # fallback to CPU
        dev = 'cpu'
        model = _YOLO(model_path).to(dev)

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
            # read from shared memory; msg includes 'shape'
            shape = msg.get('shape')
            if shape is None:
                continue
            h, w, c = shape
            # view the shared memory as numpy array with max dims then slice
            buf = in_shms[cam].buf
            arr = _np.ndarray((max_h, max_w, 3), dtype=_np.uint8, buffer=buf)
            small = arr[:h, :w, :].copy()
            if small is None:
                continue
            # run model
            try:
                t0 = time.time()
                with _torch.no_grad():
                    results = model(small, device=dev, half=use_half)
                infer_ms = (time.time() - t0) * 1000.0
                out_q.put({'metric_infer_ms': infer_ms})
            except Exception:
                results = model(small, device=dev, half=False)

            annotated = small.copy()
            try:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy()
            except Exception:
                boxes, scores, class_ids = [], [], []

            # draw masks on small if enabled (use per-class color and alpha blending)
            if enable_masks and getattr(results[0], 'masks', None) is not None:
                masks = results[0].masks.xy
                palette = [
                    (255, 0, 0),    # Red
                    (0, 255, 0),    # Green
                    (0, 0, 255),    # Blue
                    (255, 255, 0),  # Yellow
                    (255, 0, 255),  # Magenta
                    (0, 255, 255),  # Cyan
                    (128, 128, 128),# Gray
                    (255, 165, 0),  # Orange
                    (128, 0, 128),  # Purple
                    (0, 128, 128),  # Teal
                ]
                for i, mask in enumerate(masks):
                    try:
                        try:
                            class_id_int = int(class_ids[i])
                        except Exception:
                            class_id_int = 0
                        color = palette[class_id_int % len(palette)]
                        small_h, small_w = annotated.shape[:2]
                        mask_pts = (mask * 1).astype(_np.int32)
                        mask_img = _np.zeros((small_h, small_w), dtype=_np.uint8)
                        _cv2.fillPoly(mask_img, [mask_pts], 255)
                        colored_mask = _np.zeros_like(annotated, dtype=_np.uint8)
                        colored_mask[:] = color
                        blended = _cv2.addWeighted(annotated, 1.0 - alpha, colored_mask, alpha, 0)
                        annotated[mask_img == 255] = blended[mask_img == 255]
                    except Exception:
                        pass

            for i, (box, score, cls) in enumerate(zip(boxes, scores, class_ids)):
                x1, y1, x2, y2 = [int(v) for v in box]
                color = (0, 255, 0)
                _cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{int(cls)} {score:.2f}"
                _cv2.putText(annotated, label, (x1, max(0, y1-6)), _cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # write annotated into out shared memory and notify via out_q
            try:
                out_buf = out_shms[cam].buf
                h2, w2 = annotated.shape[:2]
                # ensure fits into out shm
                if h2 <= max_h and w2 <= max_w:
                    dest = _np.ndarray((max_h, max_w, 3), dtype=_np.uint8, buffer=out_buf)
                    dest[:h2, :w2, :] = annotated
                    out_q.put({'cam': cam, 'shape': (h2, w2, 3)})
                else:
                    # fallback: encode jpeg and send
                    ok, buf = _cv2.imencode('.jpg', annotated, [int(_cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        out_q.put({'cam': cam, 'jpeg': buf.tobytes()})
            except Exception:
                out_q.put({'error': 'worker output write failed'})
        except Exception:
            out_q.put({'error': 'worker processing exception'})


def inference_tick_loop(target_fps=8, inference_width=640, alpha=0.45):
    """Synchronized tick loop: on each tick, take latest raw frames for both streams,
    run inference at reduced resolution and update annotated frames so both outputs
    are processed at the same time (keeps streams in sync)."""
    # use env defaults if provided
    target_fps = TARGET_FPS if target_fps is None else target_fps
    inference_width = INFERENCE_WIDTH if inference_width is None else inference_width
    alpha = ALPHA if alpha is None else alpha
    tick_interval = 1.0 / float(target_fps)
    print(f"Starting inference tick loop at {target_fps} FPS, width={inference_width}, masks={ENABLE_MASKS}")
    global display_frame_id
    global display_frame_ids
    # motion detection previous frames for each camera
    prev_small = [None, None]
    current_target_fps = target_fps
    while True:
        t0 = time.time()
        # copy raw frames quickly
        with raw_locks[0]:
            f0 = raw_frames[0].copy() if raw_frames[0] is not None else None
            ts0 = raw_timestamps[0]
        with raw_locks[1]:
            f1 = raw_frames[1].copy() if raw_frames[1] is not None else None
            ts1 = raw_timestamps[1]

        # If neither has a frame yet, wait
        if f0 is None and f1 is None:
            time.sleep(0.01)
            continue

        # Basic motion detection to adapt inference frequency (cheap)
        motion_present = False
        for idx, frame in enumerate((f0, f1)):
            if frame is None:
                continue
            # downscale for motion detection
            h, w = frame.shape[:2]
            if w > MOTION_DETECTION_WIDTH:
                scale = MOTION_DETECTION_WIDTH / float(w)
                ms_w = int(w * scale)
                ms_h = int(h * scale)
                small_for_motion = cv2.resize(frame, (ms_w, ms_h))
            else:
                small_for_motion = cv2.resize(frame, (frame.shape[1], frame.shape[0]))
            gray = cv2.cvtColor(small_for_motion, cv2.COLOR_BGR2GRAY)
            if prev_small[idx] is not None:
                diff = cv2.absdiff(gray, prev_small[idx])
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_score = (np.count_nonzero(th) / float(th.size))
                if motion_score >= MOTION_THRESHOLD:
                    motion_present = True
            prev_small[idx] = gray

        # choose inference FPS based on motion
        desired_fps = MOTION_HIGH_FPS if motion_present else MOTION_LOW_FPS
        current_target_fps = desired_fps
        tick_interval = 1.0 / float(current_target_fps)

        # Send downscaled frames to the inference worker and collect annotated results
        temp_outs = [None, None]
        # send small frames to worker if available
        for idx, frame in enumerate((f0, f1)):
            if frame is None:
                continue
            h, w = frame.shape[:2]
            if w > inference_width:
                scale = inference_width / float(w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                small = cv2.resize(frame, (new_w, new_h))
            else:
                small = frame.copy()

            # write small frame into shared memory input buffer and notify worker
            if worker_in_q is not None and shm_in_names[0] is not None:
                try:
                    # ensure small fits in shared mem dims
                    sw, sh = small.shape[1], small.shape[0]
                    if sw > shm_max_w or sh > shm_max_h:
                        # scale again to fit
                        fit_scale = min(shm_max_w / float(sw), shm_max_h / float(sh))
                        new_w2 = int(sw * fit_scale)
                        new_h2 = int(sh * fit_scale)
                        small = cv2.resize(small, (new_w2, new_h2))
                        sw, sh = new_w2, new_h2
                    # write into shared memory
                    shm = shared_memory.SharedMemory(name=shm_in_names[idx])
                    dest = np.ndarray((shm_max_h, shm_max_w, 3), dtype=np.uint8, buffer=shm.buf)
                    dest[:small.shape[0], :small.shape[1], :] = small
                    # send metadata
                    try:
                        worker_in_q.put_nowait({'cam': idx, 'shape': (small.shape[0], small.shape[1], 3)})
                    except Exception:
                        pass
                except Exception as e:
                    print('Error writing to shm input:', e)

        # collect any worker results available and map them back to full-res
        if worker_out_q is not None:
            while True:
                try:
                    res = worker_out_q.get_nowait()
                except Exception:
                    break
                if not res:
                    continue
                if 'error' in res:
                    print('Worker error:', res.get('error'))
                    continue
                # worker may send inference metric
                if 'metric_infer_ms' in res:
                    with metrics_lock:
                        metrics['last_inference_ms'] = res['metric_infer_ms']
                        metrics['inference_samples_ms'].append(res['metric_infer_ms'])
                        if len(metrics['inference_samples_ms']) > 50:
                            metrics['inference_samples_ms'].pop(0)
                    continue
                cam = int(res.get('cam', 0))
                shape = res.get('shape')
                if shape:
                    h2, w2, _ = shape
                    try:
                        shm_out = shared_memory.SharedMemory(name=shm_out_names[cam])
                        arr = np.ndarray((shm_max_h, shm_max_w, 3), dtype=np.uint8, buffer=shm_out.buf)
                        ann_small = arr[:h2, :w2, :].copy()
                        orig_frame = f0 if cam == 0 else f1
                        if orig_frame is None:
                            continue
                        annotated_orig = cv2.resize(ann_small, (orig_frame.shape[1], orig_frame.shape[0]))
                        temp_outs[cam] = annotated_orig
                        with metrics_lock:
                            metrics['processed_frames'] += 1
                        with metrics_lock:
                            metrics['processed_frames'] += 1
                    except Exception as e:
                        print('Error reading shm out:', e)
                else:
                    jpeg = res.get('jpeg')
                    if jpeg is None:
                        continue
                    try:
                        arr = np.frombuffer(jpeg, dtype=np.uint8)
                        ann_small = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if ann_small is None:
                            continue
                        orig_frame = f0 if cam == 0 else f1
                        if orig_frame is None:
                            continue
                        annotated_orig = cv2.resize(ann_small, (orig_frame.shape[1], orig_frame.shape[0]))
                        temp_outs[cam] = annotated_orig
                    except Exception as e:
                        print('Error decoding worker result:', e)

        # atomically publish both annotated frames so viewers see same tick
        # publish annotated frames (atomic) so viewers see latest processed frame
        with display_lock:
            if temp_outs[0] is not None:
                with lock1:
                    frames1[0] = temp_outs[0]
            if temp_outs[1] is not None:
                with lock2:
                    frames2[0] = temp_outs[1]
            # update display_frame_ids for bookkeeping (not used for playback)
            display_frame_id += 1
            display_frame_ids[0] = display_frame_id
            display_frame_ids[1] = display_frame_id

        # sleep until next tick
        elapsed = time.time() - t0
        to_sleep = max(0.0, tick_interval - elapsed)
        time.sleep(to_sleep)

# NOTE: MJPEG generator/HTTP streaming removed — using WebRTC / aiortc for low-latency


# FFmpeg relay: push annotated frames to an output URL (rtmp/rtsp)
def start_ffmpeg_relay(idx, output_url, fps=15, preset='veryfast'):
    """Start an ffmpeg subprocess that reads rawvideo BGR24 from stdin and pushes H.264 to output_url.
    output_url should be a valid RTMP/RTSP endpoint (e.g. rtmp://localhost/live/cam1).
    """
    def relay():
        print(f"Starting ffmpeg relay for cam{idx+1} -> {output_url}")
        proc = None
        while True:
            # wait for a frame to know size
            with (lock1 if idx == 0 else lock2):
                frame = frames1[0] if idx == 0 else frames2[0]
            if frame is None:
                time.sleep(0.5)
                continue
            h, w = frame.shape[:2]
            cmd = (
                f"ffmpeg -y -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - "
                f"-c:v libx264 -preset {preset} -tune zerolatency -pix_fmt yuv420p -f flv {shlex.quote(output_url)}"
            )
            try:
                proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE)
            except Exception as e:
                print('Failed to start ffmpeg:', e)
                time.sleep(2.0)
                continue

            try:
                while True:
                    with (lock1 if idx == 0 else lock2):
                        frame = frames1[0] if idx == 0 else frames2[0]
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    # ensure size matches
                    if frame.shape[1] != w or frame.shape[0] != h:
                        # restart ffmpeg with new size
                        print('Frame size changed, restarting ffmpeg relay')
                        try:
                            proc.stdin.close()
                        except Exception:
                            pass
                        proc.terminate()
                        break
                    # write raw BGR frame
                    try:
                        proc.stdin.write(frame.tobytes())
                    except Exception as e:
                        print('ffmpeg write error:', e)
                        try:
                            proc.stdin.close()
                        except Exception:
                            pass
                        proc.terminate()
                        break
                    time.sleep(1.0 / float(fps))
            finally:
                try:
                    proc.kill()
                except Exception:
                    pass
                proc = None
            time.sleep(1.0)

    t = threading.Thread(target=relay, daemon=True)
    t.start()
    return t


def get_default_output_urls():
    # configurable via environment variables, otherwise default local RTMP paths
    out1 = os.environ.get('OUTPUT_URL_1', 'rtmp://localhost/live/cam1')
    out2 = os.environ.get('OUTPUT_URL_2', 'rtmp://localhost/live/cam2')
    return out1, out2


# ICE / TURN config endpoint
@app.route('/ice')
def ice_config():
    # Provide ICE servers (STUN/TURN) to clients. Configure via env vars for TURN.
    stun = os.environ.get('STUN_URL', 'stun:stun.l.google.com:19302')
    turn_url = os.environ.get('TURN_URL')
    ice = [{'urls': stun}]
    if turn_url:
        turn_user = os.environ.get('TURN_USER', '')
        turn_pass = os.environ.get('TURN_PASS', '')
        ice.append({'urls': turn_url, 'username': turn_user, 'credential': turn_pass})
    return { 'iceServers': ice }


class AnnotatedTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, idx, fps=15):
        super().__init__()
        self.idx = idx
        self.fps = fps
        self._start = None
        self._last_display_id = -1

    async def recv(self):
        # produce frames from annotated buffer
        if self._start is None:
            self._start = time.time()
            self._pts = 0
        while True:
            # wait for the steady display tick (decoupled from inference)
            with display_tick_lock:
                cur_tick = display_tick
            if cur_tick == self._last_display_id:
                await asyncio.sleep(0.005)
                continue
            # get the latest annotated frame (may be older than tick but provides smoothness)
            with (lock1 if self.idx == 0 else lock2):
                buf = frames1[0] if self.idx == 0 else frames2[0]
                frame = buf.copy() if buf is not None else None
            if frame is None:
                await asyncio.sleep(0.005)
                continue
            # mark we've consumed this display tick
            self._last_display_id = cur_tick

            # convert BGR to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            # increment pts in 90k clock (common for video)
            self._pts += int(90000 / float(self.fps))
            video_frame.pts = self._pts
            video_frame.time_base = Fraction(1, 90000)
            await asyncio.sleep(1.0 / float(self.fps))
            return video_frame

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/webrtc')
def webrtc_page():
    return render_template('webrtc.html')

# MJPEG endpoints removed — use WebRTC via / (index) which auto-starts streams.

if __name__ == '__main__':
    # Start low-latency RTSP reader threads and the synchronized inference tick loop
    threading.Thread(target=rtsp_reader, args=(rtsp1, 0), daemon=True).start()
    threading.Thread(target=rtsp_reader, args=(rtsp2, 1), daemon=True).start()
    # start inference worker process and tick loop using environment-configured defaults
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except Exception:
        pass

    # prepare shared memory dims
    shm_max_w = INFERENCE_WIDTH
    shm_max_h = max(16, int(INFERENCE_WIDTH * 0.75))
    # create shared memory segments for input and output for each camera
    shms_in = []
    shms_out = []
    for i in range(2):
        shm_in = shared_memory.SharedMemory(create=True, size=shm_max_w * shm_max_h * 3)
        shm_out = shared_memory.SharedMemory(create=True, size=shm_max_w * shm_max_h * 3)
        shms_in.append(shm_in)
        shms_out.append(shm_out)
        shm_in_names[i] = shm_in.name
        shm_out_names[i] = shm_out.name

    worker_in_q = multiprocessing.Queue(maxsize=8)
    worker_out_q = multiprocessing.Queue(maxsize=16)
    worker_proc = multiprocessing.Process(
        target=worker_process,
        args=(worker_in_q, worker_out_q, MODEL_PATH, device, USE_HALF, ENABLE_MASKS, ALPHA, INFERENCE_WIDTH, shm_in_names, shm_out_names, shm_max_w, shm_max_h),
        daemon=True,
    )
    worker_proc.start()
    print('Inference worker process started (pid=%s)' % worker_proc.pid)

    # register cleanup for shared memory and worker
    def _cleanup():
        try:
            if worker_in_q is not None:
                worker_in_q.put(None)
        except Exception:
            pass
        try:
            worker_proc.terminate()
        except Exception:
            pass
        for s in shms_in + shms_out:
            try:
                s.close()
                s.unlink()
            except Exception:
                pass

    atexit.register(_cleanup)

    # start watchdog thread to restart worker if it dies
    def worker_watchdog():
        global worker_proc
        while True:
            time.sleep(1.0)
            try:
                alive = worker_proc.is_alive() if worker_proc is not None else False
            except Exception:
                alive = False
            if not alive:
                print('Worker process died, restarting...')
                try:
                    if worker_proc is not None:
                        worker_proc.terminate()
                except Exception:
                    pass
                worker_proc = multiprocessing.Process(
                    target=worker_process,
                    args=(worker_in_q, worker_out_q, MODEL_PATH, device, USE_HALF, ENABLE_MASKS, ALPHA, INFERENCE_WIDTH, shm_in_names, shm_out_names, shm_max_w, shm_max_h),
                    daemon=True,
                )
                worker_proc.start()
                print('Worker restarted (pid=%s)' % worker_proc.pid)

    threading.Thread(target=worker_watchdog, daemon=True).start()

    threading.Thread(target=inference_tick_loop, args=(None, None, None), daemon=True).start()
    # start aiohttp + aiortc server for WebRTC offers
    def start_aiohttp():
        pcs = set()

        async def index(request):
            return web.Response(text="aiortc server running")

        async def offer(request):
            headers = {'Access-Control-Allow-Origin': '*'}
            try:
                params = await request.json()
                offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
                pc = RTCPeerConnection()
                pcs.add(pc)

                # choose which camera by query param ?cam=1 or cam=2
                cam_idx = int(request.query.get('cam', '1')) - 1

                # set remote description first, then add our track
                await pc.setRemoteDescription(offer)
                track = AnnotatedTrack(cam_idx, fps=15)
                pc.addTrack(track)

                # log connection state and close on failure
                def on_conn_state():
                    print('PC connection state:', pc.connectionState)
                    if pc.connectionState == 'failed':
                        asyncio.ensure_future(pc.close())
                pc.on('connectionstatechange', on_conn_state)

                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                resp_obj = {'sdp': pc.localDescription.sdp, 'type': pc.localDescription.type}
                return web.Response(content_type='application/json', text=json.dumps(resp_obj), headers=headers)
            except Exception as e:
                tb = traceback.format_exc()
                print('Exception in /offer:', tb)
                err = {'error': str(e), 'trace': tb}
                return web.Response(status=500, content_type='application/json', text=json.dumps(err), headers=headers)

        async def offer_options(request):
            headers = {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type'
            }
            return web.Response(status=200, headers=headers)

        app_aio = web.Application()
        app_aio.router.add_get('/', index)
        app_aio.router.add_post('/offer', offer)
        app_aio.router.add_options('/offer', offer_options)

        web.run_app(app_aio, host='0.0.0.0', port=8080)

    threading.Thread(target=start_aiohttp, daemon=True).start()

    # start steady display ticker to provide smooth playback independent of inference
    def display_ticker():
        global display_tick
        interval = 1.0 / float(DISPLAY_FPS)
        while True:
            time.sleep(interval)
            with display_tick_lock:
                display_tick += 1

    threading.Thread(target=display_ticker, daemon=True).start()

    # start ffmpeg relays to re-broadcast annotated frames only if explicitly enabled
    out1, out2 = get_default_output_urls()
    if START_FFMPEG_RELAY:
        start_ffmpeg_relay(0, out1, fps=FFMPEG_FPS)
        start_ffmpeg_relay(1, out2, fps=FFMPEG_FPS)
    else:
        print('FFmpeg relays disabled (START_FFMPEG_RELAY=0). Set env var to enable.')

    @app.route('/streams')
    def streams():
        # expose outputs and basic metrics
        with metrics_lock:
            avg_ms = mean(metrics['inference_samples_ms']) if metrics['inference_samples_ms'] else 0.0
            last_ms = metrics['last_inference_ms']
            processed = metrics['processed_frames']
        qlen = 0
        try:
            qlen = worker_in_q.qsize() if worker_in_q is not None else 0
        except Exception:
            qlen = 0
        return { 'outputs': { 'cam1': out1, 'cam2': out2 }, 'metrics': { 'last_inference_ms': last_ms, 'avg_inference_ms': avg_ms, 'worker_queue_len': qlen, 'processed_frames': processed } }

    @app.route('/metrics')
    def metrics_route():
        with metrics_lock:
            avg_ms = mean(metrics['inference_samples_ms']) if metrics['inference_samples_ms'] else 0.0
            last_ms = metrics['last_inference_ms']
            processed = metrics['processed_frames']
        qlen = 0
        try:
            qlen = worker_in_q.qsize() if worker_in_q is not None else 0
        except Exception:
            qlen = 0
        return { 'last_inference_ms': last_ms, 'avg_inference_ms': avg_ms, 'worker_queue_len': qlen, 'processed_frames': processed }

    # start Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)