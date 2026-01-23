import os
from dotenv import load_dotenv

# Load env file from backend root
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

class Config:
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not set in .env")

    # Snapshots
    SNAPSHOT_DIR = os.environ.get('SNAPSHOT_DIR', '../snapshots')

    # Matching parameters
    MATCH_MAX_DIST_PX = float(os.environ.get('MATCH_MAX_DIST_PX', '80.0'))
    FUSE_TIME_WINDOW = float(os.environ.get('FUSE_TIME_WINDOW', '2.0'))
    ASSOC_WEIGHT_SIDE = float(os.environ.get('ASSOC_WEIGHT_SIDE', '1.0'))
    ASSOC_WEIGHT_TOP = float(os.environ.get('ASSOC_WEIGHT_TOP', '0.6'))
    ASSOC_TIME_DECAY = float(os.environ.get('ASSOC_TIME_DECAY', '3.0'))

    # Model
    MODEL_PATH = os.environ.get('MODEL_PATH', '../models/yolov8s-seg_openvino_model/')
    TARGET_FPS = int(os.environ.get('TARGET_FPS', '12'))
    INFERENCE_WIDTH = int(os.environ.get('INFERENCE_WIDTH', '480'))
    ENABLE_MASKS = os.environ.get('ENABLE_MASKS', '1').lower() in ('1', 'true', 'yes')
    ALPHA = float(os.environ.get('ALPHA', '0.35'))
    USE_HALF = os.environ.get('USE_HALF', '1').lower() in ('1', 'true', 'yes')
    TORCH_THREADS = int(os.environ.get('TORCH_THREADS', '1'))

    # Display / Motion
    DISPLAY_FPS = int(os.environ.get('DISPLAY_FPS', '15'))
    MOTION_DETECTION_WIDTH = int(os.environ.get('MOTION_DETECTION_WIDTH', '320'))
    MOTION_THRESHOLD = float(os.environ.get('MOTION_THRESHOLD', '0.02'))
    MOTION_HIGH_FPS = int(os.environ.get('MOTION_HIGH_FPS', '8'))
    MOTION_LOW_FPS = int(os.environ.get('MOTION_LOW_FPS', '1'))
    
    # Sync
    SYNC_TOL_MS = int(os.environ.get('SYNC_TOL_MS', '500'))

    # RTSP
    RTSP_URL_1 = os.environ.get('RTSP_URL_1')
    if not RTSP_URL_1:
         raise ValueError("RTSP_URL_1 not set in .env")
         
    RTSP_URL_2 = os.environ.get('RTSP_URL_2')
    if not RTSP_URL_2:
         raise ValueError("RTSP_URL_2 not set in .env")

    # Homography
    HOMOGRAPHY_PATH = os.environ.get('HOMOGRAPHY_PATH', '../homography_top_to_side.npy')

    # FFmpeg Relay
    START_FFMPEG_RELAY = os.environ.get('START_FFMPEG_RELAY', '0').lower() in ('1', 'true', 'yes')
    FFMPEG_FPS = int(os.environ.get('FFMPEG_FPS', str(TARGET_FPS)))
    OUTPUT_URL_1 = os.environ.get('OUTPUT_URL_1', 'rtmp://localhost/live/cam1')
    OUTPUT_URL_2 = os.environ.get('OUTPUT_URL_2', 'rtmp://localhost/live/cam2')

    # WebRTC / ICE
    STUN_URL = os.environ.get('STUN_URL', 'stun:stun.l.google.com:19302')
    TURN_URL = os.environ.get('TURN_URL')
    TURN_USER = os.environ.get('TURN_USER', '')
    TURN_PASS = os.environ.get('TURN_PASS', '')

settings = Config()
