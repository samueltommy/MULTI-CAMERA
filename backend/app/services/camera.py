import cv2
import threading
import time
from typing import List, Optional

class RTSPReader:
    def __init__(self, url: str, index: int):
        self.url = url
        self.index = index
        self.frame = None
        self.timestamp = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def get_frame(self):
        with self.lock:
            return (self.frame.copy() if self.frame is not None else None), self.timestamp

    def _run(self):
        print(f"RTSP reader starting for {self.url}")
        cap = cv2.VideoCapture(self.url)
        if not cap.isOpened():
            print(f"Failed to open {self.url} in reader")
        else:
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        while self.running:
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.url)
                time.sleep(1.0)
                continue
            
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(self.url)
                continue
            
            with self.lock:
                self.frame = frame
                self.timestamp = time.time()
        
        cap.release()

class CameraManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CameraManager, cls).__new__(cls)
            cls._instance.readers = []
            cls._instance.annotated_frames = [None, None] # Holds the displayed/annotated frames
            cls._instance.annotated_locks = [threading.Lock(), threading.Lock()]
            cls._instance.display_tick = 0
            cls._instance.display_tick_lock = threading.Lock()
        return cls._instance

    def add_reader(self, url: str, index: int):
        reader = RTSPReader(url, index)
        self.readers.append(reader)
        reader.start()

    def get_raw_frame(self, index: int):
        if 0 <= index < len(self.readers):
            return self.readers[index].get_frame()
        return None, 0.0

    def set_annotated_frame(self, index: int, frame):
        with self.annotated_locks[index]:
            self.annotated_frames[index] = frame
    
    def get_annotated_frame(self, index: int):
        with self.annotated_locks[index]:
            return self.annotated_frames[index]

    def increment_tick(self):
        with self.display_tick_lock:
            self.display_tick += 1

    def get_tick(self):
        with self.display_tick_lock:
            return self.display_tick

camera_manager = CameraManager()
