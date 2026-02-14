import os
import cv2
import threading
import time
from datetime import datetime
from pathlib import Path
from app.core.config import settings

class VideoRecorder:
    def __init__(self, output_root='../videos'):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        self.session_dir = None
        self.raw_writers = [None, None]  # For cam 0, cam 1
        self.yolo_writers = [None, None]
        self.recording = False
        self.lock = threading.Lock()
        self.fps = settings.DISPLAY_FPS
        
    def start_session(self, fps: int = None):
        """Start a new recording session with timestamp-based folder.

        If `fps` is provided, writers will use that fps. Otherwise falls
        back to `settings.DISPLAY_FPS`.
        """
        with self.lock:
            if self.recording:
                print("[video_recorder] already recording, ignoring start_session")
                return

            # Configure fps
            if fps is None:
                self.fps = settings.DISPLAY_FPS
            else:
                self.fps = int(fps)

            # Create session folder with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = self.output_root / timestamp

            # Create subdirectories
            raw_dir = self.session_dir / "raw"
            yolo_dir = self.session_dir / "yolo"
            raw_dir.mkdir(parents=True, exist_ok=True)
            yolo_dir.mkdir(parents=True, exist_ok=True)

            # Initialize video writers (codec=mp4v)
            # We'll create writers on first frame (when we know frame size)
            self.raw_writers = [None, None]
            self.yolo_writers = [None, None]
            self.recording = True

            print(f"[video_recorder] session started at {self.session_dir} fps={self.fps}")
    
    def stop_session(self):
        """Stop recording and close all video writers"""
        with self.lock:
            if not self.recording:
                return
            
            self.recording = False
            
            # Release all writers
            for i in range(2):
                if self.raw_writers[i] is not None:
                    self.raw_writers[i].release()
                    self.raw_writers[i] = None
                if self.yolo_writers[i] is not None:
                    self.yolo_writers[i].release()
                    self.yolo_writers[i] = None
            
            print(f"[video_recorder] session stopped, videos saved to {self.session_dir}")
    
    def write_frame(self, cam_idx, raw_frame, annotated_frame):
        """Write both raw and annotated frames for a camera"""
        if not self.recording or cam_idx < 0 or cam_idx >= 2:
            return
        
        with self.lock:
            if not self.recording:
                return
            
            try:
                # Camera labels for filename
                cam_name = "top" if cam_idx == 0 else "side"
                
                # Write raw frame
                if raw_frame is not None:
                    if self.raw_writers[cam_idx] is None:
                        # Initialize writer on first frame
                        h, w = raw_frame.shape[:2]
                        raw_path = self.session_dir / "raw" / f"{cam_name}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.raw_writers[cam_idx] = cv2.VideoWriter(
                            str(raw_path), fourcc, float(self.fps), (w, h)
                        )
                        print(f"[video_recorder] opened raw video writer for cam {cam_idx} ({cam_name})")
                    
                    if self.raw_writers[cam_idx] is not None:
                        self.raw_writers[cam_idx].write(raw_frame)
                
                # Write annotated frame
                if annotated_frame is not None:
                    if self.yolo_writers[cam_idx] is None:
                        # Initialize writer on first frame
                        h, w = annotated_frame.shape[:2]
                        yolo_path = self.session_dir / "yolo" / f"{cam_name}.mp4"
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.yolo_writers[cam_idx] = cv2.VideoWriter(
                            str(yolo_path), fourcc, float(self.fps), (w, h)
                        )
                        print(f"[video_recorder] opened yolo video writer for cam {cam_idx} ({cam_name})")
                    
                    if self.yolo_writers[cam_idx] is not None:
                        self.yolo_writers[cam_idx].write(annotated_frame)
            
            except Exception as e:
                print(f"[video_recorder] error writing frames for cam {cam_idx}: {e}")

# Global instance
video_recorder = VideoRecorder()
