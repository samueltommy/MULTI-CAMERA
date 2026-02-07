import asyncio
import json
import traceback
import threading
import cv2
import av
from fractions import Fraction
import time
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from app.services.camera import camera_manager

import cv2.aruco as aruco

# Initialize ArUco detector
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
aruco_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)

class AnnotatedTrack(MediaStreamTrack):
    kind = "video"
    def __init__(self, idx, fps=15, mode="pipeline"):
        super().__init__()
        self.idx = idx
        self.fps = fps
        self._start = None
        self._last_tick = -1
        self.mode = mode

    async def recv(self):
        if self._start is None:
            self._start = time.time()
        
        while True:
            if self.mode == "calibrate":
                frame, ts = camera_manager.get_raw_frame(self.idx)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Detect markers for visualization
                corners, ids, _ = aruco_detector.detectMarkers(frame)
                if ids is not None:
                    aruco.drawDetectedMarkers(frame, corners, ids)
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Throttling for calibrate mode to avoid high CPU
                await asyncio.sleep(1.0 / self.fps)
            else:
                cur_tick = camera_manager.get_tick()
                if cur_tick == self._last_tick:
                    await asyncio.sleep(0.01)
                    continue
                    
                frame = camera_manager.get_annotated_frame(self.idx)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                self._last_tick = cur_tick
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            video_frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            
            # PTS based on wall clock for steady streaming
            elapsed = time.time() - self._start
            video_frame.pts = int(elapsed * 90000)
            video_frame.time_base = Fraction(1, 90000)
            return video_frame

def start_webrtc_server():
    pcs = set()

    async def offer(request):
        # Allow CORS for all origins
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        }

        try:
            params = await request.json()
            offer = RTCSessionDescription(sdp=params['sdp'], type=params['type'])
            pc = RTCPeerConnection()
            pcs.add(pc)

            cam_idx = int(request.query.get('cam', '1')) - 1
            mode = request.query.get('mode', 'pipeline')
            
            await pc.setRemoteDescription(offer)
            track = AnnotatedTrack(cam_idx, fps=15, mode=mode)
            pc.addTrack(track)

            @pc.on('connectionstatechange')
            def on_state():
                if pc.connectionState == 'failed':
                    asyncio.ensure_future(pc.close())
            
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return web.Response(content_type='application/json', text=json.dumps({
                'sdp': pc.localDescription.sdp,
                'type': pc.localDescription.type
            }), headers=headers)
        except Exception as e:
            return web.Response(status=500, text=str(e), headers=headers)

    async def options(request):
        return web.Response(headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        })

    app = web.Application()
    app.router.add_post('/offer', offer)
    app.router.add_options('/offer', options)
    
    runner = web.AppRunner(app)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, '0.0.0.0', 8080)
    print("Starting WebRTC server on 8080")
    loop.run_until_complete(site.start())
    loop.run_forever()

def run_webrtc_thread():
    t = threading.Thread(target=start_webrtc_server, daemon=True)
    t.start()
