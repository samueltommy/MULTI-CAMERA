import os
import atexit
import multiprocessing
from flask import Flask
from app.core.config import settings
from app.api import rest
from app.api.webrtc import run_webrtc_thread
from app.services.camera import camera_manager
from app.services.pipeline import pipeline_service
from app.database.session import engine
from app.database.models import Base

def create_app(start_services=True):
    # Template folder is at the project root: ../../templates
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'templates'))
    app = Flask(__name__, template_folder=template_dir)
    app.register_blueprint(rest.api)

    if start_services:
        # Start Services
        # 1. Cameras
        camera_manager.add_reader(settings.RTSP_URL_1, 0)
        camera_manager.add_reader(settings.RTSP_URL_2, 1)
        # 2. Pipeline (always running to capture/display raw frames)
        #    Inference will be enabled on-demand when the user clicks `/trigger`.
        pipeline_service.start(inference_enabled=False)
        print("[app] pipeline started with inference DISABLED (raw frames only)")

        # 3. WebRTC Server
        run_webrtc_thread()

    return app

_cleanup_done = False
def cleanup():
    global _cleanup_done
    if _cleanup_done:
        return
    _cleanup_done = True
    print("Cleaning up...")
    pipeline_service.stop()
    for reader in camera_manager.readers:
        reader.stop()

if __name__ == '__main__':
    # Windows support for direct execution of main.py
    multiprocessing.freeze_support()
    
    app = create_app()
    # Register cleanup only in main process
    atexit.register(cleanup)
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
else:
    # When imported by run.py or WSGI, just create the app object
    # BUT DO NOT START BACKGROUND THREADS YET!
    pass
