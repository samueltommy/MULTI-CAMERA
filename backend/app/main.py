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

        # 2. Pipeline (Inference + Fusion)
        pipeline_service.start()

        # 3. WebRTC Server
        run_webrtc_thread()

    return app

def cleanup():
    print("Cleaning up...")
    pipeline_service.stop()
    for reader in camera_manager.readers:
        reader.stop()

atexit.register(cleanup)

if __name__ == '__main__':
    # Windows support for direct execution of main.py
    multiprocessing.freeze_support()
    
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
else:
    # When imported by run.py or WSGI, just create the app object
    # BUT DO NOT START BACKGROUND THREADS YET!
    pass
