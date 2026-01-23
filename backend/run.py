import sys
import os

# Add the backend directory to sys.path so 'app' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.main import create_app, cleanup
import atexit
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    app = create_app(start_services=True)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        pass
