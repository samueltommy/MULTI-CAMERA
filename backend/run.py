import sys
import os
import subprocess

def check_requirements():
    """Check if requirements are installed, if not try to install them."""
    print("Checking dependencies...")
    try:
        # We check one key package to see if we need to install
        import ultralytics
    except ImportError:
        print("Dependencies missing. Installing from requirements.txt...")
        req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
        if os.path.exists(req_path):
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_path], check=True)
            print("Dependencies installed successfully.")
        else:
            print("requirements.txt not found, skipping auto-install.")

# Add the backend directory to sys.path so 'app' can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import atexit
import multiprocessing
from tools.setup_db import setup_database

def run_migrations():
# ... (rest of the file)
    """Run database creation and alembic migrations."""
    print("Checking database and applying migrations...")
    try:
        # 1. Ensure DB exists (setup_db logic)
        setup_database()
        
        # 2. Run alembic upgrade head
        # We use sys.executable to ensure we use the same python/venv
        # -m alembic upgrade head is safer than just calling 'alembic'
        result = subprocess.run(
            [sys.executable, "-m", "alembic", "upgrade", "head"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Database is up to date.")
        else:
            print(f"Migration error: {result.stderr}")
            # If it's the duplicate table error, we might want to stamp it, 
            # but usually, manual intervention is better for errors.
            # However, for a "flawless" flow, we just report it.
    except Exception as e:
        print(f"Failed to auto-migrate: {e}")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 1. Check requirements
    check_requirements()
    
    # 2. Run setup and migrations before starting app
    run_migrations()
    
    # Delay import of app components until migrations are complete
    from app.main import create_app, cleanup
    
    # Register cleanup only in main process
    atexit.register(cleanup)
    
    app = create_app(start_services=True)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        pass
