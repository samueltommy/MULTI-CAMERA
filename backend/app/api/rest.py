from flask import Blueprint, jsonify, request, send_from_directory, render_template, current_app
from app.services.fusion import fusion_service
from app.services.pipeline import pipeline_service
from app.services.calibration import calibration_service
from app.core.config import settings
import numpy as np
import os
import time

api = Blueprint('api', __name__)

@api.route('/')
def index():
    return render_template('index.html')

@api.route('/trigger', methods=['POST'])
def trigger_session():
    duration = int(request.json.get('duration', 60))
    pipeline_service.start_session(duration)
    return jsonify({'status': 'started', 'duration': duration})

@api.route('/trigger/status')
def trigger_status():
    return jsonify({
        'active': pipeline_service.session_active,
        'best_count': pipeline_service.best_session_result['count'],
        'time_left': max(0, pipeline_service.session_end_time - time.time()) if pipeline_service.session_active else 0,
        'has_homography': fusion_service.H is not None
    })

@api.route('/ice')
def ice_config():
    # Provide ICE servers (STUN/TURN) to clients.
    ice = [{'urls': settings.STUN_URL}]
    # Add TURN if needed/available in settings later
    return jsonify({ 'iceServers': ice })

@api.route('/webrtc')
def webrtc_page():
    return render_template('webrtc.html')

@api.route('/calibrate')
def calibrate_page():
    return render_template('calibrate.html')

@api.route('/gallery')
def gallery_page():
    return render_template('gallery.html')

@api.route('/calibrate/reset', methods=['POST'])
def calibrate_reset():
    calibration_service.reset()
    return jsonify({'ok': True})

@api.route('/calibrate/capture', methods=['POST'])
def calibrate_capture():
    success, msg = calibration_service.capture_points()
    if success:
        return jsonify({'ok': True, 'message': msg, 'count': len(calibration_service.src_pts)})
    else:
        return jsonify({'ok': False, 'error': msg}), 400

@api.route('/calibrate/finish', methods=['POST'])
def calibrate_finish():
    data = request.get_json() or {}
    name = data.get('name')
    success, msg = calibration_service.compute_and_save(name=name)
    if success:
        return jsonify({'ok': True, 'message': msg})
    else:
        return jsonify({'ok': False, 'error': msg}), 400

@api.route('/calibrate/compute', methods=['POST'])
def calibrate_compute():
    try:
        data = request.get_json()
        src = data.get('src')
        dst = data.get('dst')
        name = data.get('name', 'Manual Calibration')
        if not src or not dst:
            return jsonify({'error': 'invalid points'}), 400
        
        from app.utils.geometry import compute_homography
        H, status = compute_homography(src, dst)
        if H is None:
             return jsonify({'error': 'failed'}), 500
        
        fusion_service.set_homography(H, name=name)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/calibrate/history')
def calibrate_history():
    from app.database.session import SessionLocal
    from app.database.models import Calibration
    db = SessionLocal()
    history = db.query(Calibration).order_by(Calibration.created_at.desc()).all()
    out = [c.to_dict() for c in history]
    db.close()
    return jsonify({'history': out})

@api.route('/calibrate/auto', methods=['POST'])
def calibrate_auto():
    try:
        from tools.auto_calibrate import auto_calibrate
        required = int(request.json.get('points', os.environ.get('CALIBRATION_POINTS', 5)))
        # We run this in a headless way if called from API to avoid window issues on server
        success = auto_calibrate(required_points=required, headless=True)
        if success:
            fusion_service.load_homography()
            return jsonify({'ok': True, 'message': 'Auto-calibration finished'})
        else:
            return jsonify({'error': 'Calibration failed or no markers found'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/calibrate/activate/<int:cal_id>', methods=['POST'])
def calibrate_activate(cal_id):
    from app.database.session import SessionLocal
    from app.database.models import Calibration
    try:
        db = SessionLocal()
        # Deactivate all
        db.query(Calibration).update({Calibration.is_active: False})
        
        # Activate specific
        cal = db.query(Calibration).filter(Calibration.id == cal_id).first()
        if not cal:
            db.close()
            return jsonify({'error': 'calibration not found'}), 404
        
        cal.is_active = True
        db.commit()
        db.close()
        
        # Reload in service
        fusion_service.load_homography()
        
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/fused')
def fused_list():
    out = []
    for ft in fusion_service.fused_tracks:
        # Frontend expects 'snaps' to be a list of objects {top:..., side:...}
        snaps_list = []
        paths = ft.get('snapshot_paths')
        if paths:
            snaps_list.append({'top': paths[0], 'side': paths[1], 'ts': ft['last_seen']})
            
        out.append({
            'id': ft['id'],
            'first_seen': ft['first_seen'],
            'last_seen': ft['last_seen'],
            'top_center': ft.get('top_center'),
            'side_center': ft.get('side_center'),
            'snaps': snaps_list
        })
    return jsonify({'tracks': out})

@api.route('/snapshots/<path:filename>')
def snapshot_file(filename):
    # Resolve relative path to absolute
    snapshot_dir = os.path.abspath(settings.SNAPSHOT_DIR)
    return send_from_directory(snapshot_dir, filename)

@api.route('/streams')
def streams():
    # Basic metrics placeholder
    return jsonify({
        'outputs': {'cam1': settings.OUTPUT_URL_1, 'cam2': settings.OUTPUT_URL_2}
    })
