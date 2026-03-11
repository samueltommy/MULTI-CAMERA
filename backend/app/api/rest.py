from flask import Blueprint, jsonify, request, send_from_directory, render_template, current_app
from app.services.fusion import fusion_service
from app.services.pipeline import pipeline_service
from app.services.calibration import calibration_service
from app.core.config import settings
from app.database.session import SessionLocal
from app.database.models import FusedObject
from sqlalchemy import desc
import numpy as np
import os
import time
from datetime import datetime

api = Blueprint('api', __name__)

@api.route('/')
def index():
    return render_template('index.html')

@api.route('/trigger', methods=['POST'])
def trigger_session():
    import traceback
    duration = int(request.json.get('duration', 60))
    # Enable YOLO inference for this capture session
    try:
        if not getattr(pipeline_service, 'inference_enabled', False):
            try:
                pipeline_service.enable_inference(True)
                pipeline_service.mark_started_on_demand(True)
                print("[rest.trigger] inference ENABLED for capture session")
            except Exception as e:
                print(f"[rest.trigger] error enabling inference: {e}")
                traceback.print_exc()
    except Exception as e:
        print(f"[rest.trigger] outer exception: {e}")
        traceback.print_exc()

    try:
        pipeline_service.start_session(duration)
    except Exception as e:
        print(f"[rest.trigger] error during session start: {e}")
        traceback.print_exc()
    
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


@api.route('/calibrate/identity', methods=['POST'])
def calibrate_identity():
    """Explicitly set an identity homography.

    This is useful when the two camera feeds are actually the same video
    (e.g. you are testing with a duplicated source).  With an identity
    matrix every point projects to itself and the fusion logic treats the
    two streams as perfectly overlapping.  The endpoint simply writes the
    3x3 identity into the database via :class:`FusionService`.
    """
    try:
        H = np.eye(3, dtype=np.float32)
        fusion_service.set_homography(H, name="Identity (same video)")
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

@api.route('/object_details', methods=['GET'])
def get_object_details():
    """
    Mengambil detail lengkap (Foto Top/Side, Berat) untuk satu ID ayam.
    Dipanggil saat user mengklik baris di tabel kiri.
    """
    track_id = request.args.get('track_id')
    session_id = request.args.get('session_id')
    
    if not track_id:
        return jsonify({"error": "track_id is required"}), 400
        
    db = SessionLocal()
    try:
        query = db.query(FusedObject).filter(FusedObject.track_id == track_id)
        if session_id:
            query = query.filter(FusedObject.session_id == session_id)
            
        # Ambil data paling update (terakhir direkam)
        obj = query.order_by(desc(FusedObject.created_at)).first()
        
        if not obj:
            return jsonify({"error": "Object not found"}), 404
            
        data = obj.to_dict()
        
        # Generate URL Gambar untuk Frontend
        host_url = request.host_url.rstrip('/') # http://localhost:5000
        
        if data.get('snapshot_top'):
            data['image_url_top'] = f"{host_url}/snapshots/{data['snapshot_top']}"
        else:
            data['image_url_top'] = None
            
        if data.get('snapshot_side'):
            data['image_url_side'] = f"{host_url}/snapshots/{data['snapshot_side']}"
        else:
            data['image_url_side'] = None
            
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

# --- BARU: List Semua Sesi ---
@api.route('/sessions', methods=['GET'])
def get_sessions():
    """Mengambil daftar semua Session ID yang ada di database."""
    db = SessionLocal()
    try:
        # Ambil session_id yang unik
        sessions = db.query(FusedObject.session_id)\
                     .filter(FusedObject.session_id.isnot(None))\
                     .distinct().all()
        
        # Convert list of tuples ke list of strings
        session_list = [s[0] for s in sessions]
        
        # Urutkan dari yang terbaru (Descending)
        session_list.sort(reverse=True)
        
        return jsonify({
            "sessions": session_list,
            "current_active": pipeline_service.current_session_id
        })
    finally:
        db.close()

# --- BARU: List Objek untuk Tabel Kiri ---
@api.route('/session_objects', methods=['GET'])
def get_session_objects():
    """
    Mengambil daftar ID ayam dan berat terakhirnya dalam satu sesi.
    Output urut dari ID terkecil ke terbesar.
    """
    session_id = request.args.get('session_id')
    
    # Jika parameter kosong, gunakan sesi yang sedang berjalan (jika ada)
    if not session_id:
        session_id = pipeline_service.current_session_id
        
    if not session_id:
        return jsonify([]) # Tidak ada sesi aktif

    db = SessionLocal()
    try:
        # Ambil semua data pada sesi ini, urutkan dari yang terbaru
        rows = db.query(FusedObject)\
                 .filter(FusedObject.session_id == session_id)\
                 .order_by(desc(FusedObject.created_at))\
                 .all()
        
        # Filter: Hanya ambil data TERBARU untuk setiap track_id
        unique_map = {}
        for row in rows:
            if row.track_id not in unique_map:
                unique_map[row.track_id] = {
                    "track_id": row.track_id,
                    "estimated_weight": row.estimated_weight,
                    "last_seen": row.created_at.isoformat() if row.created_at else None
                }
        
        # Ubah ke list
        result = list(unique_map.values())
        
        # SORTING: Urutkan berdasarkan ID Terkecil (Ascending)
        # Ini memudahkan frontend untuk auto-select index[0]
        result.sort(key=lambda x: x['track_id'])
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@api.route('/farm_settings', methods=['GET', 'POST'])
def handle_farm_settings():
    from app.database.models import FarmSettings
    db = SessionLocal()
    try:
        settings = db.query(FarmSettings).first()
        if not settings:
            settings = FarmSettings()
            db.add(settings)
            db.commit()

        if request.method == 'POST':
            data = request.json
            if 'chick_in_date' in data:
                if data['chick_in_date']:
                    settings.chick_in_date = datetime.strptime(data['chick_in_date'], '%Y-%m-%d').date()
                else:
                    settings.chick_in_date = None
                    
            if 'manual_age_override' in data:
                settings.manual_age_override = data['manual_age_override'] if data['manual_age_override'] != "" else None
                
            db.commit()
            
        return jsonify(settings.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()