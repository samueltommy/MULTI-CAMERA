from flask import Blueprint, jsonify, request, send_from_directory, render_template, current_app, Response
from app.services.fusion import fusion_service
from app.services.pipeline import pipeline_service
from app.services.calibration import calibration_service
from app.services.growth_predictor import growth_predictor
from app.core.config import settings
from app.database.session import SessionLocal
from app.database.models import FusedObject
from app.database.models import DailyStat, FarmSettings
from app.services.camera import camera_manager
from sqlalchemy import desc
import numpy as np
import os
import cv2
import time
from datetime import datetime, date, timedelta

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
    return jsonify({ 'iceServers': ice })

@api.route('/webrtc')
def webrtc_page():
    return render_template('webrtc.html')

@api.route('/gallery')
def gallery_page():
    return render_template('gallery.html')


# =========================================================================
# CALIBRATION ENDPOINTS (Diperbarui untuk React Calibration Modal)
# =========================================================================

@api.route('/api/calibration/stream/<int:cam_idx>')
def calibration_stream(cam_idx):
    """Stream MJPEG khusus untuk pop-up kalibrasi dengan overlay kotak ArUco"""
    def generate():
        while True:
            frame, _ = camera_manager.get_raw_frame(cam_idx)
            if frame is not None:
                # Copy frame agar tidak merusak frame utama pipeline
                display_frame = frame.copy()
                
                # Coba deteksi ArUco di frame ini
                corners, ids, _ = calibration_service.detector.detectMarkers(display_frame)
                
                # Jika terdeteksi, gambar kotak dan ID-nya secara real-time!
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(display_frame, corners, ids)
                    # Tambahkan teks panduan di layar
                    cv2.putText(display_frame, f"ARUCO DETECTED: {len(ids)}", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "SEARCHING FOR ARUCO...", (20, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Encode ke JPEG untuk dikirim ke browser
                ret, buffer = cv2.imencode('.jpg', display_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Batasi FPS sekitar 20fps agar Edge Node tidak keberatan beban
            time.sleep(0.05) 
            
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@api.route('/api/calibration/capture', methods=['POST'])
def calibration_capture():
    """Menangkap titik sudut ArUco dari kedua kamera."""
    success, msg = calibration_service.capture_points()
    return jsonify({'success': success, 'message': msg})

@api.route('/api/calibration/save', methods=['POST'])
def calibration_save():
    """Menghitung matriks 2.5D dan menyimpannya ke database."""
    data = request.get_json() or {}
    name = data.get('name', 'Web Auto-Calib')
    success, msg = calibration_service.compute_and_save(name=name)
    return jsonify({'success': success, 'message': msg})

@api.route('/api/calibration/reset', methods=['POST'])
def calibration_reset():
    """Mereset tangkapan titik (jika pengguna ingin membatalkan)."""
    calibration_service.reset()
    return jsonify({'success': True, 'message': 'Tangkapan kamera di-reset.'})

@api.route('/calibrate/history')
def calibrate_history():
    """Melihat riwayat kalibrasi di database."""
    from app.database.session import SessionLocal
    from app.database.models import Calibration
    db = SessionLocal()
    history = db.query(Calibration).order_by(Calibration.created_at.desc()).all()
    out = [c.to_dict() for c in history]
    db.close()
    return jsonify({'history': out})

@api.route('/calibrate/activate/<int:cal_id>', methods=['POST'])
def calibrate_activate(cal_id):
    """Mengaktifkan kembali kalibrasi lama dari riwayat."""
    from app.database.session import SessionLocal
    from app.database.models import Calibration
    try:
        db = SessionLocal()
        # Nonaktifkan semua
        db.query(Calibration).update({Calibration.is_active: False})
        
        # Aktifkan yang dipilih
        cal = db.query(Calibration).filter(Calibration.id == cal_id).first()
        if not cal:
            db.close()
            return jsonify({'error': 'calibration not found'}), 404
        
        cal.is_active = True
        db.commit()
        db.close()
        
        # Minta Fusion Service dan Weight Predictor memuat ulang matriks
        fusion_service.load_homography()
        from app.services.weight_predictor import weight_predictor
        weight_predictor.load_calibration()
        
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@api.route('/calibrate/identity', methods=['POST'])
def calibrate_identity():
    """Fallback Darurat: Memaksa matriks Identity (Anggap Kamera 1 & 2 persis sama)"""
    try:
        H = np.eye(3, dtype=np.float32)
        fusion_service.set_homography(H, name="Identity (same video)")
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =========================================================================


@api.route('/fused')
def fused_list():
    out = []
    for ft in fusion_service.fused_tracks:
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
    snapshot_dir = os.path.abspath(settings.SNAPSHOT_DIR)
    return send_from_directory(snapshot_dir, filename)

@api.route('/streams')
def streams():
    return jsonify({
        'outputs': {'cam1': settings.OUTPUT_URL_1, 'cam2': settings.OUTPUT_URL_2}
    })

@api.route('/object_details', methods=['GET'])
def get_object_details():
    track_id = request.args.get('track_id')
    session_id = request.args.get('session_id')
    
    if not track_id:
        return jsonify({"error": "track_id is required"}), 400
        
    db = SessionLocal()
    try:
        query = db.query(FusedObject).filter(FusedObject.track_id == track_id)
        if session_id:
            query = query.filter(FusedObject.session_id == session_id)
            
        obj = query.order_by(desc(FusedObject.created_at)).first()
        
        if not obj:
            return jsonify({"error": "Object not found"}), 404
            
        data = obj.to_dict()
        
        host_url = request.host_url.rstrip('/') 
        
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

@api.route('/sessions', methods=['GET'])
def get_sessions():
    db = SessionLocal()
    try:
        sessions = db.query(FusedObject.session_id)\
                     .filter(FusedObject.session_id.isnot(None))\
                     .distinct().all()
        
        session_list = [s[0] for s in sessions]
        session_list.sort(reverse=True)
        
        return jsonify({
            "sessions": session_list,
            "current_active": pipeline_service.current_session_id
        })
    finally:
        db.close()

@api.route('/session_objects', methods=['GET'])
def get_session_objects():
    session_id = request.args.get('session_id')
    
    if not session_id:
        session_id = pipeline_service.current_session_id
        
    if not session_id:
        return jsonify([]) 

    db = SessionLocal()
    try:
        rows = db.query(FusedObject)\
                 .filter(FusedObject.session_id == session_id)\
                 .order_by(desc(FusedObject.created_at))\
                 .all()
        
        unique_map = {}
        for row in rows:
            if row.track_id not in unique_map:
                unique_map[row.track_id] = {
                    "track_id": row.track_id,
                    "estimated_weight": row.estimated_weight,
                    "last_seen": row.created_at.isoformat() if row.created_at else None
                }
        
        result = list(unique_map.values())
        result.sort(key=lambda x: x['track_id'])
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@api.route('/farm_settings', methods=['GET', 'POST'])
def handle_farm_settings():
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
                settings.chick_in_date = datetime.strptime(data['chick_in_date'], '%Y-%m-%d').date() if data['chick_in_date'] else None
            if 'manual_age_override' in data:
                settings.manual_age_override = data['manual_age_override'] if data['manual_age_override'] != "" else None
            if 'target_harvest_weight_kg' in data:
                settings.target_harvest_weight_kg = float(data['target_harvest_weight_kg'])
                
            db.commit()
        return jsonify(settings.to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@api.route('/daily_statistics', methods=['GET'])
def get_daily_statistics():
    db = SessionLocal()
    try:
        date_str = request.args.get('date')
        if date_str:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            target_date = date.today()

        stat = db.query(DailyStat).filter(DailyStat.date == target_date).first()

        if not stat:
            return jsonify({
                "date": target_date.isoformat(),
                "total_sessions": 0,
                "total_chickens": 0,
                "daily_average_kg": 0.0,
                "message": "Belum ada data penimbangan untuk tanggal ini."
            })

        return jsonify(stat.to_dict())

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@api.route('/harvest_prediction', methods=['GET'])
def get_harvest_prediction():
    db = SessionLocal()
    try:
        settings = db.query(FarmSettings).first()
        target_weight = settings.target_harvest_weight_kg if settings else 2.0
        
        today = date.today()
        daily_stat = db.query(DailyStat).filter(DailyStat.date == today).first()
        
        if not daily_stat or daily_stat.daily_average_kg <= 0:
            return jsonify({
                "status": "waiting_for_data",
                "message": "Silakan jalankan deteksi kamera terlebih dahulu hari ini untuk mendapatkan prediksi panen."
            })
            
        current_weight = daily_stat.daily_average_kg
        prediction = growth_predictor.predict_harvest(current_weight, target_weight)
        
        if settings and settings.chick_in_date:
            calendar_age = (today - settings.chick_in_date).days
            prediction["calendar_age_days"] = max(0, calendar_age)
        
        return jsonify({
            "status": "success",
            "prediction": prediction
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()

@api.route('/growth_chart', methods=['GET'])
def get_growth_chart():
    db = SessionLocal()
    try:
        settings = db.query(FarmSettings).first()
        
        if not settings or not settings.chick_in_date:
            return jsonify({
                "status": "error", 
                "error": "Silakan atur 'Chick-in Date' pada menu Settings (⚙️) di pojok kanan atas agar grafik dapat ditampilkan."
            }), 200

        chick_in = settings.chick_in_date
        daily_stats = db.query(DailyStat).all()
        
        actual_data_map = {}
        for stat in daily_stats:
            age_days = (stat.date - chick_in).days
            if age_days >= 0:
                actual_data_map[age_days] = stat.daily_average_kg
        
        ciomas_standard = {
            0: 0.042, 1: 0.056, 2: 0.073, 3: 0.094, 4: 0.118, 5: 0.145, 6: 0.176,
            7: 0.210, 8: 0.247, 9: 0.288, 10: 0.332, 11: 0.379, 12: 0.429, 13: 0.483,
            14: 0.540, 15: 0.600, 16: 0.663, 17: 0.729, 18: 0.798, 19: 0.870, 20: 0.945,
            21: 1.024, 22: 1.105, 23: 1.189, 24: 1.276, 25: 1.365, 26: 1.457, 27: 1.552,
            28: 1.649, 29: 1.747, 30: 1.846, 31: 1.945, 32: 2.045, 33: 2.146, 34: 2.247,
            35: 2.348
        }

        chart_data = []
        
        for age_days in range(41):
            target_bw = ciomas_standard.get(age_days)
            if target_bw is None:
                 target_bw = 2.348 + ((age_days - 35) * 0.1)

            current_date = chick_in + timedelta(days=age_days)

            chart_data.append({
                "date": current_date.strftime("%d %b"),
                "age_days": age_days,
                "actual_weight_kg": round(actual_data_map[age_days], 3) if age_days in actual_data_map else None,
                "target_weight_kg": round(target_bw, 3)
            })

        return jsonify({
            "status": "success", 
            "data": chart_data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()