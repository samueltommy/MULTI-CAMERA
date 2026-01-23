from flask import Blueprint, jsonify, request, send_from_directory, render_template, current_app
from app.services.fusion import fusion_service
from app.core.config import settings
import numpy as np
import os

api = Blueprint('api', __name__)

@api.route('/')
def index():
    return render_template('index.html')

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

@api.route('/calibrate/compute', methods=['POST'])
def calibrate_compute():
    try:
        data = request.get_json()
        src = data.get('src')
        dst = data.get('dst')
        if not src or not dst:
            return jsonify({'error': 'invalid points'}), 400
        
        from app.utils.geometry import compute_homography
        H, status = compute_homography(src, dst)
        if H is None:
             return jsonify({'error': 'failed'}), 500
        
        fusion_service.set_homography(H)
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
