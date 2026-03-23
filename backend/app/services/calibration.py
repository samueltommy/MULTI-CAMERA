import cv2
import numpy as np
import time
import json
from app.services.camera import camera_manager
from app.database.session import SessionLocal
from app.database.models import Calibration
from app.core.config import settings

class CalibrationService:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # Wadah penyimpanan sementara saat tombol Capture ditekan
        self.src_pts = [] # Titik piksel di Kamera Atas
        self.dst_pts = [] # Titik piksel di Kamera Samping
        self.captured_ids = [] # ID ArUco yang berhasil ditangkap
        
        # =========================================================================
        # PETA DUNIA NYATA (MASTER MAP) - FLEKSIBEL UNTUK 4 HINGGA 12+ ARUCO
        # Format: { ID_Aruco : [Posisi_X_cm, Posisi_Y_cm] }
        # =========================================================================
        # Anda mengukur ini dari titik tengah ArUco ke titik tengah ArUco lainnya.
        # Jika Anda hanya memakai 4 (ID 1,2,3,4), sistem hanya akan membaca 4.
        self.real_world_map = {
            1: [0, 0],         2: [100, 0],       # Kiri Atas, Kanan Atas
            3: [100, 100],     4: [0, 100],       # Kanan Bawah, Kiri Bawah
            
            # Tambahan jika Anda iseng menaruh lebih banyak ArUco di tengah/pinggir:
            5: [100, 0],       6: [200, 100],
            7: [100, 200],     8: [0, 100],
            9: [100, 100]                         # Titik persis di tengah kandang
        }

    def reset(self):
        self.src_pts = []
        self.dst_pts = []
        self.captured_ids = []

    def get_marker_centers(self, corners, ids):
        """Menghitung titik tengah (Center) dari setiap ArUco yang terdeteksi"""
        centers = {}
        if ids is None: return centers
        ids = ids.flatten()
        for i, marker_id in enumerate(ids):
            c = corners[i][0]
            cx = int(np.mean(c[:, 0]))
            cy = int(np.mean(c[:, 1]))
            centers[marker_id] = [cx, cy]
        return centers

    def capture_points(self):
        """Fungsi ini dipanggil saat tombol 'Capture' ditekan di Web Dashboard"""
        frame1, _ = camera_manager.get_raw_frame(0) # Kamera Atas
        frame2, _ = camera_manager.get_raw_frame(1) # Kamera Samping
        
        if frame1 is None or frame2 is None:
            return False, "Kamera tidak memberikan frame yang valid."

        # Deteksi ArUco di kedua kamera
        corners1, ids1, _ = self.detector.detectMarkers(frame1)
        corners2, ids2, _ = self.detector.detectMarkers(frame2)

        # Ambil titik tengahnya
        centers1 = self.get_marker_centers(corners1, ids1)
        centers2 = self.get_marker_centers(corners2, ids2)

        # FILTER CERDAS: Cari ID yang terlihat di Kamera Atas, Kamera Samping, 
        # DAN ada di dalam 'Peta Dunia Nyata' yang kita buat di atas.
        common_ids = set(centers1.keys()).intersection(set(centers2.keys())).intersection(set(self.real_world_map.keys()))

        if not common_ids:
            return False, "Tidak ada ArUco yang cocok di kedua kamera dan Peta."

        # Bersihkan tangkapan sebelumnya, lalu simpan yang baru
        self.reset()
        for cid in common_ids:
            self.src_pts.append(centers1[cid])
            self.dst_pts.append(centers2[cid])
            self.captured_ids.append(cid)
            
        return True, f"Berhasil menangkap {len(self.captured_ids)} titik ArUco (ID: {self.captured_ids})."

    def compute_and_save(self, name=None):
        """Fungsi ini dipanggil saat tombol 'Save Calibration' ditekan di Web Dashboard"""
        if len(self.src_pts) < 4:
            return False, f"Hanya tertangkap {len(self.src_pts)} titik. Butuh minimal 4 ArUco."

        # =========================================================
        # 1. MATRIKS FUSI (Kamera Atas -> Samping)
        # =========================================================
        src_fusion = np.array(self.src_pts, dtype=np.float32)
        dst_fusion = np.array(self.dst_pts, dtype=np.float32)
        
        # Menggunakan RANSAC agar stabil berapapun jumlah titiknya (4, 8, atau 12)
        H_fusion, _ = cv2.findHomography(src_fusion, dst_fusion, cv2.RANSAC, 5.0)

        # =========================================================
        # 2. MATRIKS SKALA DUNIA NYATA (Kamera Atas -> CM)
        # =========================================================
        # Menarik data [X_cm, Y_cm] dari Peta Master sesuai dengan ID yang ditangkap
        dst_cm_list = [self.real_world_map[cid] for cid in self.captured_ids]
        dst_cm = np.array(dst_cm_list, dtype=np.float32)
        
        H_scale_top, _ = cv2.findHomography(src_fusion, dst_cm, cv2.RANSAC, 5.0)

        # =========================================================
        # 3. SIMPAN KE DATABASE JSON
        # =========================================================
        if H_fusion is not None and H_scale_top is not None:
            if not name:
                name = f"Flexible Auto-Calib {time.strftime('%Y-%m-%d %H:%M')}"
                
            db = SessionLocal()
            try:
                db.query(Calibration).update({Calibration.is_active: False})
                matrix_data = {
                    "H_fusion": H_fusion.tolist(),
                    "H_scale_top": H_scale_top.tolist()
                }
                new_cal = Calibration(name=name, matrix_json=json.dumps(matrix_data), is_active=True)
                db.add(new_cal)
                db.commit()
                return True, f"Kalibrasi ({len(self.src_pts)} Titik) Sukses Disimpan!"
            except Exception as e:
                db.rollback()
                return False, f"Database Error: {e}"
            finally:
                db.close()
                
        return False, "Gagal menghitung matriks. Pastikan posisi ArUco tidak segaris lurus semua."

calibration_service = CalibrationService()