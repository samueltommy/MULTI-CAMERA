import numpy as np
import cv2
import json
from app.database.session import SessionLocal
from app.database.models import Calibration

class WeightPredictor:
    def __init__(self, standard_width=1280, standard_height=720, camera_height_cm=200.0):
        self.standard_width = standard_width
        self.standard_height = standard_height
        
        # Jarak vertikal lensa kamera atas ke lantai kandang (Penting untuk Optik 2.5D)
        self.camera_height_cm = camera_height_cm
        
        self.weight_factors = {
            'chicken': 1.0,
            'chicken drumstick': 0.64,
            'chicken neck': 0.68,
            'chicken wing': 0.51
        }

        # ==========================================================
        # TABEL STANDAR CIOMAS & DYNAMIC OUTLIER LIMITS
        # ==========================================================
        self.ciomas_standard = {
            0: 0.042, 1: 0.056, 2: 0.073, 3: 0.094, 4: 0.118, 5: 0.145, 6: 0.176,
            7: 0.210, 8: 0.247, 9: 0.288, 10: 0.332, 11: 0.379, 12: 0.429, 13: 0.483,
            14: 0.540, 15: 0.600, 16: 0.663, 17: 0.729, 18: 0.798, 19: 0.870, 20: 0.945,
            21: 1.024, 22: 1.105, 23: 1.189, 24: 1.276, 25: 1.365, 26: 1.457, 27: 1.552,
            28: 1.649, 29: 1.747, 30: 1.846, 31: 1.945, 32: 2.045, 33: 2.146, 34: 2.247,
            35: 2.348
        }
        
        self.min_valid_kg = 0.050
        self.max_valid_kg = 3.000
        
        # Matriks Kalibrasi Kamera
        self.H_scale_top = None
        self.load_calibration()

    def load_calibration(self):
        """Menarik Matriks Skala Geometri Nyata dari Database"""
        try:
            db = SessionLocal()
            cal = db.query(Calibration).filter(Calibration.is_active == True).order_by(Calibration.created_at.desc()).first()
            if cal:
                data = json.loads(cal.matrix_json)
                if isinstance(data, dict) and data.get('H_scale_top'):
                    self.H_scale_top = np.array(data['H_scale_top'], dtype=np.float32)
                    print("[WeightPredictor] Matriks H_scale_top (Geometri Nyata) berhasil dimuat!")
            db.close()
        except Exception as e:
            print(f"[WeightPredictor] Gagal memuat kalibrasi geometri: {e}")

    def update_age_limits(self, age_days: int):
        """Filter Outlier Cerdas Berdasarkan Umur Ciomas"""
        target_bw = self.ciomas_standard.get(age_days)
        if target_bw is None:
            if age_days > 35: target_bw = 2.348 + ((age_days - 35) * 0.1)
            else: target_bw = 0.05

        self.min_valid_kg = max(0.030, target_bw * 0.3)
        self.max_valid_kg = max(0.800, target_bw * 1.8)
        
        print(f"[Weight Predictor] Age: {age_days} Days | Target: {target_bw:.3f} kg")
        print(f"[Weight Predictor] OUTLIER Filter set to: {self.min_valid_kg:.3f} kg - {self.max_valid_kg:.3f} kg")

    def _get_real_world_area(self, top_det):
        """Menghitung Luas Area asli (cm2) yang BEBAS dari Distorsi Perspektif Lensa"""
        x1, y1, x2, y2 = top_det['box']
        
        # Hitung luas kotak bounding box di piksel
        bbox_area_px = max(1.0, float((x2 - x1) * (y2 - y1)))
        # Ambil luas tubuh ayam asli (mask) dari YOLO, jika tidak ada, pakai luas kotak
        mask_area_px = top_det.get('mask_area', bbox_area_px)
        
        # Cari rasio (berapa persen kotak tersebut terisi oleh tubuh ayam?)
        ratio = min(1.0, mask_area_px / bbox_area_px)
        
        if self.H_scale_top is None:
            return (bbox_area_px * (0.05 ** 2)) * ratio

        pts_pixel = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Proyeksi Piksel ke Dunia Nyata (cm2) untuk kotak
        pts_cm = cv2.perspectiveTransform(pts_pixel, self.H_scale_top)
        bbox_area_cm2 = cv2.contourArea(pts_cm)
        
        # LUAS ASLI AYAM = Luas kotak dunia nyata dikali persentase kepadatan tubuh
        true_mask_area_cm2 = bbox_area_cm2 * ratio
        return true_mask_area_cm2

    def _compensate_z_axis(self, area_cm2, chicken_height_cm):
        """Kompensasi Optik untuk Metode 2.5D"""
        H = self.camera_height_cm
        if chicken_height_cm >= H or chicken_height_cm <= 0: return area_cm2
        return area_cm2 * (((H - chicken_height_cm) / H) ** 2)

    def predict_from_area(self, top_det, class_name, frame_shape):
        """Algoritma 2D Area (Baseline)"""
        try:
            # Area dihitung dengan Homografi (Piksel miring jadi Lurus)
            base_area_cm2 = self._get_real_world_area(top_det)
            if base_area_cm2 < 10.0: return 0.0
            
            # Perhitungan 2D Murni
            DENSITY_2D = 3.5
            base_weight_kg = ((base_area_cm2 ** 1.5) * DENSITY_2D) / 1000.0
            
            factor = self.weight_factors.get(class_name, 1.0)
            final_weight = base_weight_kg * factor
            
            if final_weight < self.min_valid_kg or final_weight > self.max_valid_kg:
                return 0.0 
                
            return final_weight
        except Exception as e:
            return 0.0

    def predict_from_volume(self, top_det, side_det, class_name, frame_shape):
        """Algoritma 3D Volume (Fusi Top + Side)"""
        try:
            base_area_cm2 = self._get_real_world_area(top_det)
            
            y1_s, y2_s = side_det['box'][1], side_det['box'][3]
            chicken_height_px = float(y2_s - y1_s)
            chicken_height_cm = chicken_height_px * 0.1 # Nanti bisa disesuaikan
            
            # Kompensasi Z-Axis (Metode 2.5D Canggih)
            true_surface_area_cm2 = self._compensate_z_axis(base_area_cm2, chicken_height_cm)
            
            volume_cm3 = true_surface_area_cm2 * chicken_height_cm
            if volume_cm3 < 10.0: return 0.0
            
            DENSITY_G_PER_CM3 = 2.7 
            base_weight_kg = (volume_cm3 * DENSITY_G_PER_CM3) / 1000.0
            
            factor = self.weight_factors.get(class_name, 1.0)
            final_weight = base_weight_kg * factor
            
            # Cek Outlier Ciomas
            if final_weight < self.min_valid_kg or final_weight > self.max_valid_kg:
                return 0.0 
            
            return final_weight
        except Exception as e:
            return 0.0

weight_predictor = WeightPredictor()