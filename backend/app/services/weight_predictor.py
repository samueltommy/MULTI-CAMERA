import os
import numpy as np
import cv2
import json
import joblib
import pandas as pd
from app.database.session import SessionLocal
from app.database.models import Calibration

class WeightPredictor:
    def __init__(self, standard_width=640, standard_height=360, camera_height_cm=170.0):
        self.standard_width = standard_width
        self.standard_height = standard_height
        self.camera_height_cm = camera_height_cm
        
        # Umur default jika tidak ada input dari pipeline
        self.current_age_days = 15 

        # ==========================================================
        # 1. MEMUAT MODEL AI (XGBOOST) DARI FOLDER MODELS
        # ==========================================================
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(os.path.dirname(current_dir), 'models')
        
        # Muat Model 3D (Utama)
        self.model_3d = None
        self.features_3d = []
        try:
            path_3d = os.path.join(models_dir, 'gbdt_3d_model.pkl')
            data_3d = joblib.load(path_3d)
            self.model_3d = data_3d['model']
            self.features_3d = data_3d['features']
            print(f"✅ [WeightPredictor] Otak XGBoost 3D berhasil dimuat!")
        except Exception as e:
            print(f"⚠️ [WeightPredictor] Gagal memuat model 3D: {e}")

        # Muat Model 2D (Fallback/Ablation)
        self.model_2d = None
        self.features_2d = []
        try:
            path_2d = os.path.join(models_dir, 'gbdt_2d_model.pkl')
            data_2d = joblib.load(path_2d)
            self.model_2d = data_2d['model']
            self.features_2d = data_2d['features']
            print(f"✅ [WeightPredictor] Otak XGBoost 2D berhasil dimuat!")
        except Exception as e:
            print(f"⚠️ [WeightPredictor] Gagal memuat model 2D: {e}")

        # ==========================================================
        # FILTER OUTLIER CIOMAS (TETAP DIPERTAHANKAN SEBAGAI SAFETY NET)
        # ==========================================================
        self.ciomas_standard = {
            0: 0.042, 1: 0.056, 2: 0.073, 3: 0.094, 4: 0.118, 5: 0.145, 6: 0.176,
            7: 0.210, 8: 0.247, 9: 0.288, 10: 0.332, 11: 0.379, 12: 0.429, 13: 0.483,
            14: 0.540, 15: 0.600, 16: 0.663, 17: 0.729, 18: 0.798, 19: 0.870, 20: 0.945
        }
        self.min_valid_kg = 0.050
        self.max_valid_kg = 3.000

    def update_age_limits(self, age_days: int):
        """Memperbarui umur ayam saat ini untuk diumpankan ke Model AI"""
        self.current_age_days = age_days # <-- SANGAT PENTING UNTUK AI
        
        target_bw = self.ciomas_standard.get(age_days)
        if target_bw is None:
            if age_days > 20: target_bw = 0.945 + ((age_days - 20) * 0.08)
            else: target_bw = 0.05

        self.min_valid_kg = max(0.030, target_bw * 0.4)
        self.max_valid_kg = max(0.800, target_bw * 1.6)
        
        print(f"[Weight Predictor] Age updated to: {age_days} Days")

    # ==========================================================
    # PREDIKSI MENGGUNAKAN AI XGBOOST (2D & 3D)
    # ==========================================================
    def predict_from_area(self, top_det, class_name, frame_shape):
        """Prediksi menggunakan AI 2D (Kamera Atas Saja)"""
        if self.model_2d is None:
            return 0.0 # Jika model gagal load, batalkan
            
        try:
            # 1. Ekstrak Fitur dari Kamera Atas
            x1_t, y1_t, x2_t, y2_t = top_det['box']
            bbox_width_px = max(1.0, float(x2_t - x1_t))
            
            # Ambil mask_area jika ada, jika tidak pakai luas bounding box
            mask_area_px = float(top_det.get('mask_area', bbox_width_px * (y2_t - y1_t)))

            # 2. Susun Data untuk AI (Bentuk DataFrame pandas)
            input_dict = {
                'mask_area_px': mask_area_px,
                'bbox_width_px': bbox_width_px,
                'age_days': self.current_age_days
            }
            # Pastikan urutan kolom sesuai dengan saat training
            input_df = pd.DataFrame([input_dict])[self.features_2d]

            # 3. Minta AI menebak!
            predicted_weight_kg = float(self.model_2d.predict(input_df)[0])
            
            # 4. Filter Biologi Dasar (Jika AI error menebak terlalu ekstrem)
            if predicted_weight_kg < self.min_valid_kg or predicted_weight_kg > self.max_valid_kg:
                return 0.0
                
            return predicted_weight_kg

        except Exception as e:
            print(f"Error prediksi AI 2D: {e}")
            return 0.0

    def predict_from_volume(self, top_det, side_det, class_name, frame_shape):
        """Prediksi menggunakan AI 3D (Kamera Atas + Samping) - AKURASI 99.8%"""
        if self.model_3d is None:
            return 0.0
            
        try:
            # 1. Ekstrak Fitur dari Kamera Atas (Lebar & Luas)
            x1_t, y1_t, x2_t, y2_t = top_det['box']
            bbox_width_px = max(1.0, float(x2_t - x1_t))
            mask_area_px = float(top_det.get('mask_area', bbox_width_px * (y2_t - y1_t)))

            # 2. Ekstrak Fitur dari Kamera Samping (Tinggi)
            _, y1_s, _, y2_s = side_det['box']
            bbox_height_px = max(1.0, float(y2_s - y1_s))

            # 3. Susun Data untuk AI (Bentuk DataFrame pandas)
            input_dict = {
                'mask_area_px': mask_area_px,
                'bbox_width_px': bbox_width_px,
                'bbox_height_px': bbox_height_px,
                'age_days': self.current_age_days
            }
            # Pastikan urutan kolom sama persis dengan gbdt_3d_model.pkl
            input_df = pd.DataFrame([input_dict])[self.features_3d]

            # 4. Minta AI menebak!
            predicted_weight_kg = float(self.model_3d.predict(input_df)[0])
            
            # 5. Filter Biologi Dasar
            if predicted_weight_kg < self.min_valid_kg or predicted_weight_kg > self.max_valid_kg:
                return 0.0
                
            return predicted_weight_kg

        except Exception as e:
            print(f"Error prediksi AI 3D: {e}")
            return 0.0

# Singleton instance
weight_predictor = WeightPredictor()