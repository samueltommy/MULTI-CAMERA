import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

class GrowthPredictor:
    def __init__(self):
        # =====================================================================
        # DATASET KURVA PERTUMBUHAN (STANDAR CIOMAS)
        # Berdasarkan referensi tabel Standar Ciomas (Kolom BW):
        # Umur 0  = 42g
        # Umur 7  = 210g
        # Umur 11 = 379g
        # Umur 14 = 540g
        # Umur 21 = 1024g (1.024 kg)
        # Umur 28 = 1649g (1.649 kg)
        # Umur 34 = 2247g (2.247 kg)
        # =====================================================================
        
        # Input Fitur (X): Berat dalam KG sesuai Standar Ciomas
        self.weights = np.array([
            0.042, 
            0.210, 
            0.379, 
            0.540, 
            1.024, 
            1.649, 
            2.247
        ]).reshape(-1, 1)
        
        # Target (Y): Umur dalam Hari
        self.ages = np.array([0, 7, 11, 14, 21, 28, 34])
        
        # Membuat Model ML Non-Linear (Polynomial Degree 3)
        # Polinomial derajat 3 sangat sempurna untuk menangkap Kurva-S pertumbuhan hewan
        self.model = make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0))
        
        # Melatih model menggunakan Standar Ciomas saat server menyala
        self.model.fit(self.weights, self.ages)

    def predict_harvest(self, current_weight_kg: float, target_weight_kg: float):
        """Memprediksi sisa hari menuju panen berdasarkan berat aktual vs target Ciomas"""
        if current_weight_kg <= 0 or target_weight_kg <= 0:
            return None
            
        # 1. AI Memprediksi "Umur Biologis" dari berat saat ini
        # Jika ayam umur 11 hari tapi beratnya baru 300g (di bawah standar Ciomas 379g),
        # ML akan menganggap umur biologis ayam masih sekitar 9-10 hari.
        current_bio_age = self.model.predict([[current_weight_kg]])[0]
        
        # 2. AI Memprediksi "Umur Target" untuk mencapai berat panen
        target_bio_age = self.model.predict([[target_weight_kg]])[0]
        
        # 3. Hitung selisih hari
        days_remaining = target_bio_age - current_bio_age
        
        # Jika ayam sudah melebihi target, sisa hari = 0 (Siap Panen)
        days_remaining = max(0.0, days_remaining)
        
        return {
            "current_weight_kg": round(float(current_weight_kg), 3),
            "target_weight_kg": round(float(target_weight_kg), 3),
            "current_biological_age_days": round(float(current_bio_age), 1),
            "target_harvest_age_days": round(float(target_bio_age), 1),
            "estimated_days_remaining": round(float(days_remaining), 1),
            "is_ready_to_harvest": bool(days_remaining <= 0.5)
        }

growth_predictor = GrowthPredictor()