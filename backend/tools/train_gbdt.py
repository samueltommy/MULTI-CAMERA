import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Setup path agar skrip bisa membaca/menyimpan ke folder 'app/models'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# =========================================================
# FUNGSI TRAINING & EVALUASI
# =========================================================
def train_and_save_model(X, y, model_filename, step_label, scenario_name):
    print(f"\n{step_label} Melatih model XGBoost Regressor - Skenario {scenario_name}...")
    
    # Split Data (80% Train, 20% Test) dengan seed yang sama agar 2D dan 3D adil (Apples-to-Apples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Parameter XGBoost yang sudah dioptimasi
    model = XGBRegressor(
        n_estimators=200,      
        learning_rate=0.05,    
        max_depth=5,           
        subsample=0.8,         
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1 # Gunakan seluruh core CPU agar sangat cepat
    )
    
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("=" * 60)
    print(f"📊 HASIL EVALUASI '{scenario_name}' (PADA 20% DATA UJI):")
    print(f"  - MAE  (Rata-rata error absolut)  : {mae*1000:.1f} Gram")
    print(f"  - RMSE (Standar deviasi akurasi)  : {rmse*1000:.1f} Gram")
    print(f"  - R²   (Kecocokan Model Prediksi) : {r2*100:.2f} %")
    print("=" * 60)

    # --- EKSTRAK FEATURE IMPORTANCE UNTUK BAB 4 TESIS ---
    if scenario_name == "3D":
        importance = model.feature_importances_
        plt.figure(figsize=(8, 5))
        plt.barh(X.columns, importance, color='skyblue')
        plt.xlabel("Tingkat Kepentingan (0 - 1.0)")
        plt.title("Feature Importance - XGBoost 3D Model")
        
        plot_path = os.path.join(current_dir, "feature_importance_3d.png")
        plt.savefig(plot_path, bbox_inches='tight')
        print(f"📈 Grafik Feature Importance disimpan di: {plot_path}")

    # --- SIMPAN MODEL KE FOLDER BACKEND ---
    model_dir = os.path.join(parent_dir, 'app', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    artifact = {
        'model': model,
        'features': list(X.columns) # Penting: agar API Edge Node tahu urutan kolomnya
    }
    
    save_path = os.path.join(model_dir, model_filename)
    joblib.dump(artifact, save_path)
    print(f"✅ Model berhasil di-deploy ke: {save_path}")


def train_model():
    print("="*60)
    print("🚀 MEMULAI PROSES MLOps: TRAINING GBDT (DUAL MODEL: 3D vs 2D)")
    print("="*60)

    try:
        # =========================================================
        # 1. LOAD DATASET MATANG (Dari Preprocessing Pipeline)
        # =========================================================
        data_path = os.path.join(parent_dir, 'data', 'ready_for_training.csv')
        print(f"[1/3] Mengambil dataset matang dari: {data_path}...")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File {data_path} tidak ditemukan! Jalankan 'extract_and_preprocess.py' terlebih dahulu.")

        df = pd.read_csv(data_path)
        print(f"✅ Berhasil memuat {len(df)} baris data ayam yang sudah bersih dan dilabeli.")

        # =========================================================
        # 2. FEATURE SELECTION
        # =========================================================
        print("\n[2/3] Mempersiapkan Fitur Dasar (X) dan Target (y)...")
        # Kita menggunakan fitur yang tidak di-scale karena XGBoost berbasis Tree (kebal terhadap skala)
        features_3d = ['mask_area_px', 'bbox_width_px', 'bbox_height_px', 'age_days']
        features_2d = ['mask_area_px', 'bbox_width_px', 'age_days'] # Tanpa Tinggi
        
        # Target Y (Berat dalam KG)
        y = df['weight_kg']

        # =========================================================
        # 3. TRAINING MODEL (ABLATION STUDY)
        # =========================================================
        # A. Skenario 3D (Utama)
        X_3d = df[features_3d]
        train_and_save_model(X_3d, y, "gbdt_3d_model.pkl", "[3A/3]", "3D")

        # B. Skenario 2D (Pembanding/Baseline)
        X_2d = df[features_2d]
        train_and_save_model(X_2d, y, "gbdt_2d_model.pkl", "[3B/3]", "2D")

        print("\n🎉 SELURUH PROSES TRAINING SELESAI! MODEL SIAP DIGUNAKAN DI EDGE NODE.")

    except Exception as e:
        print(f"\n❌ Terjadi kesalahan saat proses MLOps: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_model()