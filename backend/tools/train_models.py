import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import Algoritma
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# 1. LOAD DATASET
DATA_PATH = "../data/ready_for_training.csv"
print(f"📦 Memuat dataset dari: {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

# 2. DEFINISI FITUR (SKENARIO 3D vs 2D)
# Target prediksi
y = df['weight_kg']

# Skenario 3D (Lengkap: Lebar, Tinggi, Luas, Umur)
# Kita pakai yang _scaled agar Neural Network lebih optimal
X_3D = df[['bbox_width_px_scaled', 'bbox_height_px_scaled', 'mask_area_px_scaled', 'age_days']]

# Skenario 2D (Tinggi Dibuang/Di-drop paksa)
X_2D = df[['bbox_width_px_scaled', 'mask_area_px_scaled', 'age_days']]

# Membagi data (80% Training, 20% Testing) dengan seed yang sama agar adil
X_3D_train, X_3D_test, y_train, y_test = train_test_split(X_3D, y, test_size=0.2, random_state=42)
X_2D_train, X_2D_test, _, _ = train_test_split(X_2D, y, test_size=0.2, random_state=42)

# 3. PERSIAPAN MODEL
models = {
    "Polynomial Ridge (Orde 2)": make_pipeline(PolynomialFeatures(2), Ridge(alpha=1.0)),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1),
    "MLP (Deep Learning)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = []

def evaluate_model(model_name, model, X_train, X_test, scenario_name):
    # Hitung waktu training (Inference Time Setup)
    start_time = time.time()
    model.fit(X_train, y_train)
    
    # Hitung waktu prediksi
    start_infer = time.time()
    y_pred = model.predict(X_test)
    end_infer = time.time()
    
    infer_time_ms = ((end_infer - start_infer) / len(X_test)) * 1000  # Latency per gambar (ms)
    
    # Metrik Evaluasi
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": model_name,
        "Skenario": scenario_name,
        "MAE (kg)": round(mae, 4),
        "RMSE (kg)": round(rmse, 4),
        "R2 Score": round(r2, 4),
        "Latency (ms)": round(infer_time_ms, 3)
    })
    print(f"✅ {model_name} ({scenario_name}) selesai dilatih.")

# 4. MULAI EKSPERIMEN (ABLATION STUDY)
print("\n🚀 MEMULAI TRAINING: SKENARIO 3D (Lebar + Luas + TINGGI)...")
for name, model in models.items():
    evaluate_model(name, model, X_3D_train, X_3D_test, "3D (Kamera Ganda)")

print("\n🚀 MEMULAI TRAINING: SKENARIO 2D (Hanya Lebar + Luas)...")
for name, model in models.items():
    evaluate_model(name, model, X_2D_train, X_2D_test, "2D (Kamera Atas Saja)")

# 5. CETAK HASIL TESIS
results_df = pd.DataFrame(results)

# Urutkan berdasarkan RMSE terbaik (terendah)
results_df = results_df.sort_values(by=["Skenario", "RMSE (kg)"])

print("\n" + "="*80)
print("🏆 HASIL AKHIR KOMPARASI ALGORITMA & ABLATION STUDY (BAB 4 TESIS)".center(80))
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# Simpan tabel ke CSV agar mudah dipindah ke Excel / Word
results_df.to_csv("../data/ablation_study_results.csv", index=False)
print("\n📊 Tabel hasil telah disimpan di '../data/ablation_study_results.csv'")