import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import KNNImputer
from xgboost import XGBRegressor

# Setup path agar skrip bisa membaca modul dari folder 'app'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from app.database.session import SessionLocal
from app.database.models import FusedObject

def turn_outliers_to_nan(df, column):
    """
    Fungsi untuk mendeteksi nilai anomali (outlier) menggunakan metode IQR,
    lalu mengubahnya menjadi NaN (Kosong) agar nantinya direkonstruksi oleh KNN.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Deteksi yang di luar batas kewajaran
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    outliers_count = outliers.sum()
    
    if outliers_count > 0:
        print(f"    - Ditemukan {outliers_count} anomali piksel pada '{column}'. Diubah menjadi NaN (Kosong).")
        df.loc[outliers, column] = np.nan
        
    return df

def train_model():
    print("="*60)
    print("🚀 MEMULAI PROSES TRAINING GBDT (DENGAN KNN IMPUTATION)")
    print("="*60)

    db = SessionLocal()
    try:
        # =========================================================
        # 1. DATA INGESTION (Tarik Data Timbangan Asli)
        # =========================================================
        print("[1/6] Mengambil dataset dari PostgreSQL...")
        query = db.query(FusedObject).filter(
            FusedObject.actual_weight_gram != None,
            FusedObject.actual_weight_gram > 0  # Pastikan target berat manual masuk akal
        )
        
        df = pd.read_sql(query.statement, db.bind)
        initial_count = len(df)
        print(f"✅ Ditemukan {initial_count} baris data mentah berlabel emas.")

        if initial_count < 50:
            print("⚠️ Data terlalu sedikit untuk dilatih (Minimal 50 baris). Silakan timbang ayam lebih banyak.")
            return

        # =========================================================
        # 2. DATA CLEANSING & PREPROCESSING LOGIS
        # =========================================================
        print("\n[2/6] Melakukan Data Cleansing & Deteksi Anomali...")
        
        # A. Batas Logis Sensor (Buang error sensor parah)
        df = df[(df['mask_area_px'] > 50) | (df['mask_area_px'].isnull())] # Luas piksel tidak mungkin 0
        df.loc[df['score'] < 0.1, 'score'] = 0.1 # Batasi skor konfidensi minimum 10%
        df.loc[df['score'] > 1.0, 'score'] = 1.0 
        
        # Isi kategori teks yang kosong dengan default 'chicken'
        df['class_name'] = df['class_name'].fillna('chicken')

        # B. Deteksi Outlier Sensor -> Ubah ke NaN
        df = turn_outliers_to_nan(df, 'mask_area_px')
        df = turn_outliers_to_nan(df, 'bbox_width_px')
        if 'bbox_height_px' in df.columns:
            df = turn_outliers_to_nan(df, 'bbox_height_px')

        # C. Filter Outlier Human Error (Berat Asli)
        # Jika peternak salah ketik berat manual (misal 15000 gram padahal harusnya 1500), DIBUANG.
        Q1_y = df['actual_weight_gram'].quantile(0.25)
        Q3_y = df['actual_weight_gram'].quantile(0.75)
        IQR_y = Q3_y - Q1_y
        valid_y = (df['actual_weight_gram'] >= Q1_y - 1.5 * IQR_y) & (df['actual_weight_gram'] <= Q3_y + 1.5 * IQR_y)
        
        df_dropped = len(df) - valid_y.sum()
        df = df[valid_y]
        if df_dropped > 0:
            print(f"    - Membuang {df_dropped} baris karena 'actual_weight_gram' (Target) tidak masuk akal (salah ketik manual).")

        # =========================================================
        # 3. K-NEAREST NEIGHBORS (KNN) IMPUTATION
        # =========================================================
        print("\n[3/6] Merekonstruksi data sensor yang rusak menggunakan KNN...")
        
        # Pilih HANYA kolom angka untuk KNN Imputer
        numeric_cols_for_knn = [
            'mask_area_px', 'bbox_width_px', 'bbox_height_px', 
            'score', 'age_days', 'actual_weight_gram'
        ]
        
        # Pastikan kolom-kolom tersebut bertipe numerik float
        for col in numeric_cols_for_knn:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        
        # Lakukan rekonstruksi pada kolom numerik
        df_imputed_numeric = imputer.fit_transform(df[numeric_cols_for_knn])
        df[numeric_cols_for_knn] = df_imputed_numeric
        
        print(f"✅ Rekonstruksi selesai. Semua nilai NaN berhasil ditambal secara matematis.")

        # =========================================================
        # 4. FEATURE ENGINEERING
        # =========================================================
        print("\n[4/6] Mempersiapkan Fitur (X) dan Target (y)...")
        features = ['mask_area_px', 'bbox_width_px', 'bbox_height_px', 'score', 'age_days', 'is_fused', 'class_name']
        
        X = df[features].copy()
        
        # Ubah variabel kategori ('chicken wing', dll) menjadi angka biner (One-Hot Encoding)
        X = pd.get_dummies(X, columns=['class_name'], drop_first=False)
        
        # Target (y) diubah ke satuan Kilogram (kg)
        y = df['actual_weight_gram'] / 1000.0 

        # =========================================================
        # 5. TRAINING MODEL (XGBOOST)
        # =========================================================
        print("\n[5/6] Melatih model XGBoost Regressor (Tree-Based)...")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(
            n_estimators=200,      
            learning_rate=0.05,    
            max_depth=5,           
            subsample=0.8,         
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)

        # =========================================================
        # 6. EVALUASI DAN PENYIMPANAN MODEL
        # =========================================================
        print("\n[6/6] Mengevaluasi Performa Model...")
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print("=" * 60)
        print("📊 HASIL EVALUASI GBDT (PADA 20% DATA UJI/TEST SET):")
        print(f"  - MAE  (Rata-rata error absolut)  : {mae*1000:.1f} Gram")
        print(f"  - RMSE (Standar deviasi akurasi)  : {rmse*1000:.1f} Gram")
        print(f"  - R²   (Kecocokan Model Prediksi) : {r2*100:.2f} %")
        print("=" * 60)

        # Simpan Model dan Nama Kolom Fitur (Penting!)
        model_dir = os.path.join(parent_dir, 'app', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        artifact = {
            'model': model,
            'features': list(X.columns) # Disimpan agar urutan kolom konsisten saat live prediction
        }
        
        save_path = os.path.join(model_dir, 'best_gbdt_model.pkl')
        joblib.dump(artifact, save_path)
        print(f"\n✅ Model berhasil disimpan dengan nama: best_gbdt_model.pkl")
        print(f"✅ Lokasi: {save_path}")

    except Exception as e:
        print(f"\n❌ Terjadi kesalahan saat proses MLOps: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    train_model()