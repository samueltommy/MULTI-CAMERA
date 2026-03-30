import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import psycopg2 
from sqlalchemy import create_engine 
from scipy import stats

# ==========================================
# KONFIGURASI DATABASE POSTGRESQL ASLI
# ==========================================
DB_HOST = "localhost"        # Ganti dengan IP database Anda (misal: 127.0.0.1)
DB_PORT = "5432"             # Port default PostgreSQL
DB_NAME = "chickens_db"    # Ganti dengan nama database Anda
DB_USER = "postgres"         # Ganti dengan username database Anda
DB_PASS = "admin"    # Ganti dengan password database Anda

TABLE_NAME = "fused_objects" 

# Pastikan folder data tersedia
os.makedirs("../data", exist_ok=True)
RAW_DATA_PATH = "../data/raw_fused_detections.csv"
READY_DATA_PATH = "../data/ready_for_training.csv"

class PoultryDataPipeline:
    def __init__(self):
        self.df = None

    # ==========================================
    # STEP 1: EKSTRAKSI DATA DARI POSTGRESQL
    # ==========================================
    def extract_from_postgres(self):
        print(f"📥 [STEP 1] Mengunduh data dari PostgreSQL (Tabel: {TABLE_NAME})...")
        
        try:
            connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
            engine = create_engine(connection_string)
            
            query = f"SELECT * FROM {TABLE_NAME};"
            self.df = pd.read_sql_query(query, engine)
            
            # --- [TAMBAHAN 1] MENGHITUNG UMUR OTOMATIS (AGE_DAYS) ---
            print("   -> Menghitung umur ayam (age_days) secara otomatis...")
            # 1. Ubah kolom created_at menjadi format Waktu (Datetime)
            self.df['created_at'] = pd.to_datetime(self.df['created_at'])
            
            # 2. Buang jam/menit (Hanya ambil tanggalnya saja agar presisi 1 hari)
            tanggal_rekaman = self.df['created_at'].dt.normalize()
            
            # 3. Masukkan Tanggal Jangkar (Anchor Date) dari Peternak
            # Format: 'YYYY-MM-DD'
            tanggal_jangkar = pd.to_datetime('2026-03-25')
            umur_pada_jangkar = 13
            
            # 4. Hitung selisih hari dan ciptakan kolom age_days
            selisih_hari = (tanggal_rekaman - tanggal_jangkar).dt.days
            self.df['age_days'] = umur_pada_jangkar + selisih_hari
            # --------------------------------------------------------

            # --- [TAMBAHAN 2] FILTER ABLATION STUDY (MURNI 3D) ---
            if 'is_fused' in self.df.columns:
                initial_count = len(self.df)
                
                # Konversi ke boolean jika terbaca sebagai text dari PostgreSQL
                if self.df['is_fused'].dtype == object:
                    self.df['is_fused'] = self.df['is_fused'].astype(str).str.lower() == 'true'
                
                # Saring hanya data yang is_fused == True
                self.df = self.df[self.df['is_fused'] == True] 
                dropped_2d = initial_count - len(self.df)
                print(f"   -> FILTER: Membuang {dropped_2d} data yang hanya 2D (is_fused=false) demi keadilan Ablation Study.")
            # --------------------------------------------------------
            
            self.df.to_csv(RAW_DATA_PATH, index=False)
            print(f"✅ Ekstraksi Selesai! Total data 3D murni: {len(self.df)} baris. Tersimpan di {RAW_DATA_PATH}\n")
            
        except Exception as e:
            raise Exception(f"Gagal koneksi PostgreSQL: {e}")

    # ==========================================
    # STEP 2: IMPUTASI DATA KOSONG DENGAN KNN
    # ==========================================
    def impute_missing_data_knn(self):
        print("💉 [STEP 2] Mulai Imputasi KNN untuk menambal data dimensi yang kosong/0...")
        
        # Pastikan nilai 0 atau negatif diubah menjadi NaN agar bisa dideteksi oleh KNN
        fitur_dimensi = ['bbox_width_px', 'bbox_height_px', 'mask_area_px']
        
        # Pengecekan apakah kolom dimensi ada di database
        missing_cols = [col for col in fitur_dimensi if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Error: Kolom {missing_cols} tidak ditemukan di database Anda! Pastikan query SQL menarik kolom tersebut.")

        for col in fitur_dimensi:
            self.df[col] = self.df[col].replace(0, np.nan)
            # --- TAMBAHAN WAJIB ---
            # Paksa tipe data menjadi angka murni (Float)
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Cek berapa banyak data yang kosong
        missing_count = self.df[fitur_dimensi].isna().sum().sum()
        if missing_count == 0:
            print("✅ Tidak ada data kosong. Melewati proses KNN.\n")
            return

        print(f"   -> Ditemukan {missing_count} sel data kosong. Menjalankan KNNImputer(k=5)...")
        
        kolom_knn = ['age_days', 'bbox_width_px', 'mask_area_px', 'bbox_height_px']
        if 'age_days' not in self.df.columns:
            raise ValueError("Error: Kolom 'age_days' tidak ditemukan! Diperlukan oleh KNN untuk mencari umur yang sama.")

        imputer = KNNImputer(n_neighbors=5, weights='distance')
        self.df[kolom_knn] = imputer.fit_transform(self.df[kolom_knn])
        print("✅ Imputasi KNN Selesai! Seluruh data dimensi telah terisi secara logis.\n")

    # ==========================================
    # STEP 3: DATA CLEANSING DENGAN ISOLATION FOREST
    # ==========================================
    def cleanse_data(self):
        print("🧹 [STEP 3] Mulai Cleansing dengan Isolation Forest...")
        
        features_to_check = ['bbox_width_px', 'bbox_height_px', 'mask_area_px']
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        self.df['is_anomaly'] = iso_forest.fit_predict(self.df[features_to_check])
        
        anomalies_dropped = len(self.df[self.df['is_anomaly'] == -1])
        self.df = self.df[self.df['is_anomaly'] == 1].drop(columns=['is_anomaly'])
        
        print(f"✅ Cleansing Selesai! Membuang {anomalies_dropped} data anomali (outlier/error bounding box).\n")

    # ==========================================
    # STEP 4: PELABELAN PROPORSIONAL (PSEUDO-LABELING)
    # ==========================================
    def proportional_pseudo_labeling(self, manual_ground_truth):
        print("⚖️ [STEP 4] Mulai Pelabelan Quantile Mapping (Distribusi Normal Biologis)...")
        from scipy import stats # Pastikan di-import
        
        self.df['weight_kg'] = np.nan
        days = self.df['age_days'].dropna().unique()
        
        for day in days:
            if day in manual_ground_truth:
                actual_avg_weight = manual_ground_truth[day]
                day_mask = self.df['age_days'] == day
                
                # Standar deviasi biologi broiler (CV = 12% dari rata-rata)
                cv = 0.12 
                std_weight = actual_avg_weight * cv
                
                # 1. Hitung Pendekatan Volume 3D (Luas Atas x Tinggi Samping)
                volume_semu = self.df.loc[day_mask, 'mask_area_px'] * self.df.loc[day_mask, 'bbox_height_px']
                
                # 2. Me-ranking VOLUME ayam dari yang terkecil sampai terbesar
                percentiles = volume_semu.rank(pct=True)
                
                # 3. Mencegah nilai mutlak 0 atau 1
                percentiles = np.clip(percentiles, 0.001, 0.999)
                
                # 4. Memetakan persentil tersebut ke Kurva Lonceng Distribusi Normal
                self.df.loc[day_mask, 'weight_kg'] = stats.norm.ppf(
                    percentiles, 
                    loc=actual_avg_weight,
                    scale=std_weight
                )
                
                print(f"   -> Hari ke-{int(day)}: Melabeli {day_mask.sum()} ayam.")
                print(f"      [Cek Biologi] Max: {self.df.loc[day_mask, 'weight_kg'].max():.3f} kg | Min: {self.df.loc[day_mask, 'weight_kg'].min():.3f} kg")
            else:
                print(f"   -> PERINGATAN: Hari ke-{int(day)} tidak memiliki data manual! Dilewati.")

        self.df = self.df.dropna(subset=['weight_kg'])
        print(f"✅ Pelabelan Selesai! Total {len(self.df)} baris data sukses dilabeli.\n")

    # ==========================================
    # STEP 5: EKSPOR DATA MATANG
    # ==========================================
    def export_ready_data(self):
        print("⚙️ [STEP 5] Mulai Feature Engineering (Standard Scaler)...")
        
        scaler = StandardScaler()
        features_to_scale = ['bbox_width_px', 'bbox_height_px', 'mask_area_px']
        
        for col in features_to_scale:
            self.df[f'{col}_scaled'] = scaler.fit_transform(self.df[[col]])

        self.df.to_csv(READY_DATA_PATH, index=False)
        print(f"🚀 SUCCESS! Dataset matang disimpan di: {READY_DATA_PATH}\n")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # ⚠️ INPUT MANUAL: Masukkan Rata-rata Timbangan Manual (Ground Truth) Peternak di sini
    timbangan_peternak = {
        13: 0.466,
        14: 0.536,
        15: 0.582,
        16: 0.654,
        17: 0.711
    }
    
    pipeline = PoultryDataPipeline()
    
    try:
        pipeline.extract_from_postgres()            # <--- MENGGUNAKAN POSTGRESQL LANGSUNG
        pipeline.impute_missing_data_knn()          
        pipeline.cleanse_data()                     
        pipeline.proportional_pseudo_labeling(timbangan_peternak)
        pipeline.export_ready_data()
    except Exception as e:
        print(f"\n❌ TERJADI KESALAHAN: {e}")