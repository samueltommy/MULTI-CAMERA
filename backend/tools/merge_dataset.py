import pandas as pd
import os

def merge_cloud_and_edge_data():
    print("🔄 Memulai proses penggabungan dataset Cloud dan Edge...")

    # 1. Tentukan path file (Sesuaikan jika lokasi foldernya berbeda)
    cloud_path = "../data/ready_for_training.csv"
    edge_path = "../data/ready_for_training_edge.csv"
    output_path = "../data/ready_for_training_combined.csv"

    # 2. Baca kedua dataset
    print(f"📥 Membaca data Cloud dari: {cloud_path}")
    df_cloud = pd.read_csv(cloud_path)
    
    print(f"📥 Membaca data Edge dari: {edge_path}")
    df_edge = pd.read_csv(edge_path)

    # 3. (Opsional) Tambahkan penanda sumber data untuk kemudahan analisis nanti
    df_cloud['data_source'] = 'cloud'
    df_edge['data_source'] = 'edge'

    # 4. Gabungkan kedua dataframe
    # ignore_index=True akan membuat ulang nomor urut index dari 0
    df_combined = pd.concat([df_cloud, df_edge], ignore_index=True)

    # 5. Cek jika ada duplikasi ID ayam (opsional, jika dirasa perlu)
    # Jika sistem Cloud dan Edge merekam ID yang sama di waktu yang sama, kita bisa drop duplikat
    initial_len = len(df_combined)
    # Misalnya kita anggap duplikat jika track_id dan session_id sama
    # df_combined = df_combined.drop_duplicates(subset=['session_id', 'track_id'])
    # print(f"🧹 Membuang {initial_len - len(df_combined)} data duplikat (jika ada).")

    # 6. Simpan hasil gabungan ke file baru
    df_combined.to_csv(output_path, index=False)
    
    print("=" * 50)
    print(f"✅ Penggabungan Selesai!")
    print(f"   - Total Data Cloud : {len(df_cloud)} baris")
    print(f"   - Total Data Edge  : {len(df_edge)} baris")
    print(f"   - Total Gabungan   : {len(df_combined)} baris")
    print(f"📁 File baru disimpan di: {output_path}")
    print("=" * 50)

if __name__ == "__main__":
    merge_cloud_and_edge_data()