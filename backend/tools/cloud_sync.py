import os
import subprocess
from datetime import datetime

def trigger_cloud_sync():
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{date_str}] 🚀 Memulai Sinkronisasi Instan ke Google Drive...")

    # 1. Definisi Path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    video_dir = os.path.abspath(os.path.join(base_dir, '..', 'videos')) 
    backup_dir = os.path.join(base_dir, 'backups')
    env_path = os.path.join(base_dir, '.env')

    os.makedirs(backup_dir, exist_ok=True)

    # =========================================================
    # 2. AMANKAN DATABASE POSTGRESQL (MENGGUNAKAN PG_DUMP)
    # =========================================================
    # Kita harus membaca DATABASE_URL dari file .env
    from dotenv import load_dotenv
    load_dotenv(env_path)
    
    db_url = os.environ.get('DATABASE_URL')
    
    if db_url and 'postgresql' in db_url:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_db_copy = os.path.join(backup_dir, f"db_snapshot_{timestamp}.sql")
        
        try:
            print("  ⏳ Mengekstrak data dari PostgreSQL...")
            # Menggunakan perintah pg_dump untuk membuat file backup murni (.sql)
            subprocess.run(
                ["pg_dump", db_url, "-f", safe_db_copy], 
                check=True
            )
            print(f"  ✅ Snapshot Database PostgreSQL berhasil dibuat: {safe_db_copy}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Gagal melakukan dump PostgreSQL. Pastikan 'pg_dump' sudah terinstall. Error: {e}")
        except Exception as e:
            print(f"  ❌ Terjadi kesalahan saat memproses database: {e}")
    else:
        print("  ⚠️ DATABASE_URL tidak ditemukan di .env atau bukan format PostgreSQL!")

    # =========================================================
    # 3. EKSEKUSI SINKRONISASI CLOUD (RCLONE)
    # =========================================================
    try:
        # A. Mengirim folder Backups (File .sql hasil dari PostgreSQL)
        print("  ⏳ Mengirim Backup Database ke Google Drive...")
        subprocess.run(
            ["rclone", "copy", backup_dir, "gdrive:SmartCoop_Tesis/Database/"], 
            check=True
        )

        # B. Mengirim folder Videos
        if os.path.exists(video_dir):
            print("  ⏳ Sinkronisasi Video Kandang...")
            subprocess.run(
                ["rclone", "sync", video_dir, "gdrive:SmartCoop_Tesis/Videos/"], 
                check=True
            )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Sinkronisasi File SELESAI!")
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Gagal melakukan sinkronisasi jaringan (Rclone): {e}")
    except Exception as e:
        print(f"  ❌ Terjadi kesalahan sistem Rclone: {e}")

if __name__ == "__main__":
    # Jika file ini dijalankan manual lewat terminal untuk testing
    trigger_cloud_sync()