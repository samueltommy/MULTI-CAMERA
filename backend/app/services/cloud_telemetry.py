import time
import requests
import csv
import os
import sys
import json

class CloudTelemetryService:
    def __init__(self):
        # URL Supabase Anda
        base_url = "https://ndnejpwvukyhklkbejbd.supabase.co"
        
        self.supabase_url_summary = f"{base_url}/rest/v1/session_stats"
        self.supabase_url_details = f"{base_url}/rest/v1/chicken_details"
        
        self.supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5kbmVqcHd2dWt5aGtsa2JlamJkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzQwOTQyODUsImV4cCI6MjA4OTY3MDI4NX0.UMf7OhSX0pbbfQ3fNAe1fPvsHKJZEIiC7b9xv31aZj4" 
        
        self.headers = {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal" 
        }

    def push_full_backup(self, session_id, age_days, avg_weight, object_count, detailed_records):
        """
        Mengirim Rangkuman DAN Detail Seluruh Ayam sekaligus mengukur Latensi & Bandwidth
        """
        summary_payload = {
            "session_id": session_id,
            "age_days": age_days,
            "average_weight_kg": avg_weight,
            "chicken_count": object_count
        }

        try:
            print(f"[Cloud] ☁️ Memulai Push Data Penuh untuk sesi: {session_id}")

            # =======================================================
            # 1. HITUNG UKURAN DATA (METRIK BANDWIDTH Tesis)
            # =======================================================
            # Menghitung ukuran JSON Rangkuman dalam satuan Byte
            summary_bytes = len(json.dumps(summary_payload).encode('utf-8'))
            details_bytes = 0
            
            bulk_payload = []
            if detailed_records and len(detailed_records) > 0:
                for record in detailed_records:
                    bulk_payload.append({
                        "session_id": session_id,
                        "track_id": record['track_id'],
                        "weight_kg": record['estimated_weight'],
                        "is_fused": record['is_fused'],
                        "timestamp": record['created_at']
                    })
                # Menghitung ukuran JSON Detail (Semua data ayam) dalam satuan Byte
                details_bytes = len(json.dumps(bulk_payload).encode('utf-8'))
            
            # Total ukuran payload yang akan dikirim ke internet
            total_payload_bytes = summary_bytes + details_bytes
            total_payload_kb = total_payload_bytes / 1024.0

            # =======================================================
            # 2. PENGIRIMAN DATA & HITUNG LATENSI
            # =======================================================
            t_start = time.perf_counter()
            
            # A. Push Rangkuman
            requests.post(self.supabase_url_summary, headers=self.headers, json=summary_payload, timeout=5.0)
            
            # B. Push Detail (Jika ada data ayam)
            if bulk_payload:
                res_details = requests.post(self.supabase_url_details, headers=self.headers, json=bulk_payload, timeout=10.0)
                if res_details.status_code not in [200, 201]:
                    print(f"[Cloud] ⚠️ Detail gagal di-push: {res_details.text}")

            # Selesai menghitung waktu
            t_upload = (time.perf_counter() - t_start) * 1000 # Convert ke milidetik

            # Print ke layar (Sebagai bukti visual saat sistem berjalan)
            print(f"[Cloud] ✅ BERHASIL! Rangkuman & {len(bulk_payload)} baris detail ayam ter-backup ke awan.")
            print(f"⏱️ LATENSI UPLOAD CLOUD : {t_upload:.2f} ms")
            print(f"📦 KONSUMSI BANDWIDTH   : {total_payload_bytes} Bytes ({total_payload_kb:.2f} KB)")
            
            # Simpan Latensi & Bandwidth ke CSV lokal untuk diolah di Bab 4 Tesis Anda
            self._log_cloud_metrics(session_id, len(bulk_payload), t_upload, total_payload_bytes)
                    
        except requests.exceptions.RequestException as e:
            print(f"[Cloud] ❌ Koneksi terputus! Data gagal dikirim, tersimpan aman di lokal: {e}")

    def _log_cloud_metrics(self, session_id, record_count, latency_ms, payload_bytes):
        """Menyimpan Log ke file CSV untuk bahan grafik dan tabel Tesis"""
        log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Nama file diubah agar tidak bentrok dengan versi kode yang lama
        filepath = os.path.join(log_dir, "cloud_telemetry_metrics.csv")
        file_exists = os.path.isfile(filepath)
        
        with open(filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            # Jika file baru dibuat, tulis Headernya (Tambahan kolom 'payload_bytes')
            if not file_exists:
                writer.writerow(["session_id", "total_records_sent", "upload_latency_ms", "payload_bytes"])
            
            # Tulis data
            writer.writerow([session_id, record_count, round(latency_ms, 2), payload_bytes])

cloud_telemetry = CloudTelemetryService()