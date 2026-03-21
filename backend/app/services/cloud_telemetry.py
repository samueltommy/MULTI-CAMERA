import requests
import json

class CloudTelemetryService:
    def __init__(self):
        # URL Supabase Anda (Ganti dengan URL asli dari dashboard Supabase Anda)
        base_url = "https://ndnejpwvukyhklkbejbd.supabase.co"
        
        # Endpoint otomatis dibuat oleh Supabase berdasarkan nama tabel!
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
        Mengirim Rangkuman DAN Detail Seluruh Ayam dalam satu kali jalan
        """
        summary_payload = {
            "session_id": session_id,
            "age_days": age_days,
            "average_weight_kg": avg_weight,
            "chicken_count": object_count
        }

        try:
            print(f"[Cloud] ☁️ Memulai Push Data Penuh untuk sesi: {session_id}")
            
            # 1. PUSH RANGKUMAN (SUMMARY)
            requests.post(self.supabase_url_summary, headers=self.headers, json=summary_payload, timeout=5.0)
            
            # 2. PUSH DETAIL SELURUH AYAM (BULK INSERT)
            # Supabase REST API bisa menerima sebuah List [...] untuk memasukkan banyak baris sekaligus
            if detailed_records and len(detailed_records) > 0:
                # Format ulang data dari lokal agar sesuai dengan kolom di tabel Supabase
                bulk_payload = []
                for record in detailed_records:
                    bulk_payload.append({
                        "session_id": session_id,
                        "track_id": record['track_id'],
                        "weight_kg": record['estimated_weight'],
                        "is_fused": record['is_fused'],
                        "timestamp": record['created_at'] # Jika Anda menyimpan waktu deteksinya
                    })
                
                res_details = requests.post(self.supabase_url_details, headers=self.headers, json=bulk_payload, timeout=10.0)
                
                if res_details.status_code in [200, 201]:
                    print(f"[Cloud] ✅ BERHASIL! Rangkuman & {len(bulk_payload)} baris detail ayam ter-backup ke awan.")
                else:
                    print(f"[Cloud] ⚠️ Detail gagal di-push: {res_details.text}")
                    
        except requests.exceptions.RequestException as e:
            print(f"[Cloud] ❌ Koneksi terputus! Data gagal dikirim, tersimpan aman di lokal: {e}")

cloud_telemetry = CloudTelemetryService()