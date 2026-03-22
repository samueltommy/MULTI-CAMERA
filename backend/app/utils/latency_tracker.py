import os
import csv
import time

class LatencyTracker:
    def __init__(self, session_id):
        self.session_id = session_id
        
        # Buat folder logs jika belum ada
        self.log_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.filepath = os.path.join(self.log_dir, f"latency_{session_id}.csv")
        
        # Buat file dan tulis Header CSV
        with open(self.filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "frame_id", 
                "rtsp_capture_ms", 
                "yolo_inference_ms", 
                "fusion_geometry_ms", 
                "total_pipeline_ms"
            ])
            
        self.records = []
        
    def record_frame(self, frame_id, t_capture, t_infer, t_fuse, t_total):
        """Menyimpan data (dalam milidetik) ke RAM sementara"""
        self.records.append([
            frame_id,
            round(t_capture * 1000, 2), # Konversi ke Milidetik (ms)
            round(t_infer * 1000, 2),
            round(t_fuse * 1000, 2),
            round(t_total * 1000, 2)
        ])
        
        # Flush ke CSV setiap 50 frame agar RAM tidak jebol, tapi I/O tetap ringan
        if len(self.records) >= 50:
            self._flush()

    def _flush(self):
        """Menulis isi RAM ke Harddisk (CSV)"""
        if not self.records:
            return
            
        with open(self.filepath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.records)
        self.records.clear()

    def close(self):
        """Dipanggil saat sesi selesai untuk menulis sisa data"""
        self._flush()
        print(f"📊 [Latency Tracker] Log performa tersimpan di: {self.filepath}")