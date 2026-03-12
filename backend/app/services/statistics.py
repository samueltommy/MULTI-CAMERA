import numpy as np
from datetime import datetime, date
from sqlalchemy import func
from sklearn.ensemble import IsolationForest
from app.database.session import SessionLocal
from app.database.models import FusedObject, SessionStat, DailyStat, FarmSettings

class MLStatisticsService:
    def __init__(self):
        # AI untuk membuang outlier (asumsi maksimal 15% data error)
        self.outlier_model = IsolationForest(contamination=0.15, random_state=42)

    def process_session(self, session_id: str):
        db = SessionLocal()
        try:
            print(f"[ML Stats] Starting analysis for session: {session_id}")
            
            # 1. Ambil data mentah
            records = db.query(FusedObject).filter(
                FusedObject.session_id == session_id,
                FusedObject.estimated_weight > 0.0
            ).all()

            if not records:
                print(f"[ML Stats] No valid weight data found for {session_id}.")
                return

            # 2. Ambil nilai median per ayam
            weight_map = {}
            for r in records:
                if r.track_id not in weight_map:
                    weight_map[r.track_id] = []
                weight_map[r.track_id].append(r.estimated_weight)

            unique_chicken_weights = [np.median(weights) for weights in weight_map.values()]
            total_detected = len(unique_chicken_weights)
            
            if total_detected == 0:
                return

            # 3. JALANKAN AI ISOLATION FOREST
            X = np.array(unique_chicken_weights).reshape(-1, 1)
            if total_detected >= 4:
                preds = self.outlier_model.fit_predict(X)
                valid_weights = X[preds == 1].flatten() 
            else:
                valid_weights = X.flatten()

            # 4. Hitung Statistik Bersih
            if len(valid_weights) > 0:
                final_average = float(np.mean(valid_weights))
                final_std = float(np.std(valid_weights))
                valid_count = len(valid_weights)
            else:
                final_average = float(np.median(unique_chicken_weights))
                final_std = 0.0
                valid_count = len(unique_chicken_weights)

            # 5. Simpan ke SessionStat
            stats = db.query(SessionStat).filter(SessionStat.session_id == session_id).first()
            if not stats:
                stats = SessionStat(session_id=session_id)
                db.add(stats)
                
            stats.total_chickens_detected = total_detected
            stats.valid_chickens_used = valid_count
            stats.ai_average_weight_kg = final_average
            stats.ai_std_dev_kg = final_std
            db.commit()
            
            # 6. TRIGGER OTOMATIS RATA-RATA HARIAN
            session_date = stats.created_at.date()
            self._update_daily_stats(db, session_date)
            
            print(f"[ML Stats] Session {session_id} DONE! Avg: {final_average:.3f} kg")

        except Exception as e:
            print(f"[ML Stats] Error: {e}")
            db.rollback()
        finally:
            db.close()

    def _update_daily_stats(self, db, target_date: date):
        try:
            sessions_today = db.query(SessionStat).filter(
                func.date(SessionStat.created_at) == target_date
            ).all()

            total_weight_mass = 0.0
            total_valid_chickens = 0

            # Hitung Weighted Average (Rata-rata Tertimbang)
            for s in sessions_today:
                total_weight_mass += (s.ai_average_weight_kg * s.valid_chickens_used)
                total_valid_chickens += s.valid_chickens_used

            daily_cumulative_average = (total_weight_mass / total_valid_chickens) if total_valid_chickens > 0 else 0.0

            # Prediksi Waktu Panen
            farm_setting = db.query(FarmSettings).first()
            target_kg = farm_setting.target_harvest_weight_kg if farm_setting else 2.0
            
            from app.services.growth_predictor import growth_predictor
            bio_age = 0.0
            days_remaining = 0.0

            if daily_cumulative_average > 0:
                pred = growth_predictor.predict_harvest(daily_cumulative_average, target_kg)
                if pred:
                    bio_age = float(pred['current_biological_age_days'])
                    days_remaining = float(pred['estimated_days_remaining'])

            # Simpan/Update DailyStat
            daily_record = db.query(DailyStat).filter(DailyStat.date == target_date).first()
            if not daily_record:
                daily_record = DailyStat(date=target_date)
                db.add(daily_record)

            daily_record.total_sessions = int(len(sessions_today))
            daily_record.total_valid_chickens = int(total_valid_chickens)
            daily_record.daily_average_kg = float(daily_cumulative_average)
            daily_record.biological_age_days = float(bio_age)
            daily_record.estimated_days_remaining = float(days_remaining)
            
            db.commit()
            print(f"[Daily Stats] UPDATE SUCCESS for {target_date}: {daily_cumulative_average:.3f} kg (from {len(sessions_today)} sessions)")
            
        except Exception as e:
            print(f"[Daily Stats] Failed: {e}")
            db.rollback()

ml_statistics_service = MLStatisticsService()