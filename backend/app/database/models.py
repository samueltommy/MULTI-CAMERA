from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Date
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, date

Base = declarative_base()

class FusedObject(Base):
    __tablename__ = 'fused_objects'

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True) # The ID assigned by the tracker logic
    session_id = Column(String, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Snapshot paths
    snapshot_top = Column(String, nullable=True)
    snapshot_side = Column(String, nullable=True)
    
    # Metadata at capture time
    top_center_x = Column(Float, nullable=True)
    top_center_y = Column(Float, nullable=True)
    side_center_x = Column(Float, nullable=True)
    side_center_y = Column(Float, nullable=True)
    
    class_id = Column(Integer, nullable=True)
    score = Column(Float, nullable=True)

    estimated_weight = Column(Float, nullable=True)
    is_fused = Column(Boolean, default=False) # True = Volume method, False = Area method

    age_days = Column(Integer, nullable=True) # Umur ayam saat data diambil
    mask_area_px = Column(Float, nullable=True) # Luas segmentasi
    bbox_width_px = Column(Float, nullable=True) # Lebar kotak
    bbox_height_px = Column(Float, nullable=True) # Tinggi kotak
    cm_per_pixel = Column(Float, nullable=True) # Skala ArUco saat perekaman
    actual_weight_gram = Column(Float, nullable=True) # Kolom kosong untuk diisi manual hasil timbangan

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "track_id": self.track_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "snapshot_top": self.snapshot_top,
            "snapshot_side": self.snapshot_side,
            "estimated_weight": self.estimated_weight,
            "is_fused": self.is_fused,
            "age": self.age_days, # Kirim umur ke frontend
            "confidence": round((self.score or 0.98) * 100, 1),
            "position": {
                "top": [self.top_center_x, self.top_center_y],
                "side": [self.side_center_x, self.side_center_y]
            },
            "mask_area": self.mask_area_px,
            "bbox_width": self.bbox_width_px,
            "bbox_height": self.bbox_height_px,
            "cm_per_pixel": self.cm_per_pixel,
            "actual_weight": self.actual_weight_gram
        }

class Calibration(Base):
    __tablename__ = 'calibrations'

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    name = Column(String, nullable=True)
    
    # Store matrix as a JSON string
    matrix_json = Column(String, nullable=False)
    
    # Metadata
    is_active = Column(Boolean, default=True) 
    notes = Column(String, nullable=True)

    def to_dict(self):
        import json
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "name": self.name,
            "matrix": json.loads(self.matrix_json),
            "is_active": self.is_active
        }

class FarmSettings(Base):
    __tablename__ = 'farm_settings'
    
    id = Column(Integer, primary_key=True, index=True)
    chick_in_date = Column(Date, nullable=True) 
    manual_age_override = Column(Integer, nullable=True)
    target_harvest_weight_kg = Column(Float, default=2.0) 
    
    def to_dict(self):
        return {
            "chick_in_date": self.chick_in_date.isoformat() if self.chick_in_date else None,
            "manual_age_override": self.manual_age_override,
            "target_harvest_weight_kg": self.target_harvest_weight_kg
        }

class SessionStat(Base):
    __tablename__ = 'session_stats'

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Statistik dari AI (ML)
    total_chickens_detected = Column(Integer, default=0) 
    valid_chickens_used = Column(Integer, default=0)     
    ai_average_weight_kg = Column(Float, default=0.0)    
    ai_std_dev_kg = Column(Float, default=0.0)           

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "total_chickens_detected": self.total_chickens_detected,
            "valid_chickens_used": self.valid_chickens_used,
            "ai_average_weight_kg": round(self.ai_average_weight_kg, 3),
        }

class DailyStat(Base):
    __tablename__ = 'daily_stats'
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, unique=True, index=True, default=date.today) # Menyimpan YYYY-MM-DD
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    total_sessions = Column(Integer, default=0)
    total_valid_chickens = Column(Integer, default=0)
    daily_average_kg = Column(Float, default=0.0) # Akumulasi Otomatis (Weighted Average)
    biological_age_days = Column(Float, default=0.0)
    estimated_days_remaining = Column(Float, default=0.0)

    def to_dict(self):
        return {
            "date": self.date.isoformat() if self.date else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "total_sessions": self.total_sessions,
            "total_chickens": self.total_valid_chickens,
            "daily_average_kg": round(self.daily_average_kg, 3),
            "biological_age_days": round(self.biological_age_days, 1),
            "estimated_days_remaining": round(self.estimated_days_remaining, 1)
        }