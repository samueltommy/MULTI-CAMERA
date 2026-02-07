from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class FusedObject(Base):
    __tablename__ = 'fused_objects'

    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, unique=True, index=True) # The ID assigned by the tracker logic
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

    def to_dict(self):
        return {
            "id": self.id,
            "track_id": self.track_id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "snapshot_top": self.snapshot_top,
            "snapshot_side": self.snapshot_side,
            "position": {
                "top": [self.top_center_x, self.top_center_y],
                "side": [self.side_center_x, self.side_center_y]
            }
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

