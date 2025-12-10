import logging
from sqlalchemy.orm import Session
from datetime import datetime

from .models import IngestionRecord, SceneQualityRecord, init_db

logger = logging.getLogger(__name__)

def update_index(
    db_url: str, 
    image_id: str, 
    quality_record: dict,
    final_path: str
):
    """
    Update database with final location and quality scores.
    """
    SessionLocal = init_db(db_url)
    session = SessionLocal()
    
    try:
        # Update Ingestion Status
        record = session.query(IngestionRecord).filter_by(image_id=image_id).first()
        if not record:
            logger.error(f"Record not found for {image_id}")
            return
            
        record.status = "complete"
        # Ideally we'd add a 'storage_path' column to ingestion record or separate table
        # For now, logging it.
        
        # Create/Update Quality Record
        # We flatten the nested quality dict to match the model
        q_data = {}
        
        # Map Tier 1
        t1 = quality_record.get('tier_1_geometric', {})
        q_data['mesh_integrity'] = t1.get('mesh_integrity', 0)
        q_data['reconstruction_confidence'] = t1.get('reconstruction_confidence', 0)
        q_data['collision_mesh_quality'] = t1.get('collision_mesh_quality', 0)
        
        # Map Tier 2
        t2 = quality_record.get('tier_2_physical', {})
        q_data['stability'] = t2.get('stability', 0)
        q_data['mass_plausibility'] = t2.get('mass_plausibility', 0)
        
        # Map Composite
        comp = quality_record.get('composite', {})
        q_data['overall_score'] = comp.get('overall', 0)
        q_data['simulation_score'] = comp.get('simulation', 0)
        
        # Check existing
        q_rec = session.query(SceneQualityRecord).filter_by(scene_id=image_id).first()
        if not q_rec:
            q_rec = SceneQualityRecord(scene_id=image_id)
            session.add(q_rec)
            
        # Apply updates
        for k, v in q_data.items():
            if hasattr(q_rec, k):
                setattr(q_rec, k, v)
                
        q_rec.computed_at = datetime.utcnow()
        
        session.commit()
        logger.info(f"Indexed quality scores for {image_id}")
        
    except Exception as e:
        session.rollback()
        logger.error(f"Indexing failed: {e}")
        raise e
    finally:
        session.close()

