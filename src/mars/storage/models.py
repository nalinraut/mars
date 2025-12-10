from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, DateTime, JSON, ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship

class Base(DeclarativeBase):
    pass

class IngestionRecord(Base):
    __tablename__ = "ingestion_records"

    image_id: Mapped[str] = mapped_column(String, primary_key=True)
    file_hash: Mapped[str] = mapped_column(String, index=True)
    original_path: Mapped[str] = mapped_column(String)
    staging_path: Mapped[str] = mapped_column(String)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    file_format: Mapped[str] = mapped_column(String)
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String)
    
    # Additional Metadata
    task_type: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    dataset_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    ingestion_version: Mapped[str] = mapped_column(String, default="1.0.0")
    
    # Relationship (One-to-One)
    quality_record: Mapped[Optional["SceneQualityRecord"]] = relationship(
        back_populates="ingestion_record", 
        uselist=False,
        cascade="all, delete-orphan"
    )

class SceneQualityRecord(Base):
    __tablename__ = "scene_quality_records"
    
    scene_id: Mapped[str] = mapped_column(String, ForeignKey("ingestion_records.image_id"), primary_key=True)
    computed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    pipeline_version: Mapped[str] = mapped_column(String, default="1.0.0")

    # Tier 1: Geometric
    mesh_integrity: Mapped[float] = mapped_column(Float, default=0.0)
    reconstruction_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    collision_mesh_quality: Mapped[float] = mapped_column(Float, default=0.0)

    # Tier 2: Physical
    stability: Mapped[float] = mapped_column(Float, default=0.0)
    mass_plausibility: Mapped[float] = mapped_column(Float, default=0.0)
    contact_quality: Mapped[float] = mapped_column(Float, default=0.0)

    # Tier 3: Scene Completeness
    task_readiness: Mapped[float] = mapped_column(Float, default=0.0)
    object_count_score: Mapped[float] = mapped_column(Float, default=0.0)
    spatial_layout: Mapped[float] = mapped_column(Float, default=0.0)

    # Tier 4: Semantic
    segmentation_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    category_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    scene_coherence: Mapped[float] = mapped_column(Float, default=0.0)

    # Tier 5: Training Utility
    diversity: Mapped[float] = mapped_column(Float, default=0.0)
    difficulty_estimate: Mapped[float] = mapped_column(Float, default=0.0)
    randomization_potential: Mapped[float] = mapped_column(Float, default=0.0)

    # Composite Scores
    overall_score: Mapped[float] = mapped_column(Float, default=0.0)
    quick_score: Mapped[float] = mapped_column(Float, default=0.0)
    simulation_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Flags (stored as JSON for flexibility)
    flags: Mapped[dict] = mapped_column(JSON, default={})
    
    # Relationship
    ingestion_record: Mapped["IngestionRecord"] = relationship(back_populates="quality_record")

def init_db(db_url: str):
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)
