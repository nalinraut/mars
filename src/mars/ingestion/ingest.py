import logging
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import uuid

import imagehash
from PIL import Image, UnidentifiedImageError
from prefect import task, flow, get_run_logger
from omegaconf import OmegaConf
from sqlalchemy.orm import Session

from src.mars.storage.models import IngestionRecord, init_db

# Configure logging
logger = logging.getLogger(__name__)

def get_session(db_url: str) -> Session:
    SessionLocal = init_db(db_url)
    return SessionLocal()

@task
def setup_storage(db_url: str):
    """Initialize database tables."""
    init_db(db_url)

@task
def load_ingest_config(config_path: str = "config/ingest.yaml") -> Dict[str, Any]:
    """Load ingestion configuration."""
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)

@task
def validate_image(
    image_path: str, 
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate image format, resolution, and corruption.
    Returns image metadata if valid, raises error if not.
    """
    logger = get_run_logger()
    path = Path(image_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Check extension
    suffix = path.suffix.lower()
    supported = config['ingestion']['supported_formats']
    if suffix not in supported:
        raise ValueError(f"Unsupported format {suffix}. Supported: {supported}")
    
    try:
        with Image.open(path) as img:
            img.verify()  # Check for corruption
            
        # Re-open to check properties (verify consumes the file pointer)
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format
            
            # Resolution check
            min_w = config['ingestion']['resolution']['min_width']
            min_h = config['ingestion']['resolution']['min_height']
            max_w = config['ingestion']['resolution']['max_width']
            max_h = config['ingestion']['resolution']['max_height']
            
            if width < min_w or height < min_h:
                raise ValueError(f"Resolution too low: {width}x{height}. Min: {min_w}x{min_h}")
            if width > max_w or height > max_h:
                raise ValueError(f"Resolution too high: {width}x{height}. Max: {max_w}x{max_h}")
            
            return {
                "width": width,
                "height": height,
                "format": fmt,
                "size_bytes": path.stat().st_size,
                "valid": True
            }
            
    except UnidentifiedImageError:
        raise ValueError(f"Corrupted or invalid image file: {image_path}")
    except Exception as e:
        raise ValueError(f"Validation failed: {str(e)}")

@task
def compute_perceptual_hash(image_path: str) -> str:
    """Compute perceptual hash of the image."""
    with Image.open(image_path) as img:
        # specific hash algorithm can be configurable, defaulting to phash
        img_hash = imagehash.phash(img)
        return str(img_hash)

@task
def check_duplicate(
    file_hash: str, 
    db_url: str
) -> Optional[str]:
    """
    Check if hash exists in index. 
    Returns existing image_id if duplicate, None otherwise.
    """
    session = get_session(db_url)
    try:
        record = session.query(IngestionRecord).filter_by(file_hash=file_hash).first()
        if record:
            return record.image_id
        return None
    finally:
        session.close()

@task
def get_existing_record(image_id: str, db_url: str) -> Optional[Dict[str, Any]]:
    """Fetch existing ingestion record by image_id."""
    session = get_session(db_url)
    try:
        record = session.query(IngestionRecord).filter_by(image_id=image_id).first()
        if record:
            return {
                "image_id": record.image_id,
                "hash": record.file_hash,
                "paths": {
                    "original_path": record.original_path,
                    "staging_path": record.staging_path,
                    "thumbnail_path": record.thumbnail_path
                }
            }
        return None
    finally:
        session.close()

@task
def generate_staging_files(
    image_path: str, 
    image_id: str, 
    config: Dict[str, Any]
) -> Dict[str, str]:
    """
    Copy image to staging and generate thumbnail.
    Returns dictionary of paths.
    """
    logger = get_run_logger()
    source_path = Path(image_path)
    staging_root = Path(config['ingestion']['staging_dir'])
    staging_root.mkdir(parents=True, exist_ok=True)
    
    # Define paths
    # Use image_id to avoid filename collisions
    ext = source_path.suffix
    staging_filename = f"{image_id}{ext}"
    staging_path = staging_root / staging_filename
    
    # Copy original
    shutil.copy2(source_path, staging_path)
    logger.info(f"Copied to staging: {staging_path}")
    
    paths = {
        "original_path": str(source_path),
        "staging_path": str(staging_path),
        "thumbnail_path": None
    }
    
    # Generate thumbnail if enabled
    if config['ingestion']['thumbnails']['enabled']:
        thumb_size = tuple(config['ingestion']['thumbnails']['size'])
        thumb_filename = f"{image_id}_thumb.jpg" # Always JPG for thumbs usually
        thumb_path = staging_root / thumb_filename
        
        with Image.open(staging_path) as img:
            img.thumbnail(thumb_size)
            img.save(thumb_path, "JPEG")
            
        paths["thumbnail_path"] = str(thumb_path)
        logger.info(f"Generated thumbnail: {thumb_path}")
        
    return paths

@task
def register_ingestion(
    image_id: str,
    file_hash: str,
    metadata: Dict[str, Any],
    paths: Dict[str, str],
    db_url: str,
    source_meta: Optional[Dict[str, Any]] = None,
    task_type: Optional[str] = None,
    dataset_id: Optional[str] = None,
    ingestion_version: str = "1.0.0"
) -> Dict[str, Any]:
    """Persist ingestion record to database."""
    session = get_session(db_url)
    try:
        record = IngestionRecord(
            image_id=image_id,
            file_hash=file_hash,
            original_path=paths['original_path'],
            staging_path=paths['staging_path'],
            thumbnail_path=paths['thumbnail_path'],
            width=metadata['width'],
            height=metadata['height'],
            file_format=metadata['format'],
            file_size_bytes=metadata['size_bytes'],
            metadata_json=source_meta,
            status="ingested",
            task_type=task_type,
            dataset_id=dataset_id,
            ingestion_version=ingestion_version
        )
        session.add(record)
        session.commit()
        
        return {
            "image_id": image_id,
            "status": "ingested",
            "record": {
                "hash": file_hash,
                "paths": paths
            }
        }
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

@flow(name="Ingest Scene Image")
def ingest_image(
    image_path: str, 
    source_metadata: Optional[Dict[str, Any]] = None,
    config_path: str = "config/ingest.yaml",
    task_type: Optional[str] = None,
    dataset_id: Optional[str] = None,
    ingestion_version: str = "1.0.0"
) -> Dict[str, Any]:
    """
    Main ingestion flow.
    
    1. Load Config
    2. Validate Image
    3. Compute Hash
    4. Check Duplicates
    5. Generate Staging Files & Thumbnail
    6. Register Record
    """
    logger = get_run_logger()
    logger.info(f"Starting ingestion for {image_path}")
    
    # 1. Load Config
    config = load_ingest_config(config_path)
    db_url = config['ingestion']['deduplication']['index_db_url']
    
    # Initialize DB
    setup_storage(db_url)
    
    # 2. Validate
    try:
        img_meta = validate_image(image_path, config)
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        return {"status": "rejected", "error": str(e)}
    
    # 3. Compute Hash
    img_hash = compute_perceptual_hash(image_path)
    
    # 4. Check Duplicates
    existing_id = check_duplicate(img_hash, db_url)
    if existing_id:
        logger.info(f"Duplicate found. Returning existing ID: {existing_id}")
        # Fetch the existing record to get paths
        existing_record = get_existing_record(existing_id, db_url)
        return {
            "status": "duplicate",
            "image_id": existing_id,
            "message": "Image already exists in index",
            "record": existing_record if existing_record else None
        }
        
    # New ID
    new_image_id = str(uuid.uuid4())
    
    # 5. Generate Files
    paths = generate_staging_files(image_path, new_image_id, config)
    
    # 6. Register
    result = register_ingestion(
        image_id=new_image_id,
        file_hash=img_hash,
        metadata=img_meta,
        paths=paths,
        db_url=db_url,
        source_meta=source_metadata,
        task_type=task_type,
        dataset_id=dataset_id,
        ingestion_version=ingestion_version
    )
    
    logger.info(f"Ingestion complete for {new_image_id}")
    return result

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        ingest_image(sys.argv[1])

