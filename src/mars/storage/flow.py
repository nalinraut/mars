import logging
import json
from typing import Dict, Any

from prefect import task, flow, get_run_logger
from omegaconf import OmegaConf

from .archiver import SceneArchiver
from .db import update_index

@task
def load_storage_config(config_path: str = "config/storage.yaml") -> Dict[str, Any]:
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)

@task
def archive_assets(
    scene_graph: Dict[str, Any],
    quality_record: Dict[str, Any],
    image_id: str,
    config: Dict[str, Any]
) -> Dict[str, str]:
    """Move files and create final JSON."""
    archiver = SceneArchiver(config)
    return archiver.archive_scene(scene_graph, quality_record, image_id)

@task
def index_scene(
    image_id: str,
    quality_record: Dict[str, Any],
    storage_result: Dict[str, str],
    db_url: str
):
    """Update DB."""
    update_index(
        db_url, 
        image_id, 
        quality_record, 
        storage_result['local_path']
    )

@flow(name="Store Scene")
def store_scene(
    validation_result: Dict[str, Any],
    composition_result: Dict[str, Any],
    config_path: str = "config/storage.yaml"
) -> Dict[str, Any]:
    """
    Main flow for Stage 7: Storage.
    """
    logger = get_run_logger()
    
    image_id = validation_result.get("image_id")
    if not image_id or validation_result.get("status") != "validated":
         return {"status": "skipped", "reason": "validation_failed_or_missing"}

    # 1. Load Config
    config = load_storage_config(config_path)
    
    # 2. Load Scene Graph (with transforms from composition)
    # Validation result passes quality, but composition has the layout.
    # Actually, validation *reads* the scene graph but doesn't necessarily return the full object.
    # We need to reload it or pass it.
    scene_graph_path = composition_result.get("scene_path")
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)
        
    # 3. Archive
    storage_info = archive_assets(
        scene_graph, 
        validation_result.get("quality_record", {}),
        image_id,
        config
    )
    
    # 4. Index
    db_url = config['storage']['index']['db_url']
    index_scene(image_id, validation_result.get("quality_record", {}), storage_info, db_url)
    
    logger.info(f"Scene {image_id} successfully stored and indexed.")
    
    return {
        "status": "complete",
        "image_id": image_id,
        "final_path": storage_info['local_path'],
        "scene_json": storage_info['json_path']
    }

