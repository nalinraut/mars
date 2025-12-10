import logging
import json
from pathlib import Path
from typing import Dict, Any, List

from prefect import task, flow, get_run_logger
from omegaconf import OmegaConf

from .estimator import PhysicsEstimator

@task
def load_physics_config(config_path: str = "config/physics.yaml") -> Dict[str, Any]:
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)

@task
def estimate_object_physics(
    estimator: PhysicsEstimator,
    obj_data: Dict[str, Any],
    output_dir: str,
    image_id: str
) -> Dict[str, Any]:
    """
    Compute physics for a single object and save to disk.
    """
    logger = get_run_logger()
    mask_id = obj_data.get('mask_id')
    object_id = obj_data.get('object_id', mask_id)  # Use labeled ID if available
    mesh_path = obj_data.get('mesh_path') # Visual mesh from Stage 3
    collision_path = obj_data.get('collision_path') # Optimized mesh (preferable for mass calc if valid)
    
    # Use collision mesh for physics calc if available, else visual
    target_mesh = collision_path if collision_path else mesh_path
    
    if not target_mesh or not Path(target_mesh).exists():
        logger.warning(f"No mesh found for {object_id}, skipping physics.")
        return {
            "mask_id": mask_id,
            "object_id": object_id,
            "status": "failed",
            "error": "mesh_missing"
        }

    category = obj_data.get('category', 'unknown')
    label = obj_data.get('label', '')
    
    try:
        props = estimator.estimate_properties(
            mesh_path=target_mesh,
            category=category
        )
        
        # Save physics bundle
        # Structure: data/processed/{image_id}/objects/{object_id}/physics.json
        obj_dir = Path(output_dir) / image_id / "objects" / object_id
        obj_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = obj_dir / "physics.json"
        with open(out_path, 'w') as f:
            json.dump(props, f, indent=2)
            
        return {
            "mask_id": mask_id,
            "object_id": object_id,
            "label": label,
            "status": "success",
            "physics_path": str(out_path),
            "mass": props['mass'],
            "friction": props['friction'],
            "restitution": props['restitution'],
            "density": props['density'],
            "center_of_mass": props['center_of_mass'],
            "material": props['material']
        }
        
    except Exception as e:
        logger.error(f"Physics estimation failed for {object_id}: {e}")
        return {
            "mask_id": mask_id,
            "object_id": object_id,
            "status": "failed",
            "error": str(e)
        }

@flow(name="Estimate Scene Physics")
def estimate_physics(
    reconstruction_result: Dict[str, Any],
    config_path: str = "config/physics.yaml",
    output_dir: str = "data/processed"
) -> Dict[str, Any]:
    """
    Main flow for Stage 4: Physics Estimation.
    """
    logger = get_run_logger()
    
    image_id = reconstruction_result.get("image_id")
    if not image_id or reconstruction_result.get("status") != "success":
        logger.warning("Skipping physics: Invalid reconstruction result")
        return {"status": "skipped", "reason": "reconstruction_failed"}
        
    objects = reconstruction_result.get("objects", [])
    logger.info(f"Estimating physics for {len(objects)} objects...")
    
    # 1. Load Config & Estimator
    config = load_physics_config(config_path)
    estimator = PhysicsEstimator(config)
    
    # 2. Batch Process - preserve original object metadata
    physics_results = []
    for obj in objects:
        if obj.get("status") != "success":
            continue
        
        # Get physics properties
        res = estimate_object_physics(
            estimator=estimator,
            obj_data=obj,
            output_dir=output_dir,
            image_id=image_id
        )
        
        # IMPORTANT: Merge physics result with original object metadata
        # This preserves bbox, area, centroid, mesh_path, etc. for composition
        merged = obj.copy()  # Start with all original fields
        merged.update(res)   # Add physics-specific fields
        physics_results.append(merged)
        
    # 3. Aggregate
    successful = [r for r in physics_results if r['status'] == "success"]
    
    return {
        "status": "success",
        "image_id": image_id,
        "objects": physics_results,
        "image_size": reconstruction_result.get("image_size"),  # Propagate for composition
        "metrics": {
            "total": len(objects),
            "estimated": len(successful)
        }
    }

