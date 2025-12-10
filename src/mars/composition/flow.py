import logging
from typing import Dict, Any, List

from prefect import task, flow, get_run_logger
from omegaconf import OmegaConf

from .layout import LayoutEngine
from .scene_graph import SceneGraphBuilder

@task
def load_composition_config(config_path: str = "config/composition.yaml") -> Dict[str, Any]:
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)

@task
def arrange_objects(
    objects: List[Dict[str, Any]],
    config: Dict[str, Any],
    image_size: tuple = (1024, 1024),
    detections: List[Dict[str, Any]] = None,
    image_path: str = None
) -> List[Dict[str, Any]]:
    """
    Compute 3D transforms for all objects.
    Applies real-world scaling based on detected labels.
    Uses MoGe depth estimation (if available) for accurate Z positioning.
    """
    logger = get_run_logger()
    logger.info(f"arrange_objects received {len(objects)} objects, {len(detections) if detections else 0} detections")
    if objects:
        o = objects[0]
        logger.info(f"First object: {o.get('mask_id')}, area={o.get('area')}, bbox={str(o.get('bbox'))[:25] if o.get('bbox') else None}")
    layout_engine = LayoutEngine(config)
    
    # 1. Project to 3D with scaling and depth
    placed_objects = layout_engine.project_to_3d(
        objects, 
        image_size, 
        config['composition']['camera'],
        detections=detections,
        image_path=image_path
    )
    
    # 2. Resolve Overlaps
    resolved_objects = layout_engine.resolve_collisions(placed_objects)
    
    logger.info(f"Arranged {len(resolved_objects)} objects in scene.")
    return resolved_objects

@task
def build_and_save_scene(
    scene_id: str,
    objects: List[Dict[str, Any]],
    config: Dict[str, Any],
    output_dir: str,
    mesh_dir: str = None,
    export_usd: bool = True
) -> Dict[str, Any]:
    """
    Construct scene graph and save to disk (JSON and optionally USD).
    """
    builder = SceneGraphBuilder(config)
    
    scene_graph = builder.build(scene_id, objects)
    
    # Save returns dict with 'json' and 'usd' paths
    save_result = builder.save(
        scene_graph, 
        output_dir, 
        export_usd=export_usd,
        mesh_dir=mesh_dir
    )
    
    return {
        "scene_graph_path": save_result.get('json'),
        "usd_path": save_result.get('usd'),
        "object_count": len(objects),
        "zones": [z['name'] for z in scene_graph['zones']]
    }

@flow(name="Compose Scene")
def compose_scene(
    physics_result: Dict[str, Any],
    config_path: str = "config/composition.yaml",
    output_dir: str = "data/processed",
    export_usd: bool = True
) -> Dict[str, Any]:
    """
    Main flow for Stage 5: Scene Composition.
    
    Handles:
    - 3D layout from 2D segmentation data
    - Real-world scaling based on detected labels
    - Collision detection and resolution
    - Support relationship detection (stacking)
    - Scene graph generation
    - USD export (optional)
    """
    logger = get_run_logger()
    
    image_id = physics_result.get("image_id")
    if not image_id or physics_result.get("status") != "success":
        logger.warning("Skipping composition: Invalid physics result")
        return {"status": "skipped", "reason": "physics_failed"}
    
    objects = physics_result.get("objects", [])
    
    # Merge segmentation metadata (area, bbox, centroid) into objects if missing
    # This ensures composition has the data even if upstream flows lost it
    segmentation = physics_result.get("segmentation", {})
    seg_masks = segmentation.get("masks", [])
    
    # DEBUG: Log first object's data
    if objects:
        o = objects[0]
        logger.info(f"First object before merge: {o.get('mask_id')}, area={o.get('area')}, bbox={str(o.get('bbox'))[:20] if o.get('bbox') else None}")
    
    logger.info(f"Segmentation data: {len(seg_masks)} masks available, {len(objects)} objects to merge")
    
    if seg_masks:
        seg_lookup = {m.get("mask_id"): m for m in seg_masks}
        merged_count = 0
        for obj in objects:
            mask_id = obj.get("mask_id")
            if mask_id and mask_id in seg_lookup:
                seg_data = seg_lookup[mask_id]
                # Always merge (overwrite) to ensure correct data
                obj["area"] = seg_data.get("area")
                obj["bbox"] = seg_data.get("bbox")
                obj["centroid"] = seg_data.get("centroid")
                merged_count += 1
        logger.info(f"Merged segmentation metadata for {merged_count} objects")
        
        # DEBUG: Log first object after merge
        if objects:
            o = objects[0]
            logger.info(f"First object after merge: {o.get('mask_id')}, area={o.get('area')}, bbox={str(o.get('bbox'))[:20] if o.get('bbox') else None}")
    
    # Get image size from metadata if available
    image_size = physics_result.get("image_size", (1024, 1024))
    
    # Extract detections from physics_result (embedded by pipeline)
    detections = physics_result.get("detections", [])
    logger.info(f"Using {len(detections)} detections for label-based scaling")
    
    # Get image path for MoGe depth estimation
    image_path = physics_result.get("image_path")
    
    # 1. Load Config
    config = load_composition_config(config_path)
    
    # 2. Arrange Layout with collision detection, scaling, and MoGe depth
    arranged_objects = arrange_objects(
        objects, config, image_size, 
        detections=detections,
        image_path=image_path
    )
    
    # 3. Determine mesh directory for USD export
    mesh_dir = f"{output_dir}/{image_id}/objects"
    
    # 4. Build Scene Graph and Export
    result = build_and_save_scene(
        scene_id=image_id,
        objects=arranged_objects,
        config=config,
        output_dir=output_dir,
        mesh_dir=mesh_dir,
        export_usd=export_usd
    )
    
    return {
        "status": "success",
        "image_id": image_id,
        "scene_path": result["scene_graph_path"],
        "usd_path": result.get("usd_path"),
        "details": result,
        "objects": arranged_objects  # Include positioned objects for downstream
    }

