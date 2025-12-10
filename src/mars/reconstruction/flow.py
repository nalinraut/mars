import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from prefect import task, flow, get_run_logger
from omegaconf import OmegaConf

from .base import ReconstructionModel
from .factory import get_reconstruction_model


def _assign_labels_to_masks(
    masks: List[Dict[str, Any]], 
    detections: List[Dict[str, Any]],
    logger
) -> List[Dict[str, Any]]:
    """
    Assign labels from detections to masks based on bounding box overlap.
    
    For each mask, find the detection whose bbox best contains the mask.
    This allows labeled object naming during reconstruction.
    """
    if not detections:
        return masks
    
    for mask in masks:
        mask_id = mask.get('mask_id', 'unknown')
        bbox = mask.get('bbox', [0, 0, 100, 100])
        
        if len(bbox) != 4:
            continue
        
        mx1, my1, mx2, my2 = [int(float(v)) for v in bbox]
        mask_area = (mx2 - mx1) * (my2 - my1)
        
        if mask_area <= 0:
            continue
        
        best_label = ''
        best_score = 0
        best_estimated_size = None
        
        for det in detections:
            label = det.get('label', '')
            det_bbox = det.get('bbox', [0, 0, 0, 0])
            dx1, dy1, dx2, dy2 = [float(x) for x in det_bbox[:4]]
            det_area = (dx2 - dx1) * (dy2 - dy1)
            
            if det_area <= 0:
                continue
            
            # Calculate intersection
            inter_x1 = max(mx1, dx1)
            inter_y1 = max(my1, dy1)
            inter_x2 = min(mx2, dx2)
            inter_y2 = min(my2, dy2)
            
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                containment = inter_area / mask_area
                fill_ratio = inter_area / det_area
                score = containment * (0.5 + 0.5 * min(fill_ratio * 2, 1.0))
                
                if score > best_score and containment > 0.3:
                    best_score = score
                    best_label = label
                    best_estimated_size = det.get('estimated_size')
        
        if best_label:
            mask['label'] = best_label
            if best_estimated_size:
                mask['estimated_size'] = best_estimated_size
            logger.debug(f"Assigned label '{best_label}' to {mask_id}")
    
    # Log summary
    labeled = sum(1 for m in masks if m.get('label'))
    logger.info(f"Assigned labels to {labeled}/{len(masks)} masks")
    
    return masks


# Define task to load config
@task
def load_reconstruction_config(config_path: str = "config/reconstruction.yaml") -> Dict[str, Any]:
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)

@task
def initialize_reconstructor(config: Dict[str, Any]) -> ReconstructionModel:
    """Factory task to create and load the reconstruction model."""
    logger = get_run_logger()
    model_name = config['reconstruction']['active_model']
    logger.info(f"Initializing reconstruction model: {model_name}")
    
    # Get model-specific config
    model_config = config['reconstruction']['models'][model_name]
    
    # Create model with its config
    model = get_reconstruction_model(model_name, model_config)
    
    # Load the model
    checkpoint = model_config.get('checkpoint_path')
    device = model_config.get('device', 'cuda')
    model.load_model(checkpoint_path=checkpoint, device=device)
    return model

def _vhacd_decomposition(
    mesh: 'trimesh.Trimesh', 
    logger,
    vhacd_config: Optional[Dict[str, Any]] = None
) -> Optional['trimesh.Trimesh']:
    """
    Perform V-HACD (Volumetric Hierarchical Approximate Convex Decomposition).
    
    V-HACD breaks a concave mesh into multiple convex pieces, giving accurate
    collision while maintaining fast physics simulation.
    
    Args:
        mesh: Input mesh to decompose
        logger: Logger instance
        vhacd_config: V-HACD parameters from config/reconstruction.yaml
    
    Returns:
        Combined mesh of all convex hulls, or None if failed
    """
    import tempfile
    import os
    
    # Default V-HACD parameters
    cfg = vhacd_config or {}
    resolution = cfg.get('resolution', 100000)
    depth = cfg.get('depth', 20)
    concavity = cfg.get('concavity', 0.0025)
    plane_ds = cfg.get('plane_downsampling', 4)
    hull_ds = cfg.get('hull_downsampling', 4)
    alpha = cfg.get('alpha', 0.05)
    beta = cfg.get('beta', 0.05)
    gamma = cfg.get('gamma', 0.00125)
    max_verts = cfg.get('max_vertices_per_hull', 64)
    min_vol = cfg.get('min_volume_per_hull', 0.0001)
    mode = cfg.get('mode', 0)
    
    try:
        import pybullet as p
        
        physics_client = p.connect(p.DIRECT)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_obj = os.path.join(tmp_dir, "input.obj")
            output_obj = os.path.join(tmp_dir, "output.obj")
            output_log = os.path.join(tmp_dir, "vhacd.log")
            
            mesh.export(input_obj)
            
            p.vhacd(
                input_obj,
                output_obj,
                output_log,
                resolution=resolution,
                depth=depth,
                concavity=concavity,
                planeDownsampling=plane_ds,
                convexhullDownsampling=hull_ds,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                pca=0,
                mode=mode,
                maxNumVerticesPerCH=max_verts,
                minVolumePerCH=min_vol
            )
            
            if os.path.exists(output_obj):
                import trimesh
                result = trimesh.load(output_obj)
                
                if isinstance(result, trimesh.Scene):
                    meshes = list(result.geometry.values())
                    if meshes:
                        combined = trimesh.util.concatenate(meshes)
                        logger.info(f"V-HACD: {len(meshes)} hulls, {len(combined.faces)} faces (res={resolution})")
                        p.disconnect(physics_client)
                        return combined
                elif isinstance(result, trimesh.Trimesh):
                    p.disconnect(physics_client)
                    return result
        
        p.disconnect(physics_client)
        return None
        
    except ImportError:
        logger.warning("PyBullet not available for V-HACD")
        return None
    except Exception as e:
        logger.warning(f"V-HACD failed: {e}")
        try:
            p.disconnect(physics_client)
        except:
            pass
        return None


@task
def generate_collision_mesh(
    visual_mesh_path: str,
    output_dir: Path,
    mask_id: str,
    method: str = "vhacd",
    collision_config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Generate simplified collision mesh from visual mesh.
    
    Args:
        visual_mesh_path: Path to the visual (detailed) mesh
        output_dir: Directory to save collision mesh
        mask_id: Object identifier
        method: 'vhacd' (default), 'convex_hull', or 'simplified'
        collision_config: Collision mesh config from reconstruction.yaml
    
    Returns:
        Path to collision mesh or None if failed
    """
    import trimesh
    logger = get_run_logger()
    
    cfg = collision_config or {}
    vhacd_cfg = cfg.get('vhacd', {})
    simplified_cfg = cfg.get('simplified', {})
    
    try:
        mesh = trimesh.load(visual_mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        collision_mesh = None
        method_used = method
        
        if method == "vhacd":
            # V-HACD decomposition - best for concave objects
            collision_mesh = _vhacd_decomposition(mesh, logger, vhacd_cfg)
            if collision_mesh is None:
                # Fallback to convex hull
                logger.info(f"V-HACD failed for {mask_id}, falling back to convex hull")
                collision_mesh = mesh.convex_hull
                method_used = "convex_hull (fallback)"
                
        elif method == "convex_hull":
            # Simple convex hull - fast, stable physics
            collision_mesh = mesh.convex_hull
            
        elif method == "simplified":
            # Simplified mesh (reduce triangles)
            target_faces = simplified_cfg.get('target_faces', 1000)
            target_faces = min(target_faces, len(mesh.faces))
            collision_mesh = mesh.simplify_quadric_decimation(target_faces)
            # Repair simplified mesh
            collision_mesh.remove_degenerate_faces()
            collision_mesh.merge_vertices()
            trimesh.repair.fix_normals(collision_mesh)
            if not collision_mesh.is_watertight:
                trimesh.repair.fill_holes(collision_mesh)
            # Final fallback to convex hull if not watertight
            ensure_watertight = simplified_cfg.get('ensure_watertight', True)
            if ensure_watertight and not collision_mesh.is_watertight:
                logger.info(f"Simplified mesh not watertight for {mask_id}, using convex hull")
                collision_mesh = collision_mesh.convex_hull
                method_used = "convex_hull (fallback)"
        else:
            collision_mesh = mesh.convex_hull
        
        # Ensure collision mesh is clean
        collision_mesh.remove_degenerate_faces()
        collision_mesh.merge_vertices()
        
        # Save collision mesh
        collision_path = output_dir / mask_id / "collision.obj"
        collision_path.parent.mkdir(parents=True, exist_ok=True)
        collision_mesh.export(str(collision_path))
        
        logger.info(f"Generated collision mesh for {mask_id} ({method_used}): {len(collision_mesh.faces)} faces")
        return str(collision_path)
        
    except Exception as e:
        logger.warning(f"Failed to generate collision mesh for {mask_id}: {e}")
        return None


@task
def reconstruct_object(
    reconstructor: ReconstructionModel,
    image_path: str,
    mask_data: Dict[str, Any],
    output_dir: str,
    image_id: str,
    generate_collision: bool = True,
    collision_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Reconstruct a single object and generate collision mesh.
    
    Args:
        reconstructor: Loaded reconstruction model
        image_path: Path to source image
        mask_data: Mask metadata (mask_id, label, bbox, etc.)
        output_dir: Base output directory
        image_id: Scene/image identifier
        generate_collision: Whether to generate collision mesh
        collision_config: Collision mesh config (method, vhacd params, etc.)
    """
    logger = get_run_logger()
    mask_id = mask_data.get('mask_id')
    label = mask_data.get('label', '')
    
    # Create labeled object ID: {label}_{mask_id} or just mask_id if no label
    if label:
        # Sanitize label for filesystem
        safe_label = label.lower().replace(' ', '_').replace('-', '_')
        object_id = f"{safe_label}_{mask_id}"
    else:
        object_id = mask_id
    
    # Prepare metadata needed by the model
    metadata = {
        "image_id": image_id,
        "mask_id": mask_id,
        "object_id": object_id,  # New: labeled object ID
        "label": label,
        "mask_path": mask_data.get("mask_path"),
        "bbox": mask_data.get("bbox"),
        "category": mask_data.get("category")
    }
    
    # Define object-specific output location
    # Structure: data/processed/{image_id}/objects/
    scene_obj_dir = Path(output_dir) / image_id / "objects"
    scene_obj_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        result = reconstructor.reconstruct(
            image_path=image_path,
            output_dir=scene_obj_dir,
            metadata=metadata
        )
        
        # Add IDs and label back to result
        result['mask_id'] = mask_id
        result['object_id'] = object_id  # Labeled ID for display/export
        result['label'] = label
        result['status'] = "success"
        
        # IMPORTANT: Preserve segmentation metadata for composition (bbox, centroid, area)
        # Convert to native Python types to avoid JSON serialization issues
        bbox = mask_data.get('bbox')
        if bbox:
            result['bbox'] = [int(float(x)) for x in bbox]  # Handle string floats
        centroid = mask_data.get('centroid')
        if centroid:
            result['centroid'] = [int(float(x)) for x in centroid]  # Handle string floats
        area = mask_data.get('area')
        if area:
            result['area'] = int(area)
        result['category'] = mask_data.get('category')
        
        # Generate collision mesh for physics simulation
        if generate_collision and result.get('mesh_path'):
            cfg = collision_config or {}
            collision_path = generate_collision_mesh(
                visual_mesh_path=result['mesh_path'],
                output_dir=scene_obj_dir,
                mask_id=object_id,
                method=cfg.get('method', 'vhacd'),
                collision_config=cfg
            )
            if collision_path:
                result['collision_path'] = collision_path
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to reconstruct object {object_id}: {e}")
        return {
            "mask_id": mask_id,
            "status": "failed",
            "error": str(e)
        }

@flow(name="Reconstruct Scene 3D", retries=1)
def reconstruct_scene(
    segmentation_result: Dict[str, Any],
    image_path: str,
    config_path: str = "config/reconstruction.yaml",
    output_dir: str = "data/processed",
    detections: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main flow for Stage 3: 3D Reconstruction.
    Iterates over all masks from Stage 2.
    
    Args:
        detections: Detection results for label assignment to masks
    """
    logger = get_run_logger()
    
    image_id = segmentation_result.get("image_id")
    if not image_id or segmentation_result.get("status") != "success":
        logger.warning("Skipping reconstruction: Invalid segmentation result")
        return {"status": "skipped", "reason": "segmentation_failed"}
        
    masks = segmentation_result.get("masks", [])
    logger.info(f"Starting reconstruction for {len(masks)} objects in scene {image_id}")
    
    # Assign labels from detections to masks (based on bbox overlap)
    if detections:
        masks = _assign_labels_to_masks(masks, detections, logger)
    
    # 1. Load Config & Model
    config = load_reconstruction_config(config_path)
    reconstructor = initialize_reconstructor(config)
    
    # Get collision config from reconstruction config
    collision_config = config.get('reconstruction', {}).get('collision', {})
    
    # 2. Batch Process Objects
    reconstruction_results = []
    
    for mask in masks:
        # Skip if marked as ignore (e.g. too small, background)
        if mask.get("object_type") == "ignore":
            continue
            
        res = reconstruct_object(
            reconstructor=reconstructor,
            image_path=image_path,
            mask_data=mask,
            output_dir=output_dir,
            image_id=image_id,
            collision_config=collision_config
        )
        reconstruction_results.append(res)
        
    # 3. Aggregate Results
    successful = [r for r in reconstruction_results if r['status'] == "success"]
    failed = [r for r in reconstruction_results if r['status'] == "failed"]
    
    quality_score = 0.0
    if successful:
        # Simple quality metric: % of watertight meshes
        watertight_count = sum(1 for r in successful if r.get('is_watertight', False))
        quality_score = watertight_count / len(successful)
    
    logger.info(f"Reconstruction complete. Success: {len(successful)}, Failed: {len(failed)}")
    
    return {
        "status": "success" if successful else "failed",
        "image_id": image_id,
        "objects": reconstruction_results,
        "image_size": segmentation_result.get("image_size"),  # Propagate for composition
        "metrics": {
            "total_objects": len(masks),
            "reconstructed": len(successful),
            "failed": len(failed),
            "quality_score": quality_score
        }
    }

if __name__ == "__main__":
    # Test stub
    pass

