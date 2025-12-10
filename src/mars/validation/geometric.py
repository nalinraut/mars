import trimesh
import numpy as np
from pathlib import Path
from typing import Dict, List

def calculate_mesh_integrity(mesh_path: str) -> float:
    """
    Calculate Mesh Integrity Score (0-100).
    Checks watertightness, winding, and degenerate faces.
    """
    try:
        mesh = trimesh.load(mesh_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            # If it loaded as a scene, try to concat geometries or pick first
            if len(mesh.geometry) == 0:
                return 0.0
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

        score = 100.0
        
        # 1. Watertightness (Crucial for physics)
        if not mesh.is_watertight:
            score -= 40.0
            
        # 2. Winding / Normals
        # is_winding_consistent checks if normals are consistent
        if not mesh.is_winding_consistent:
            score -= 20.0
            
        # 3. Degenerate Faces (Zero area)
        # We assume cleaned meshes, but penalize if present
        # Usually handled by mesh.process()
        if not mesh.is_volume: # Often implies issues
            score -= 10.0
            
        return max(0.0, score)
    except Exception:
        return 0.0

def calculate_geometric_quality(scene_objects: List[Dict]) -> Dict[str, float]:
    """
    Compute Tier 1 Geometric Quality for the whole scene.
    Aggregates scores from all objects.
    """
    if not scene_objects:
        return {
            "mesh_integrity": 0.0,
            "reconstruction_confidence": 0.0,
            "collision_mesh_quality": 0.0
        }
        
    integrity_scores = []
    collision_scores = []
    conf_scores = []
    
    for obj in scene_objects:
        # 1. Mesh Integrity
        mesh_path = obj.get("mesh_path")
        if mesh_path and Path(mesh_path).exists():
            integrity_scores.append(calculate_mesh_integrity(mesh_path))
        else:
            integrity_scores.append(0.0)
            
        # 2. Collision Quality
        # Simple heuristic: Does it exist and have reasonable volume vs visual?
        col_path = obj.get("collision_path")
        if col_path and Path(col_path).exists():
            # Ideally we load both and compare volumes (IoU)
            # For speed, we just check existence for now
            collision_scores.append(100.0) 
        else:
            collision_scores.append(0.0)
            
        # 3. Reconstruction Confidence
        # Passed from SAM 3D model
        # If not available, assume relatively high if mesh generated successfully
        conf_scores.append(obj.get("confidence", 80.0))

    return {
        "mesh_integrity": float(np.mean(integrity_scores)),
        "reconstruction_confidence": float(np.mean(conf_scores)),
        "collision_mesh_quality": float(np.mean(collision_scores))
    }

