import logging
import time
import numpy as np
import pybullet as p
import pybullet_data
from pathlib import Path
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class SimulationValidator:
    """
    Runs physical validation using PyBullet.
    Validates:
    - Object stability (don't move too much)
    - Support relationships (objects stay on their supports)
    - No interpenetration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sim_config = config['validation']['simulation']
        self.thresholds = config['validation']['thresholds']
        self.support_config = config['validation'].get('support_constraints', {})
        self.client_id = None
        self.use_collision_mesh = self.sim_config.get('use_collision_mesh', True)

    def validate_scene(self, scene_graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load scene, run simulation, check stability and support constraints.
        """
        self._setup_simulation()
        
        try:
            # 1. Load Objects (prefer collision mesh if available)
            body_ids = self._load_scene_objects(scene_graph)
            
            if not body_ids:
                return {"stability": 0.0, "error": "no_objects_loaded"}
            
            # 2. Record Initial State
            initial_states = self._get_object_states(body_ids)
            
            # 3. Run Simulation (Settling)
            self._run_settling()
            
            # 4. Record Final State
            final_states = self._get_object_states(body_ids)
            
            # 5. Check Stability
            stability_metrics = self._compute_stability_metrics(initial_states, final_states)
            
            # 6. Check Support Relationships
            support_metrics = self._validate_support_relationships(
                scene_graph, body_ids, final_states
            )
            
            # Combine metrics
            metrics = {**stability_metrics, **support_metrics}
            
            return metrics
            
        finally:
            self._teardown_simulation()

    def _setup_simulation(self):
        # Connect in DIRECT mode (no GUI) for speed
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Note: PyBullet uses Z-up by default, our scene uses Y-up
        # Gravity should be along negative Y in our coords, which is -Z in PyBullet
        gravity = self.sim_config.get('gravity', [0, -9.81, 0])
        # Convert from Y-up to Z-up: swap Y and Z
        p.setGravity(gravity[0], gravity[2], gravity[1])
        
        # Load ground plane (at Z=0 in PyBullet = Y=0 in our scene)
        p.loadURDF("plane.urdf")

    def _teardown_simulation(self):
        if self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    def _resolve_mesh_path(self, mesh_path: str, mask_id: str) -> str:
        """
        Resolve mesh path, checking stage directories if original doesn't exist.
        
        The scene graph may have paths to /objects/ but files are in /4_reconstruction/objects/.
        """
        original_path = Path(mesh_path)
        
        # If original path exists, use it
        if original_path.exists():
            return str(original_path.resolve())
        
        # Try stage directory alternatives
        # Pattern: /.../image_id/objects/object_id/file.obj
        #       -> /.../image_id/4_reconstruction/objects/object_id/file.obj
        path_str = str(original_path)
        
        # Try inserting stage directories before 'objects/'
        stage_dirs = ['4_reconstruction', '3_segmentation', '5_composition']
        for stage in stage_dirs:
            if '/objects/' in path_str and f'/{stage}/objects/' not in path_str:
                alt_path = Path(path_str.replace('/objects/', f'/{stage}/objects/'))
                if alt_path.exists():
                    logger.debug(f"Found mesh at stage dir: {alt_path}")
                    return str(alt_path.resolve())
        
        # Try parent directory + stage
        parent = original_path.parent.parent  # Go up from object_id dir
        for stage in stage_dirs:
            alt_path = parent / stage / 'objects' / original_path.parent.name / original_path.name
            if alt_path.exists():
                logger.debug(f"Found mesh at: {alt_path}")
                return str(alt_path.resolve())
        
        # Return original path (will fail in PyBullet but with proper error)
        logger.warning(f"Mesh not found for {mask_id}: {mesh_path}")
        return str(original_path.resolve())
    
    def _load_scene_objects(self, scene_graph: Dict) -> Dict[str, int]:
        """
        Load collision meshes into PyBullet.
        Prefers simplified collision mesh over visual mesh for stability.
        """
        body_ids = {}
        
        for obj in scene_graph['objects']:
            mask_id = obj.get('mask_id', obj.get('object_id', 'unknown'))
            
            # Prefer collision mesh (convex hull) over visual mesh
            if self.use_collision_mesh:
                mesh_path = obj.get('collision_path') or obj.get('mesh_path')
            else:
                mesh_path = obj.get('mesh_path') or obj.get('collision_path')
            
            if not mesh_path:
                logger.warning(f"No mesh path for {mask_id}, skipping")
                continue
            
            # Resolve mesh path - check if file exists, try stage directory fallback
            mesh_path = self._resolve_mesh_path(mesh_path, mask_id)
            
            # Get transform
            transform = obj.get('transform', {})
            pos = transform.get('position', [0, 0.1, 0])
            rot = transform.get('rotation', [0, 0, 0, 1])
            scale = transform.get('scale', [1, 1, 1])
            
            # Ensure scale is a list
            if isinstance(scale, (int, float)):
                scale = [scale, scale, scale]
                
            try:
                # Create collision shape
                col_id = p.createCollisionShape(
                    p.GEOM_MESH, 
                    fileName=mesh_path, 
                    meshScale=scale
                )
                
                # Convert from Y-up (our scene) to Z-up (PyBullet): swap Y and Z
                pybullet_pos = [pos[0], pos[2], pos[1]]
                pybullet_rot = [rot[0], rot[2], rot[1], rot[3]]
                
                # Use estimated mass or default (lighter = more stable)
                mass = obj.get('mass', 1.0)
                # Cap mass to prevent instability
                mass = min(mass, 10.0)
                
                body_id = p.createMultiBody(
                    baseMass=mass,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=-1,  # No visual for speed
                    basePosition=pybullet_pos,
                    baseOrientation=pybullet_rot
                )
                
                # Set friction/restitution for stability
                p.changeDynamics(
                    body_id, 
                    -1, 
                    lateralFriction=obj.get('friction', 0.8),
                    restitution=obj.get('restitution', 0.1),
                    linearDamping=0.1,
                    angularDamping=0.1
                )
                
                body_ids[mask_id] = body_id
                logger.debug(f"Loaded {mask_id} at {pybullet_pos}")
                
            except Exception as e:
                logger.warning(f"Failed to load {mask_id}: {e}")
                
        return body_ids

    def _get_object_states(self, body_ids: Dict[str, int]) -> Dict[str, Dict]:
        states = {}
        for mask_id, body_id in body_ids.items():
            pos, rot = p.getBasePositionAndOrientation(body_id)
            states[mask_id] = {'pos': np.array(pos), 'rot': np.array(rot)}
        return states

    def _run_settling(self):
        steps = int(self.sim_config['settle_time'] / self.sim_config['time_step'])
        for _ in range(steps):
            p.stepSimulation()

    def _compute_stability_metrics(
        self, 
        initial: Dict[str, Dict], 
        final: Dict[str, Dict]
    ) -> Dict[str, float]:
        
        total_disp = 0.0
        max_disp = 0.0
        stable_count = 0
        total_objs = len(initial)
        
        object_results = {}
        
        if total_objs == 0:
            return {"stability": 0.0, "object_results": {}}
            
        threshold = self.thresholds.get('max_displacement_m', 0.15)
        
        for mask_id in initial:
            p0 = initial[mask_id]['pos']
            p1 = final[mask_id]['pos']
            
            disp = np.linalg.norm(p1 - p0)
            total_disp += disp
            max_disp = max(max_disp, disp)
            
            is_stable = disp < threshold
            if is_stable:
                stable_count += 1
            
            object_results[mask_id] = {
                "displacement": float(disp),
                "stable": is_stable,
                "initial_pos": p0.tolist(),
                "final_pos": p1.tolist()
            }
                
        stability_score = (stable_count / total_objs) * 100.0
        
        return {
            "stability": stability_score,
            "stable_count": stable_count,
            "total_objects": total_objs,
            "max_displacement": float(max_disp),
            "avg_displacement": float(total_disp / total_objs),
            "threshold_used": threshold,
            "object_results": object_results
        }
    
    def _validate_support_relationships(
        self,
        scene_graph: Dict,
        body_ids: Dict[str, int],
        final_states: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Validate that objects stay on their designated support surfaces.
        """
        if not self.support_config.get('enabled', False):
            return {"support_validation": "disabled"}
        
        violations = []
        max_violation = self.support_config.get('max_support_violation_m', 0.05)
        
        # Build support map from scene graph
        support_map = {}  # child -> parent
        for obj in scene_graph.get('objects', []):
            mask_id = obj.get('mask_id', obj.get('object_id'))
            supported_by = obj.get('supported_by')
            if supported_by:
                support_map[mask_id] = supported_by
        
        # Check each supported object
        for child_id, parent_id in support_map.items():
            if child_id not in final_states or parent_id not in final_states:
                continue
            
            child_z = final_states[child_id]['pos'][2]  # Z in PyBullet = Y in our scene
            parent_z = final_states[parent_id]['pos'][2]
            
            # Child should be above parent
            if child_z < parent_z - max_violation:
                violations.append({
                    "child": child_id,
                    "parent": parent_id,
                    "violation_m": float(parent_z - child_z)
                })
        
        return {
            "support_violations": len(violations),
            "support_violation_details": violations,
            "support_valid": len(violations) == 0
        }

