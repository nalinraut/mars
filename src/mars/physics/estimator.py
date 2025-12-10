import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import trimesh
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class PhysicsProperties:
    mass: float
    density: float
    friction: float
    restitution: float
    inertia_tensor: List[List[float]]
    center_of_mass: List[float]
    material: str
    # Randomization ranges [min, max]
    mass_range: List[float]
    friction_range: List[float]
    restitution_range: List[float]

class PhysicsEstimator:
    """
    Estimates physical properties based on mesh geometry and material classification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.materials = config['physics']['materials']
        self.category_map = config['physics']['category_map']
        self.default_material = config['physics']['default_material']
        self.rand_config = config['physics']['randomization']
        # Scale factors to convert SAM 3D mesh volumes to real-world scale
        self.category_scale = config['physics'].get('category_scale', {})

    def estimate_properties(
        self, 
        mesh_path: str, 
        category: str,
        image_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute physics properties for a single object.
        """
        # 1. Determine Material
        material = self._classify_material(category, image_path)
        mat_props = self.materials.get(material, self.materials[self.default_material])
        
        # 2. Load Mesh & Compute Geometry
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            if not mesh.is_watertight:
                # Fallback: use convex hull for mass calculation if original is bad
                mesh = mesh.convex_hull
                
            volume = mesh.volume
            if volume <= 1e-6: # Safety for zero volume/degenerate meshes
                volume = 0.0001
                logger.warning(f"Mesh volume too small for {mesh_path}, using epsilon.")
                
            # Center of Mass & Inertia
            com = mesh.center_mass.tolist()
            # Trimesh computes moment of inertia assuming density=1. We scale it later.
            inertia_tensor_density_1 = mesh.moment_inertia 
            
        except Exception as e:
            logger.error(f"Failed to process mesh for physics {mesh_path}: {e}")
            # Fallback to simple box approximation if mesh fails
            return self._get_fallback_physics()

        # 3. Calculate Mass & Inertia
        density = mat_props['density']
        
        # Apply scale factor to convert mesh volume to real-world volume
        # SAM 3D Objects outputs meshes in arbitrary units
        scale_factor = self.category_scale.get(
            category.lower(), 
            self.category_scale.get('default', 0.0001)
        )
        real_volume = volume * scale_factor
        mass = real_volume * density
        
        # Ensure realistic mass bounds
        mass = max(0.01, min(mass, 100.0))  # Between 10g and 100kg
        
        # Scale inertia tensor by density and scale factor
        inertia_tensor = (inertia_tensor_density_1 * density * scale_factor).tolist()
        
        # 4. Define Ranges (for Domain Randomization)
        mass_scale = self.rand_config['mass_scale']
        fric_scale = self.rand_config['friction_scale']
        rest_scale = self.rand_config['restitution_scale']
        
        props = PhysicsProperties(
            mass=mass,
            density=density,
            friction=mat_props['friction'],
            restitution=mat_props['restitution'],
            inertia_tensor=inertia_tensor,
            center_of_mass=com,
            material=material,
            mass_range=[mass * (1 - mass_scale), mass * (1 + mass_scale)],
            friction_range=[mat_props['friction'] * (1 - fric_scale), mat_props['friction'] * (1 + fric_scale)],
            restitution_range=[mat_props['restitution'] * (1 - rest_scale), mat_props['restitution'] * (1 + rest_scale)]
        )
        
        return asdict(props)

    def _classify_material(self, category: str, image_path: Optional[str]) -> str:
        """
        Determine material type.
        TODO: Integrate CLIP for visual material estimation (e.g., "is this table wood or metal?")
        For now, uses category mapping + defaults.
        """
        # Simple lookup
        if category in self.category_map:
            return self.category_map[category]
            
        return self.default_material

    def _get_fallback_physics(self) -> Dict[str, Any]:
        """Return safe default values if calculation crashes."""
        return asdict(PhysicsProperties(
            mass=0.1,
            density=100.0,
            friction=0.5,
            restitution=0.5,
            inertia_tensor=np.eye(3).tolist(),
            center_of_mass=[0, 0, 0],
            material="unknown",
            mass_range=[0.05, 0.15],
            friction_range=[0.4, 0.6],
            restitution_range=[0.4, 0.6]
        ))

