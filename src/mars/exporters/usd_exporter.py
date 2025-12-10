"""
USD (Universal Scene Description) Exporter

Exports scene graphs to USD format for use in:
- NVIDIA Isaac Sim
- NVIDIA Omniverse
- Blender
- Pixar tools
- Unity/Unreal (via USD plugins)
"""
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np

logger = logging.getLogger(__name__)

# Try to import USD libraries
try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    logger.warning("USD libraries (pxr) not available. Install with: pip install usd-core")


class USDExporter:
    """
    Export MARS scenes to USD format.
    
    Supports:
    - Mesh geometry (from OBJ files)
    - Physics properties (mass, friction, restitution)
    - Materials (basic PBR)
    - Scene hierarchy
    - Camera and lighting
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.meters_per_unit = self.config.get('meters_per_unit', 1.0)
        self.up_axis = self.config.get('up_axis', 'Y')
    
    def _sanitize_prim_name(self, name: str) -> str:
        """
        Sanitize a name to be a valid USD prim name.
        USD prim names must:
        - Not start with a number
        - Only contain alphanumeric characters and underscores
        """
        import re
        # Replace invalid characters with underscores
        safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a number
        if safe_name and safe_name[0].isdigit():
            safe_name = f"scene_{safe_name}"
        # Ensure it's not empty
        if not safe_name:
            safe_name = "scene"
        return safe_name
        
    def export(
        self,
        scene_graph: Dict[str, Any],
        output_path: str,
        mesh_dir: Optional[str] = None,
        include_physics: bool = True,
        format: str = 'usda'  # 'usda' (ASCII) or 'usdc' (binary)
    ) -> str:
        """
        Export scene graph to USD file.
        
        Args:
            scene_graph: Scene graph dictionary from SceneGraphBuilder
            output_path: Output directory for USD file
            mesh_dir: Directory containing OBJ mesh files
            include_physics: Whether to add physics properties
            format: 'usda' (human-readable) or 'usdc' (binary, smaller)
            
        Returns:
            Path to the exported USD file
        """
        if not USD_AVAILABLE:
            logger.error("USD libraries not available. Cannot export.")
            return self._export_fallback(scene_graph, output_path)
        
        scene_id = scene_graph.get('scene_id', 'scene')
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        usd_path = output_dir / f"{scene_id}.{format}"
        
        # Create USD stage
        stage = Usd.Stage.CreateNew(str(usd_path))
        
        # Set stage metadata
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if self.up_axis == 'Y' else UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, self.meters_per_unit)
        
        # Sanitize scene_id for USD prim path (no hyphens, can't start with number)
        safe_scene_id = self._sanitize_prim_name(scene_id)
        
        # Create root xform
        root_path = f"/{safe_scene_id}"
        root_xform = UsdGeom.Xform.Define(stage, root_path)
        stage.SetDefaultPrim(root_xform.GetPrim())
        
        # Add objects
        objects_path = f"{root_path}/objects"
        UsdGeom.Xform.Define(stage, objects_path)
        
        for obj in scene_graph.get('objects', []):
            self._add_object(stage, objects_path, obj, mesh_dir, include_physics)
        
        # Add zones
        zones_path = f"{root_path}/zones"
        UsdGeom.Xform.Define(stage, zones_path)
        
        for zone in scene_graph.get('zones', []):
            self._add_zone(stage, zones_path, zone)
        
        # Add camera
        camera_config = scene_graph.get('camera', {})
        if camera_config:
            self._add_camera(stage, root_path, camera_config)
        
        # Add lighting
        lighting_config = scene_graph.get('lighting', {})
        if lighting_config:
            self._add_lighting(stage, root_path, lighting_config)
        
        # Add metadata as custom attributes
        metadata = scene_graph.get('metadata', {})
        if metadata:
            root_prim = stage.GetPrimAtPath(root_path)
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    attr_name = f"mars:{key}"
                    root_prim.CreateAttribute(attr_name, Sdf.ValueTypeNames.String).Set(str(value))
        
        # Save
        stage.Save()
        logger.info(f"Exported USD scene to: {usd_path}")
        
        return str(usd_path)
    
    def _add_object(
        self,
        stage: 'Usd.Stage',
        parent_path: str,
        obj: Dict,
        mesh_dir: Optional[str],
        include_physics: bool
    ):
        """Add an object to the USD stage."""
        obj_id = obj.get('object_id', obj.get('mask_id', 'object'))
        safe_obj_id = self._sanitize_prim_name(obj_id)
        obj_path = f"{parent_path}/{safe_obj_id}"
        
        # Create xform for the object
        xform = UsdGeom.Xform.Define(stage, obj_path)
        
        # Set transform - handle nested 'transform' dict or top-level fields
        transform_data = obj.get('transform', {})
        position = transform_data.get('position', obj.get('position', [0, 0, 0]))
        rotation = transform_data.get('rotation', obj.get('rotation', [0, 0, 0, 1]))  # quaternion [x, y, z, w]
        scale = transform_data.get('scale', obj.get('scale', [1, 1, 1]))
        
        # Apply transform ops
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        if rotation != [0, 0, 0, 1]:
            # Convert quaternion to axis-angle or use quaternion directly
            xform.AddOrientOp().Set(Gf.Quatf(rotation[3], rotation[0], rotation[1], rotation[2]))
        if scale != [1, 1, 1]:
            xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        
        # Reference mesh if available
        mesh_path_str = obj.get('mesh_path')
        if mesh_path_str and Path(mesh_path_str).exists():
            # Create mesh geometry
            mesh_prim_path = f"{obj_path}/mesh"
            self._import_obj_as_mesh(stage, mesh_prim_path, mesh_path_str)
        elif mesh_dir:
            # Try to find mesh in mesh_dir (use original obj_id for file lookup)
            potential_mesh = Path(mesh_dir) / obj_id / "visual.obj"
            if potential_mesh.exists():
                mesh_prim_path = f"{obj_path}/mesh"
                self._import_obj_as_mesh(stage, mesh_prim_path, str(potential_mesh))
        
        # Add physics properties (can be nested under 'physics' or at top level)
        if include_physics:
            physics = obj.get('physics', {})
            # Also check for physics properties at top level
            if not physics:
                physics = {}
                if 'mass' in obj:
                    physics['mass'] = obj['mass']
                if 'friction' in obj:
                    physics['friction'] = obj['friction']
                if 'restitution' in obj:
                    physics['restitution'] = obj['restitution']
                if 'center_of_mass' in obj:
                    physics['center_of_mass'] = obj['center_of_mass']
            if physics:
                self._add_physics_properties(stage, obj_path, physics)
        
        # Add custom attributes
        prim = stage.GetPrimAtPath(obj_path)
        if obj.get('category'):
            prim.CreateAttribute("mars:category", Sdf.ValueTypeNames.String).Set(obj['category'])
        if obj.get('object_type'):
            prim.CreateAttribute("mars:object_type", Sdf.ValueTypeNames.String).Set(obj['object_type'])
    
    def _import_obj_as_mesh(self, stage: 'Usd.Stage', mesh_path: str, obj_file: str):
        """Import OBJ file as USD mesh with vertex colors."""
        try:
            import trimesh
            from pxr import Vt
            
            mesh = trimesh.load(obj_file)
            
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)
            
            # Create USD mesh
            usd_mesh = UsdGeom.Mesh.Define(stage, mesh_path)
            
            # Set vertices
            points = [Gf.Vec3f(*v) for v in mesh.vertices]
            usd_mesh.GetPointsAttr().Set(points)
            
            # Set faces
            face_vertex_counts = [3] * len(mesh.faces)  # All triangles
            face_vertex_indices = mesh.faces.flatten().tolist()
            
            usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            usd_mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
            
            # Set normals if available
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals]
                usd_mesh.GetNormalsAttr().Set(normals)
                usd_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            
            # Set vertex colors if available
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                vertex_colors = mesh.visual.vertex_colors
                # Convert to RGB float [0,1] - vertex_colors is RGBA uint8
                colors = [Gf.Vec3f(c[0]/255.0, c[1]/255.0, c[2]/255.0) for c in vertex_colors[:, :3]]
                display_color = usd_mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
                display_color.Set(Vt.Vec3fArray(colors))
                logger.debug(f"Added {len(colors)} vertex colors to mesh")
            
            logger.debug(f"Imported mesh: {obj_file} ({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)")
            
        except Exception as e:
            logger.warning(f"Failed to import mesh {obj_file}: {e}")
    
    def _add_physics_properties(self, stage: 'Usd.Stage', obj_path: str, physics: Dict):
        """Add physics properties to an object."""
        prim = stage.GetPrimAtPath(obj_path)
        
        # Add rigid body
        if physics.get('mass'):
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.GetMassAttr().Set(physics['mass'])
            
            if physics.get('center_of_mass'):
                com = physics['center_of_mass']
                mass_api.GetCenterOfMassAttr().Set(Gf.Vec3f(*com))
        
        # Add collision
        UsdPhysics.CollisionAPI.Apply(prim)
        
        # Add material properties
        if physics.get('friction') is not None or physics.get('restitution') is not None:
            # Create physics material
            mat_path = f"{obj_path}/physics_material"
            physics_mat = UsdShade.Material.Define(stage, mat_path)
            
            # For physics, we use PhysicsMaterialAPI
            mat_prim = stage.GetPrimAtPath(mat_path)
            phys_mat_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
            
            if physics.get('friction') is not None:
                # Static and dynamic friction
                friction = physics['friction']
                phys_mat_api.GetStaticFrictionAttr().Set(friction)
                phys_mat_api.GetDynamicFrictionAttr().Set(friction * 0.8)  # Dynamic usually lower
            
            if physics.get('restitution') is not None:
                phys_mat_api.GetRestitutionAttr().Set(physics['restitution'])
    
    def _add_zone(self, stage: 'Usd.Stage', parent_path: str, zone: Dict):
        """Add a zone (pick/place area) to the stage."""
        zone_name = zone.get('name', 'zone')
        safe_zone_name = self._sanitize_prim_name(zone_name)
        zone_path = f"{parent_path}/{safe_zone_name}"
        
        bounds = zone.get('bounds', [0, 0, 0, 1, 1, 1])
        
        # Create a cube to represent the zone
        cube = UsdGeom.Cube.Define(stage, zone_path)
        
        # Calculate center and size from bounds
        min_pt = bounds[:3]
        max_pt = bounds[3:]
        center = [(min_pt[i] + max_pt[i]) / 2 for i in range(3)]
        size = [(max_pt[i] - min_pt[i]) for i in range(3)]
        
        # Set transform
        xform_api = UsdGeom.XformCommonAPI(cube)
        xform_api.SetTranslate(Gf.Vec3d(*center))
        xform_api.SetScale(Gf.Vec3f(*size))
        
        # Make it a guide (non-renderable)
        cube.GetPurposeAttr().Set(UsdGeom.Tokens.guide)
        
        # Add zone type
        prim = stage.GetPrimAtPath(zone_path)
        prim.CreateAttribute("mars:zone_type", Sdf.ValueTypeNames.String).Set(zone.get('type', 'unknown'))
    
    def _add_camera(self, stage: 'Usd.Stage', parent_path: str, camera_config: Dict):
        """Add camera to the stage."""
        camera_path = f"{parent_path}/camera"
        camera = UsdGeom.Camera.Define(stage, camera_path)
        
        # Set camera properties
        if 'focal_length' in camera_config:
            camera.GetFocalLengthAttr().Set(camera_config['focal_length'])
        
        if 'fov' in camera_config:
            # Convert FOV to focal length (assuming 36mm sensor width)
            import math
            fov_rad = math.radians(camera_config['fov'])
            focal_length = 18.0 / math.tan(fov_rad / 2)  # 36mm / 2 = 18mm half-width
            camera.GetFocalLengthAttr().Set(focal_length)
        
        # Set position
        if 'position' in camera_config:
            pos = camera_config['position']
            xform_api = UsdGeom.XformCommonAPI(camera)
            xform_api.SetTranslate(Gf.Vec3d(*pos))
    
    def _add_lighting(self, stage: 'Usd.Stage', parent_path: str, lighting_config: Dict):
        """Add lighting to the stage."""
        lights_path = f"{parent_path}/lights"
        UsdGeom.Xform.Define(stage, lights_path)
        
        # Add ambient/dome light
        if lighting_config.get('ambient_intensity'):
            try:
                from pxr import UsdLux
                dome_path = f"{lights_path}/dome_light"
                dome = UsdLux.DomeLight.Define(stage, dome_path)
                dome.GetIntensityAttr().Set(lighting_config['ambient_intensity'])
            except ImportError:
                pass
        
        # Add directional/distant light
        if lighting_config.get('key_light'):
            try:
                from pxr import UsdLux
                key_config = lighting_config['key_light']
                key_path = f"{lights_path}/key_light"
                key_light = UsdLux.DistantLight.Define(stage, key_path)
                
                if 'intensity' in key_config:
                    key_light.GetIntensityAttr().Set(key_config['intensity'])
                if 'direction' in key_config:
                    # Set rotation to match direction
                    dir_vec = key_config['direction']
                    xform_api = UsdGeom.XformCommonAPI(key_light)
                    # Calculate rotation from direction vector
                    # This is simplified - proper implementation would use look-at
            except ImportError:
                pass
    
    def _export_fallback(self, scene_graph: Dict, output_path: str) -> str:
        """Fallback when USD libraries aren't available - export as JSON with USD-like structure."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        scene_id = scene_graph.get('scene_id', 'scene')
        fallback_path = output_dir / f"{scene_id}_usd_fallback.json"
        
        # Create USD-like structure in JSON
        usd_structure = {
            "_format": "USD-like JSON (install usd-core for real USD)",
            "defaultPrim": scene_id,
            "metersPerUnit": self.meters_per_unit,
            "upAxis": self.up_axis,
            "prims": {
                scene_id: {
                    "type": "Xform",
                    "children": {
                        "objects": scene_graph.get('objects', []),
                        "zones": scene_graph.get('zones', []),
                        "camera": scene_graph.get('camera', {}),
                        "lighting": scene_graph.get('lighting', {})
                    },
                    "metadata": scene_graph.get('metadata', {})
                }
            }
        }
        
        with open(fallback_path, 'w') as f:
            json.dump(usd_structure, f, indent=2, default=str)
        
        logger.warning(f"USD not available. Exported fallback JSON: {fallback_path}")
        return str(fallback_path)


def export_scene_to_usd(
    scene_graph: Dict[str, Any],
    output_path: str,
    mesh_dir: Optional[str] = None,
    include_physics: bool = True,
    format: str = 'usda'
) -> str:
    """
    Convenience function to export a scene to USD.
    
    Args:
        scene_graph: Scene graph from SceneGraphBuilder
        output_path: Output directory
        mesh_dir: Directory with OBJ meshes
        include_physics: Include physics properties
        format: 'usda' or 'usdc'
        
    Returns:
        Path to exported file
    """
    exporter = USDExporter()
    return exporter.export(scene_graph, output_path, mesh_dir, include_physics, format)

