import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

@dataclass
class Zone:
    name: str
    bounds: List[float] # [min_x, min_y, min_z, max_x, max_y, max_z]
    type: str # "pick", "place", "static"

@dataclass
class SceneGraph:
    scene_id: str
    objects: List[Dict]
    zones: List[Zone]
    camera: Dict
    lighting: Dict
    metadata: Dict

class SceneGraphBuilder:
    """
    Assembles the final hierarchical scene definition.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
    def build(
        self, 
        scene_id: str, 
        objects: List[Dict],
        metadata: Dict = None
    ) -> Dict[str, Any]:
        
        # 1. Define Zones from Config
        zones = []
        for name, props in self.config['composition']['zones'].items():
            c = props['center']
            s = props['size']
            # Calculate bounds: center +/- size/2
            bounds = [
                c[0] - s[0]/2, c[1] - s[1]/2, c[2] - s[2]/2,
                c[0] + s[0]/2, c[1] + s[1]/2, c[2] + s[2]/2
            ]
            zones.append(Zone(name=name, bounds=bounds, type=name))
            
        # 2. Setup Environment
        camera = self.config['composition']['camera']
        lighting = self.config['composition']['lighting']
        
        # 3. Create Graph
        graph = SceneGraph(
            scene_id=scene_id,
            objects=objects,
            zones=zones,
            camera=camera,
            lighting=lighting,
            metadata=metadata or {}
        )
        
        return asdict(graph)

    def save(
        self, 
        scene_graph: Dict, 
        output_dir: str,
        export_usd: bool = True,
        mesh_dir: str = None
    ) -> Dict[str, str]:
        """
        Save scene graph to JSON and optionally USD.
        
        Args:
            scene_graph: The scene graph dictionary
            output_dir: Output directory
            export_usd: Whether to export USD file (default: True)
            mesh_dir: Directory containing mesh files for USD export
            
        Returns:
            Dict with paths: {'json': path, 'usd': path or None}
        """
        scene_id = scene_graph['scene_id']
        scene_dir = Path(output_dir) / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON (with numpy type conversion)
        def convert_numpy(obj):
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(i) for i in obj]
            return obj
        
        json_path = scene_dir / "scene.json"
        with open(json_path, 'w') as f:
            json.dump(convert_numpy(scene_graph), f, indent=2)
        
        result = {'json': str(json_path), 'usd': None}
        
        # Export USD if requested
        if export_usd:
            try:
                from ..exporters.usd_exporter import USDExporter
                exporter = USDExporter()
                usd_path = exporter.export(
                    scene_graph=scene_graph,
                    output_path=str(scene_dir),
                    mesh_dir=mesh_dir,
                    include_physics=True,
                    format='usdc'  # Binary format for Isaac Sim
                )
                result['usd'] = usd_path
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"USD export failed: {e}")
        
        return result

