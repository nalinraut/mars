import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class SceneArchiver:
    """
    Handles final organization and persistence of scene assets.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root_dir = Path(config['storage']['root_dir'])
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
    def archive_scene(
        self, 
        scene_graph: Dict[str, Any], 
        quality_record: Dict[str, Any],
        image_id: str
    ) -> Dict[str, str]:
        """
        Move assets to final location, rewrite paths, and save metadata.
        """
        final_scene_dir = self.root_dir / image_id
        final_scene_dir.mkdir(exist_ok=True)
        
        # 1. Copy Objects & Rewrite Paths
        updated_objects = []
        for obj in scene_graph['objects']:
            updated_obj = obj.copy()
            
            # Handle Mesh paths
            # We copy them from staging/processing to final storage
            for path_key in ['mesh_path', 'collision_path']:
                src_path_str = obj.get(path_key)
                if src_path_str and Path(src_path_str).exists():
                    src_path = Path(src_path_str)
                    # Structure: {scene_id}/meshes/{object_id}_{type}.obj
                    dest_filename = f"{obj['mask_id']}_{path_key.split('_')[0]}.obj"
                    dest_path = final_scene_dir / "meshes" / dest_filename
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    shutil.copy2(src_path, dest_path)
                    
                    # Store relative path in final JSON for portability
                    updated_obj[path_key] = f"meshes/{dest_filename}"
            
            updated_objects.append(updated_obj)
            
        # 2. Serialize Final JSON
        final_scene_graph = scene_graph.copy()
        final_scene_graph['objects'] = updated_objects
        final_scene_graph['quality_scores'] = quality_record
        final_scene_graph['archived_at'] = datetime.utcnow().isoformat()
        
        scene_json_path = final_scene_dir / "scene.json"
        with open(scene_json_path, 'w') as f:
            json.dump(final_scene_graph, f, indent=2)
            
        # 3. Generate Thumbnail (Optional)
        # For now, copy original thumbnail if exists in metadata
        # TODO: Implement render-based thumbnail
        
        logger.info(f"Archived scene {image_id} to {final_scene_dir}")
        
        return {
            "scene_id": image_id,
            "local_path": str(final_scene_dir),
            "json_path": str(scene_json_path)
        }

