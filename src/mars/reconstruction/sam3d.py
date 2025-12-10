import logging
import numpy as np
import trimesh
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import sys
import os
from PIL import Image

from .base import ReconstructionModel

logger = logging.getLogger(__name__)

class Sam3DReconstructor(ReconstructionModel):
    """
    Implementation for Meta's 'SAM 3D Objects: 3Dfy Anything in Images'.
    Reconstructs single-view images into 3D meshes using SAM 3D Objects.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = "cuda"
        self.inference = None
        self.load_image_func = None
        self.load_mask_func = None
        
        # Add SAM 3D Objects to path BEFORE any imports
        sam3d_path = config.get('sam3d_path', '/opt/sam-3d-objects')
        notebook_path = os.path.join(sam3d_path, 'notebook')
        
        # Insert at beginning of path
        if notebook_path not in sys.path:
            sys.path.insert(0, notebook_path)
        if sam3d_path not in sys.path:
            sys.path.insert(0, sam3d_path)
        
        # Set environment variables early
        if 'VIRTUAL_ENV' not in os.environ:
            os.environ['VIRTUAL_ENV'] = '/opt/mars-env'
        if 'CONDA_PREFIX' not in os.environ:
            os.environ['CONDA_PREFIX'] = os.environ.get('VIRTUAL_ENV', '/opt/mars-env')
        if 'CUDA_HOME' not in os.environ:
            os.environ['CUDA_HOME'] = '/usr/local/cuda'
        os.environ['LIDRA_SKIP_INIT'] = 'true'

    def load_model(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        self.device = device
        logger.info(f"Loading SAM 3D Objects model on {device}...")
        
        try:
            # Import inference module (paths and env vars set in __init__)
            from inference import Inference, load_image, load_mask
            
            # Determine checkpoint path - prefer host-mounted checkpoints
            checkpoint_tag = self.config.get('checkpoint_tag', 'sam3d')
            
            # Priority order for checkpoints:
            # 1. Host-mounted checkpoints (/workspace/checkpoints/sam3d)
            # 2. Legacy in-container path (/opt/sam-3d-objects/checkpoints)
            checkpoint_locations = [
                # Host-mounted (preferred)
                f"/workspace/checkpoints/sam3d/checkpoints/pipeline.yaml",
                f"/workspace/checkpoints/{checkpoint_tag}/checkpoints/pipeline.yaml",
                # Legacy locations
                f"/opt/sam-3d-objects/checkpoints/hf/checkpoints/pipeline.yaml",
                f"/opt/sam-3d-objects/checkpoints/hf/pipeline.yaml",
            ]
            
            config_path = None
            for loc in checkpoint_locations:
                if os.path.exists(loc):
                    config_path = loc
                    break
            
            if config_path is None:
                raise FileNotFoundError(
                    f"SAM 3D Objects checkpoints not found. "
                    f"Run: init_checkpoints.sh or download from Hugging Face. "
                    f"Searched: {checkpoint_locations}"
                )
            
            logger.info(f"Loading SAM 3D Objects from config: {config_path}")
            
            # MoGe is now properly installed - no need for stubs
            # MoGe provides accurate depth estimation for 3D reconstruction
            try:
                from moge.model.v1 import MoGeModel
                logger.info("✓ MoGe depth estimation available")
            except ImportError:
                logger.warning("MoGe not available - depth estimation may be limited")
            
            try:
                self.inference = Inference(config_path, compile=False)
                self.load_image_func = load_image
                self.load_mask_func = load_mask
                logger.info("✓ SAM 3D Objects model loaded successfully")
            except Exception as e:
                logger.warning(f"SAM 3D Objects initialization error: {e}")
                logger.warning("Will use placeholder meshes. Check dependencies.")
                self.inference = None
            
        except Exception as e:
            logger.error(f"Failed to load SAM 3D Objects: {e}", exc_info=True)
            logger.warning("SAM 3D Objects not available. Will use size-appropriate placeholder meshes.")
            logger.warning("To enable SAM 3D Objects, run: init_checkpoints.sh")
            # Don't raise - allow pipeline to continue with placeholders
            self.inference = None 

    def reconstruct(
        self, 
        image_path: str, 
        output_dir: Path, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform 3D reconstruction for a specific object mask.
        """
        image_id = metadata.get('image_id', 'unknown')
        mask_id = metadata.get('mask_id', 'unknown')
        object_id = metadata.get('object_id', mask_id)  # Use labeled ID if available
        
        logger.info(f"Running SAM 3D on object {object_id} from {image_id}")
        
        # 1. Prepare Input
        # SAM 3D likely takes the full image + a box/mask prompt to know WHAT to 3Dfy.
        bbox = metadata.get('bbox') # [x1, y1, x2, y2]
        mask_path = metadata.get('mask_path')
        
        if not bbox:
            logger.warning(f"No bbox provided for {mask_id}, utilizing full image context.")

        # 2. Run Inference
        # We pass the crop or the full image with prompt
        mesh_data = self._run_inference(image_path, bbox, mask_path)
        
        # 3. Post-Processing & Saving
        results = self._process_and_save_mesh(mesh_data, output_dir, object_id)
        
        return results

    def _run_inference(
        self, 
        image_path: str, 
        bbox: Optional[List[int]],
        mask_path: Optional[str]
    ) -> trimesh.Trimesh:
        """
        Internal method to call the SAM 3D Objects model.
        """
        if self.inference is None:
            logger.warning("SAM 3D Objects not available, using size-appropriate placeholder mesh")
            return self._create_placeholder_mesh(image_path, mask_path, bbox)
        
        logger.info(f"Running SAM 3D Objects inference on {image_path} with mask {mask_path}")
        
        try:
            # Load image
            image = self.load_image_func(image_path)
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Load mask if provided
            mask = None
            if mask_path and os.path.exists(mask_path):
                mask = self.load_mask_func(mask_path)
                if mask.ndim == 3:
                    mask = mask[..., -1] if mask.shape[2] > 1 else mask[..., 0]
                mask = mask > 0  # Ensure boolean mask
            elif bbox:
                # Create mask from bbox if no mask file
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=bool)
                # Convert bbox values to int (may be strings from JSON)
                x1, y1, x2, y2 = [int(float(v)) for v in bbox]
                mask[y1:y2, x1:x2] = True
                logger.info(f"Created mask from bbox: {[x1, y1, x2, y2]}")
            else:
                logger.warning("No mask or bbox provided, using full image")
                # Use full image as mask
                mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
            
            # Run inference
            seed = self.config.get('seed', 42)
            output = self.inference(image, mask, seed=seed)
            
            # Extract mesh from output
            # SAM 3D Objects returns a dict with 'glb' (mesh) or 'gs' (Gaussian Splat)
            mesh = None
            
            if 'glb' in output and output['glb'] is not None:
                # Direct mesh available - preserve vertex colors
                mesh_glb = output['glb']
                
                if hasattr(mesh_glb, 'vertices'):
                    # Already a mesh-like object
                    vertices = mesh_glb.vertices
                    faces = mesh_glb.faces if hasattr(mesh_glb, 'faces') else None
                    
                    # Extract vertex colors if available
                    vertex_colors = None
                    if hasattr(mesh_glb, 'vertex_colors'):
                        vertex_colors = mesh_glb.vertex_colors
                    elif hasattr(mesh_glb, 'colors'):
                        vertex_colors = mesh_glb.colors
                    elif hasattr(mesh_glb, 'visual') and hasattr(mesh_glb.visual, 'vertex_colors'):
                        vertex_colors = mesh_glb.visual.vertex_colors
                    
                    if faces is not None:
                        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                        if vertex_colors is not None:
                            # Apply vertex colors
                            mesh.visual = trimesh.visual.ColorVisuals(
                                mesh=mesh, 
                                vertex_colors=vertex_colors
                            )
                            logger.info(f"✓ Preserved {len(vertex_colors)} vertex colors")
                    else:
                        mesh = trimesh.Trimesh(vertices=vertices)
                        
                elif isinstance(mesh_glb, bytes):
                    # GLB as bytes - load directly
                    import io
                    mesh = trimesh.load(io.BytesIO(mesh_glb), file_type='glb')
                    logger.info(f"Loaded mesh from GLB bytes")
                    
                elif isinstance(mesh_glb, str) and os.path.exists(mesh_glb):
                    # Load from file path (preserves vertex colors automatically)
                    mesh = trimesh.load(mesh_glb)
                    logger.info(f"Loaded mesh from file: {mesh_glb}")
            
            if mesh is None and 'gs' in output:
                # Convert Gaussian Splat to mesh
                gs = output['gs']
                if hasattr(gs, 'save_ply'):
                    # Export to temporary PLY and load
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        gs.save_ply(tmp_path)
                        mesh = trimesh.load(tmp_path)
                        os.unlink(tmp_path)
                    except Exception as e:
                        logger.warning(f"Could not convert Gaussian Splat to mesh: {e}")
                        if os.path.exists(tmp_path):
                            os.unlink(tmp_path)
            
            if mesh is None:
                logger.warning("No mesh found in SAM 3D Objects output, using size-appropriate placeholder")
                mesh = self._create_placeholder_mesh(image_path, mask_path, bbox)
            else:
                logger.info(f"✓ Extracted mesh with {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
            
            return mesh
            
        except Exception as e:
            logger.error(f"SAM 3D Objects inference failed: {e}", exc_info=True)
            # Fallback to size-appropriate placeholder
            logger.warning("Using size-appropriate placeholder mesh due to inference failure")
            try:
                return self._create_placeholder_mesh(image_path, mask_path, bbox)
            except Exception as e2:
                logger.error(f"Placeholder creation also failed: {e2}")
                return trimesh.creation.icosphere(radius=0.1, subdivisions=2)
    
    def _create_placeholder_mesh(self, image_path: str, mask_path: Optional[str], bbox: Optional[List[int]]) -> trimesh.Trimesh:
        """
        Create a placeholder mesh that matches the object size.
        Uses bbox or mask to estimate object dimensions.
        """
        try:
            import cv2
            
            # Load image to get dimensions
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
                if img is None:
                    img = np.array(Image.open(image_path))
                h, w = img.shape[:2]
            else:
                h, w = image_path.shape[:2] if hasattr(image_path, 'shape') else (512, 512)
            
            # Estimate object size
            if bbox:
                x1, y1, x2, y2 = bbox
                obj_w = (x2 - x1) / w
                obj_h = (y2 - y1) / h
                obj_size = max(obj_w, obj_h) * 0.3  # Scale to reasonable 3D size
            elif mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask_bool = mask > 0
                    y_coords, x_coords = np.where(mask_bool)
                    if len(x_coords) > 0:
                        obj_w = (x_coords.max() - x_coords.min()) / w
                        obj_h = (y_coords.max() - y_coords.min()) / h
                        obj_size = max(obj_w, obj_h) * 0.3
                    else:
                        obj_size = 0.1
                else:
                    obj_size = 0.1
            else:
                obj_size = 0.1  # Default small size
            
            # Create an ellipsoid that roughly matches the object proportions
            # Use a more detailed icosphere and scale it
            radius = max(obj_size, 0.05)  # Minimum size
            mesh = trimesh.creation.icosphere(radius=radius, subdivisions=2)
            
            logger.debug(f"Created placeholder mesh with radius {radius:.3f}")
            return mesh
            
        except Exception as e:
            logger.warning(f"Could not create size-appropriate placeholder: {e}")
            # Ultimate fallback
            return trimesh.creation.icosphere(radius=0.1, subdivisions=2)

    def _process_and_save_mesh(
        self, 
        mesh: trimesh.Trimesh, 
        output_dir: Path, 
        object_id: str
    ) -> Dict[str, Any]:
        """
        Clean, repair, and save the mesh.
        
        Args:
            object_id: Labeled object identifier (e.g., 'laptop_mask_0')
        """
        out_dir = output_dir / object_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # A. Comprehensive Mesh Repair
        mesh = self._repair_mesh(mesh)
            
        # B. Compute Properties
        volume = mesh.volume
        bounds = mesh.bounds 
        bbox_dims = (bounds[1] - bounds[0]).tolist()
        
        # C. Save Visual Mesh (High Detail) - OBJ format
        visual_path = out_dir / "visual.obj"
        mesh.export(visual_path)
        
        # D. Collision mesh generation moved to flow.py for config-based method selection
        # This allows V-HACD parameters to be configured via reconstruction.yaml
        
        # E. Export USDC format (for Isaac Sim / Omniverse)
        usdc_path = self._export_usdc(mesh, out_dir)
        
        return {
            "mesh_path": str(visual_path),
            "usdc_path": str(usdc_path) if usdc_path else None,
            "format": "obj",
            "vertices_count": len(mesh.vertices),
            "is_watertight": mesh.is_watertight,
            "volume": volume,
            "bbox_dims": bbox_dims,
            "model_used": "sam3d_meta"
        }
    
    def _repair_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Comprehensive mesh repair for robust physics simulation.
        
        Repairs:
        - Degenerate/duplicate faces
        - Duplicate vertices
        - Flipped/inconsistent normals
        - Holes in the mesh
        - Non-manifold edges
        """
        original_verts = len(mesh.vertices)
        original_faces = len(mesh.faces)
        
        try:
            # 1. Remove degenerate faces (zero area triangles)
            mesh.remove_degenerate_faces()
            
            # 2. Remove duplicate faces
            mesh.remove_duplicate_faces()
            
            # 3. Merge close/duplicate vertices
            mesh.merge_vertices()
            
            # 4. Remove unreferenced vertices
            mesh.remove_unreferenced_vertices()
            
            # 5. Fix face winding for consistent normals
            trimesh.repair.fix_winding(mesh)
            
            # 6. Fix inverted faces
            trimesh.repair.fix_inversion(mesh)
            
            # 7. Fix normals (ensure outward-facing)
            trimesh.repair.fix_normals(mesh)
            
            # 8. Fill holes if not watertight (important for physics)
            if not mesh.is_watertight:
                trimesh.repair.fill_holes(mesh)
            
            # 9. Final cleanup pass
            mesh.remove_degenerate_faces()
            mesh.merge_vertices()
            
            final_verts = len(mesh.vertices)
            final_faces = len(mesh.faces)
            
            if original_verts != final_verts or original_faces != final_faces:
                logger.info(f"Mesh repair: {original_verts}v/{original_faces}f -> {final_verts}v/{final_faces}f, watertight={mesh.is_watertight}")
            
        except Exception as e:
            logger.warning(f"Mesh repair partially failed: {e}")
        
        return mesh
    
    def _export_usdc(self, mesh: 'trimesh.Trimesh', out_dir: Path) -> Optional[Path]:
        """Export mesh to USDC format with vertex colors preserved."""
        try:
            from pxr import Usd, UsdGeom, Sdf, Gf, Vt
            
            usdc_path = out_dir / "visual.usdc"
            stage = Usd.Stage.CreateNew(str(usdc_path))
            
            # Set stage metadata
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            
            # Create mesh prim
            mesh_prim = UsdGeom.Mesh.Define(stage, "/mesh")
            
            # Set vertices
            points = [Gf.Vec3f(*v) for v in mesh.vertices]
            mesh_prim.GetPointsAttr().Set(points)
            
            # Set faces (all triangles)
            face_vertex_counts = [3] * len(mesh.faces)
            face_vertex_indices = mesh.faces.flatten().tolist()
            mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
            mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)
            
            # Set normals
            if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None:
                normals = [Gf.Vec3f(*n) for n in mesh.vertex_normals]
                mesh_prim.GetNormalsAttr().Set(normals)
                mesh_prim.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            
            # Set vertex colors (displayColor)
            if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                # Normalize to 0-1 range and take RGB only
                colors = mesh.visual.vertex_colors[:, :3].astype(float) / 255.0
                display_colors = [Gf.Vec3f(*c) for c in colors]
                color_primvar = mesh_prim.GetDisplayColorPrimvar()
                color_primvar.Set(display_colors)
                color_primvar.SetInterpolation(UsdGeom.Tokens.vertex)
            
            stage.SetDefaultPrim(mesh_prim.GetPrim())
            stage.Save()
            
            logger.info(f"Exported USDC: {usdc_path}")
            return usdc_path
            
        except ImportError:
            logger.warning("USD libraries not available, skipping USDC export")
            return None
        except Exception as e:
            logger.warning(f"USDC export failed: {e}")
            return None
