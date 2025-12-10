"""
MoGe-based depth estimation for scene composition.
Uses MoGe to estimate depth from the source image,
then uses intrinsics to compute real-world object sizes.

MoGe Output:
- depth: Per-pixel depth in meters (H, W)
- intrinsics: 3x3 camera matrix (normalized 0-1)
  [[fx, 0, cx],
   [0, fy, cy],
   [0,  0,  1]]
- points: 3D point cloud (H, W, 3)
- mask: Valid depth mask (H, W)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class MoGeOutput:
    """Holds MoGe inference output with helper methods."""
    
    def __init__(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        points: np.ndarray,
        mask: np.ndarray,
        inference_size: Tuple[int, int],
        original_size: Tuple[int, int]
    ):
        self.depth = depth  # (H, W) depth in meters
        self.intrinsics = intrinsics  # (3, 3) normalized intrinsics
        self.points = points  # (H, W, 3) 3D points
        self.mask = mask  # (H, W) valid depth mask
        self.inference_size = inference_size  # (W, H) size used for inference
        self.original_size = original_size  # (W, H) original image size
        
        # Extract intrinsics in pixel units (at inference resolution)
        self.fx = intrinsics[0, 0] * inference_size[0]
        self.fy = intrinsics[1, 1] * inference_size[1]
        self.cx = intrinsics[0, 2] * inference_size[0]
        self.cy = intrinsics[1, 2] * inference_size[1]
        
        # Scale factors to map from original image coords to inference coords
        self.scale_x = inference_size[0] / original_size[0]
        self.scale_y = inference_size[1] / original_size[1]
    
    def get_depth_at_pixel(self, x: int, y: int, radius: int = 5) -> float:
        """
        Get depth at a pixel location (in original image coordinates).
        Uses median over a small neighborhood for robustness.
        """
        # Convert to inference coordinates
        ix = int(x * self.scale_x)
        iy = int(y * self.scale_y)
        
        h, w = self.depth.shape
        ix = max(0, min(ix, w - 1))
        iy = max(0, min(iy, h - 1))
        
        # Sample neighborhood
        y1, y2 = max(0, iy - radius), min(h, iy + radius + 1)
        x1, x2 = max(0, ix - radius), min(w, ix + radius + 1)
        
        neighborhood = self.depth[y1:y2, x1:x2]
        mask_region = self.mask[y1:y2, x1:x2]
        
        if mask_region.any():
            return float(np.median(neighborhood[mask_region]))
        return float(np.median(neighborhood))
    
    def compute_real_world_size(
        self,
        bbox: List[int],
        depth_at_object: float = None
    ) -> Dict[str, float]:
        """
        Compute real-world size of an object given its bounding box.
        
        Formula: real_size = (pixel_size * depth) / focal_length
        
        Args:
            bbox: [x1, y1, x2, y2] in original image coordinates
            depth_at_object: Override depth (if None, compute from bbox center)
            
        Returns:
            Dict with 'width', 'height', 'max_dim' in meters
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Pixel dimensions in inference coordinates
        pixel_width = abs(x2 - x1) * self.scale_x
        pixel_height = abs(y2 - y1) * self.scale_y
        
        # Get depth at object center if not provided
        if depth_at_object is None:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            depth_at_object = self.get_depth_at_pixel(int(cx), int(cy))
        
        # Real-world size: (pixel_size * depth) / focal_length
        real_width = (pixel_width * depth_at_object) / self.fx
        real_height = (pixel_height * depth_at_object) / self.fy
        
        return {
            'width': float(real_width),
            'height': float(real_height),
            'max_dim': float(max(real_width, real_height)),
            'depth': float(depth_at_object)
        }
    
    def pixel_to_3d(self, x: int, y: int) -> Tuple[float, float, float]:
        """
        Convert pixel coordinates to 3D world coordinates.
        
        Args:
            x, y: Pixel coordinates in original image
            
        Returns:
            (X, Y, Z) in meters
        """
        # Convert to inference coordinates
        ix = int(x * self.scale_x)
        iy = int(y * self.scale_y)
        
        h, w = self.depth.shape
        ix = max(0, min(ix, w - 1))
        iy = max(0, min(iy, h - 1))
        
        # Get depth
        z = self.depth[iy, ix]
        
        # Unproject: X = (x - cx) * Z / fx, Y = (y - cy) * Z / fy
        X = (ix - self.cx) * z / self.fx
        Y = (iy - self.cy) * z / self.fy
        
        return (float(X), float(Y), float(z))


class DepthEstimator:
    """
    Estimates depth and computes real-world sizes using MoGe.
    
    MoGe provides:
    - Metric depth estimation (in meters)
    - Camera intrinsics (focal length, principal point)
    - 3D point cloud
    
    This enables accurate real-world size calculation:
        real_size = (pixel_size * depth) / focal_length
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache: Dict[str, MoGeOutput] = {}
        
        # Inference resolution (balance between accuracy and memory)
        self.inference_size = self.config.get('inference_size', 512)
    
    def load_model(self):
        """Load MoGe model."""
        if self.model is not None:
            return True
        
        try:
            from moge.model.v1 import MoGeModel
            
            model_id = self.config.get('model_id', 'Ruicheng/moge-vitl')
            logger.info(f"Loading MoGe model: {model_id}")
            
            self.model = MoGeModel.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("✓ MoGe model loaded successfully")
            return True
            
        except ImportError:
            logger.warning("MoGe not available - using fallback estimation")
            self.model = None
            return False
        except Exception as e:
            logger.warning(f"Failed to load MoGe: {e} - using fallback")
            self.model = None
            return False
    
    def estimate(self, image_path: str) -> Optional[MoGeOutput]:
        """
        Run MoGe inference on an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            MoGeOutput with depth, intrinsics, and helper methods
        """
        # Check cache
        if image_path in self._cache:
            return self._cache[image_path]
        
        if not self.load_model():
            return None
        
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (W, H)
            
            # Resize for inference (preserve aspect ratio)
            max_dim = max(original_size)
            scale = self.inference_size / max_dim
            new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
            
            # Make dimensions divisible by 32 for model compatibility
            new_size = ((new_size[0] // 32) * 32, (new_size[1] // 32) * 32)
            new_size = (max(32, new_size[0]), max(32, new_size[1]))
            
            image_resized = image.resize(new_size, Image.LANCZOS)
            inference_size = new_size
            
            logger.info(f"MoGe inference: {original_size} -> {inference_size}")
            
            # Convert to tensor
            image_np = np.array(image_resized)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model.infer(image_tensor)
            
            # Extract outputs
            depth = output['depth'][0].cpu().numpy()  # (H, W)
            intrinsics = output['intrinsics'][0].cpu().numpy()  # (3, 3)
            points = output['points'][0].cpu().numpy()  # (H, W, 3)
            mask = output['mask'][0].cpu().numpy()  # (H, W)
            
            logger.info(f"MoGe output: depth=[{depth.min():.2f}, {depth.max():.2f}]m, "
                       f"fx={intrinsics[0,0]:.3f}, fy={intrinsics[1,1]:.3f}")
            
            result = MoGeOutput(
                depth=depth,
                intrinsics=intrinsics,
                points=points,
                mask=mask,
                inference_size=inference_size,
                original_size=original_size
            )
            
            # Cache result
            self._cache[image_path] = result
            return result
            
        except Exception as e:
            logger.error(f"MoGe estimation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_object_sizes(
        self,
        image_path: str,
        objects: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute real-world sizes for all objects using MoGe.
        
        Args:
            image_path: Path to source image
            objects: List of objects with 'mask_id', 'bbox', and 'centroid'
            
        Returns:
            Dict mapping mask_id to size info {'width', 'height', 'max_dim', 'depth'}
        """
        moge_output = self.estimate(image_path)
        sizes = {}
        
        for obj in objects:
            mask_id = obj.get('mask_id', 'unknown')
            bbox = obj.get('bbox')
            
            if moge_output is not None and bbox:
                # Use MoGe for accurate sizing
                size_info = moge_output.compute_real_world_size(bbox)
                sizes[mask_id] = size_info
                logger.debug(f"MoGe size for {mask_id}: {size_info['max_dim']:.3f}m")
            else:
                # Fallback: use bbox area to estimate relative size
                sizes[mask_id] = self._fallback_size(obj)
        
        return sizes
    
    def _fallback_size(self, obj: Dict[str, Any]) -> Dict[str, float]:
        """Fallback size estimation when MoGe is unavailable."""
        bbox = obj.get('bbox', [0, 0, 100, 100])
        x1, y1, x2, y2 = bbox[:4]
        
        # Assume objects at ~2m depth, ~60 degree FOV
        # This gives rough estimates
        assumed_depth = 2.0
        assumed_focal = 500  # pixels
        
        pixel_width = abs(x2 - x1)
        pixel_height = abs(y2 - y1)
        
        real_width = (pixel_width * assumed_depth) / assumed_focal
        real_height = (pixel_height * assumed_depth) / assumed_focal
        
        return {
            'width': float(real_width),
            'height': float(real_height),
            'max_dim': float(max(real_width, real_height)),
            'depth': assumed_depth,
            'is_fallback': True
        }
    
    def get_3d_positions(
        self,
        image_path: str,
        objects: List[Dict[str, Any]]
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Get 3D world positions for objects based on their centroids.
        
        Args:
            image_path: Path to source image
            objects: List of objects with 'mask_id' and 'centroid'
            
        Returns:
            Dict mapping mask_id to (X, Y, Z) position in meters
        """
        moge_output = self.estimate(image_path)
        positions = {}
        
        for obj in objects:
            mask_id = obj.get('mask_id', 'unknown')
            centroid = obj.get('centroid', [0, 0])
            
            if moge_output is not None:
                pos = moge_output.pixel_to_3d(int(centroid[0]), int(centroid[1]))
                positions[mask_id] = pos
            else:
                # Fallback: relative positioning
                positions[mask_id] = (0.0, 0.0, 2.0)
        
        return positions
    
    # Legacy methods for backwards compatibility
    def estimate_depth(self, image_path: str) -> Optional[np.ndarray]:
        """Legacy: Get depth map only."""
        result = self.estimate(image_path)
        return result.depth if result else None
    
    def get_depth_at_centroid(
        self,
        image_path: str,
        centroid: List[int],
        mask: Optional[np.ndarray] = None
    ) -> float:
        """Legacy: Get depth at a centroid."""
        result = self.estimate(image_path)
        if result:
            return result.get_depth_at_pixel(int(centroid[0]), int(centroid[1]))
        return 2.0  # Default fallback
    
    def get_depths_for_objects(
        self,
        image_path: str,
        objects: List[Dict[str, Any]],
        image_size: Tuple[int, int]
    ) -> Dict[str, float]:
        """Legacy: Get normalized depth values for Z positioning."""
        result = self.estimate(image_path)
        depths = {}
        
        for obj in objects:
            mask_id = obj.get('mask_id', 'unknown')
            centroid = obj.get('centroid', [image_size[0] // 2, image_size[1] // 2])
            
            if result:
                depths[mask_id] = result.get_depth_at_pixel(int(centroid[0]), int(centroid[1]))
            else:
                # Fallback based on Y position
                cy = centroid[1]
                depths[mask_id] = 1.0 - (cy / image_size[1])
        
        # Normalize to scene scale (0 to 0.5m range for relative Z)
        if depths:
            min_d = min(depths.values())
            max_d = max(depths.values())
            range_d = max_d - min_d if max_d > min_d else 1.0
            
            for mask_id in depths:
                normalized = (depths[mask_id] - min_d) / range_d
                depths[mask_id] = normalized * 0.5
        
        return depths


# Singleton instance
_depth_estimator = None

def get_depth_estimator(config: Dict[str, Any] = None) -> DepthEstimator:
    """Get or create a DepthEstimator instance."""
    global _depth_estimator
    if _depth_estimator is None:
        _depth_estimator = DepthEstimator(config)
    return _depth_estimator
