"""
Depth Anything V2 integration for scene composition.

Two modes:
1. Relative depth (default) - More robust, works on any scene
2. Metric depth - Fine-tuned for indoor/outdoor, gives meters

Usage:
    estimator = DepthAnythingV2Estimator(metric=True)  # or metric=False
    depth_map = estimator.estimate(image_path)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class DepthAnythingV2Output:
    """Holds Depth Anything V2 output with helper methods."""
    
    def __init__(
        self,
        depth: np.ndarray,
        original_size: Tuple[int, int],
        is_metric: bool = False
    ):
        self.depth = depth  # (H, W) - relative or metric depth
        self.original_size = original_size  # (W, H)
        self.is_metric = is_metric
        
        # For relative depth, normalize to 0-1 range
        if not is_metric:
            self.depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        else:
            self.depth_normalized = depth
    
    def get_depth_at_pixel(self, x: int, y: int, radius: int = 5) -> float:
        """
        Get depth at a pixel location.
        Uses median over neighborhood for robustness.
        """
        h, w = self.depth.shape
        
        # Scale coordinates if depth map is different size than original
        scale_x = w / self.original_size[0]
        scale_y = h / self.original_size[1]
        
        px = int(x * scale_x)
        py = int(y * scale_y)
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        
        # Sample neighborhood
        y1, y2 = max(0, py - radius), min(h, py + radius + 1)
        x1, x2 = max(0, px - radius), min(w, px + radius + 1)
        
        neighborhood = self.depth[y1:y2, x1:x2]
        return float(np.median(neighborhood))
    
    def get_relative_depth(self, x: int, y: int) -> float:
        """Get normalized depth (0=close, 1=far) at pixel."""
        h, w = self.depth_normalized.shape
        scale_x = w / self.original_size[0]
        scale_y = h / self.original_size[1]
        
        px = int(x * scale_x)
        py = int(y * scale_y)
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        
        return float(self.depth_normalized[py, px])
    
    def get_depth_ordering(self, objects: List[Dict]) -> List[str]:
        """
        Return object mask_ids sorted by depth (closest first).
        """
        depths = []
        for obj in objects:
            mask_id = obj.get('mask_id', 'unknown')
            centroid = obj.get('centroid', [0, 0])
            d = self.get_relative_depth(int(centroid[0]), int(centroid[1]))
            depths.append((mask_id, d))
        
        # Sort by depth (lower = closer for relative depth)
        depths.sort(key=lambda x: x[1])
        return [d[0] for d in depths]


class DepthAnythingV2Estimator:
    """
    Depth Anything V2 estimator.
    
    Models:
    - Relative: depth-anything/Depth-Anything-V2-Small/Base/Large-hf
    - Metric Indoor: depth-anything/Depth-Anything-V2-Metric-Indoor-Small/Base/Large-hf
    - Metric Outdoor: depth-anything/Depth-Anything-V2-Metric-Outdoor-Small/Base/Large-hf
    """
    
    # Model configurations
    MODELS = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf',
        'metric-indoor-small': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
        'metric-indoor-base': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf',
        'metric-indoor-large': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf',
        'metric-outdoor-small': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf',
        'metric-outdoor-base': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf',
        'metric-outdoor-large': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf',
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._cache: Dict[str, DepthAnythingV2Output] = {}
        
        # Configuration
        self.model_size = self.config.get('model_size', 'large')  # small, base, large
        self.metric = self.config.get('metric', False)  # Use metric depth?
        self.scene_type = self.config.get('scene_type', 'indoor')  # indoor or outdoor
        
    def _get_model_id(self) -> str:
        """Get the appropriate model ID based on config."""
        if self.metric:
            key = f'metric-{self.scene_type}-{self.model_size}'
        else:
            key = self.model_size
        
        if key not in self.MODELS:
            logger.warning(f"Unknown model key {key}, falling back to large")
            key = 'large'
        
        return self.MODELS[key]
    
    def load_model(self) -> bool:
        """Load Depth Anything V2 model."""
        if self.model is not None:
            return True
        
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            model_id = self._get_model_id()
            logger.info(f"Loading Depth Anything V2: {model_id}")
            
            self.processor = AutoImageProcessor.from_pretrained(model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"✓ Depth Anything V2 loaded ({self.model_size}, metric={self.metric})")
            return True
            
        except ImportError as e:
            logger.error(f"transformers not available: {e}")
            logger.error("Install with: pip install transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to load Depth Anything V2: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def estimate(self, image_path: str) -> Optional[DepthAnythingV2Output]:
        """
        Run depth estimation on an image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            DepthAnythingV2Output with depth map and helper methods
        """
        # Check cache
        cache_key = f"{image_path}_{self.model_size}_{self.metric}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if not self.load_model():
            return None
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size  # (W, H)
            
            logger.info(f"Depth Anything V2 inference: {original_size}")
            
            # Process image
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Interpolate to original size
            depth = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(original_size[1], original_size[0]),  # (H, W)
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()
            
            if self.metric:
                logger.info(f"Depth range: [{depth.min():.2f}, {depth.max():.2f}] meters")
            else:
                logger.info(f"Relative depth range: [{depth.min():.2f}, {depth.max():.2f}]")
            
            result = DepthAnythingV2Output(
                depth=depth,
                original_size=original_size,
                is_metric=self.metric
            )
            
            # Cache result
            self._cache[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_depths_for_objects(
        self,
        image_path: str,
        objects: List[Dict[str, Any]],
        image_size: Tuple[int, int]
    ) -> Dict[str, float]:
        """
        Get depth values for objects (for Z positioning).
        
        Returns normalized depths (0 = front, 1 = back) suitable
        for scene Z coordinate.
        """
        result = self.estimate(image_path)
        depths = {}
        
        for obj in objects:
            mask_id = obj.get('mask_id', 'unknown')
            centroid = obj.get('centroid', [image_size[0] // 2, image_size[1] // 2])
            
            if result:
                # Get relative depth (already normalized if not metric)
                rel_depth = result.get_relative_depth(int(centroid[0]), int(centroid[1]))
                depths[mask_id] = rel_depth
            else:
                # Fallback: use Y position as proxy for depth
                cy = centroid[1]
                depths[mask_id] = cy / image_size[1]
        
        return depths
    
    def compute_depth_ordering(
        self,
        image_path: str,
        objects: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Return objects sorted by depth (closest to camera first).
        
        Useful for:
        - Occlusion reasoning
        - Layer ordering
        - Contact detection
        """
        result = self.estimate(image_path)
        if result:
            return result.get_depth_ordering(objects)
        
        # Fallback: sort by Y position (bottom = closer)
        sorted_objs = sorted(
            objects,
            key=lambda o: o.get('centroid', [0, 0])[1],
            reverse=True
        )
        return [o.get('mask_id', 'unknown') for o in sorted_objs]


# Singleton instance
_depth_anything_estimator = None

def get_depth_anything_estimator(config: Dict[str, Any] = None) -> DepthAnythingV2Estimator:
    """Get or create a DepthAnythingV2Estimator instance."""
    global _depth_anything_estimator
    if _depth_anything_estimator is None:
        _depth_anything_estimator = DepthAnythingV2Estimator(config)
    return _depth_anything_estimator

