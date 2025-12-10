from typing import Dict, Any, Type
from .base import SegmentationModel
from .sam2 import Sam2Segmentation
from .sam3 import Sam3Segmentation

# Registry of available models
MODEL_REGISTRY: Dict[str, Type[SegmentationModel]] = {
    "sam3": Sam3Segmentation,  # SAM 3 - Latest (Nov 2025), with text prompts (BEST)
    "sam2": Sam2Segmentation,  # SAM 2.1 from Transformers
    # Future additions:
    # "groundingdino": GroundingDINOSegmentation  # Text-prompted detection
}

def get_segmentation_model(name: str, config: Dict[str, Any]) -> SegmentationModel:
    """Factory function to get the requested segmentation model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[name]
    return model_class(config)

__all__ = ["SegmentationModel", "Sam2Segmentation", "Sam3Segmentation", "get_segmentation_model"]

