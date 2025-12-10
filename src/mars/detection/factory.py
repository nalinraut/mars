"""
Factory for detection models.
"""

from typing import Dict, Any, Type
from .base import DetectionModel
from .hybrid import HybridDetector
from .qwen_direct import QwenDirectDetector


# Registry of available detection models
MODEL_REGISTRY: Dict[str, Type[DetectionModel]] = {
    # Qwen Direct: Single model for labels + bboxes (FASTER, less memory)
    "qwen_direct": QwenDirectDetector,
    "qwen": QwenDirectDetector,  # Alias
    
    # Hybrid: Qwen2.5-VL labels + GroundingDINO bboxes (MORE ACCURATE bboxes)
    "hybrid": HybridDetector,
    "hybrid_qwen_gdino": HybridDetector,  # Alias
}


def get_detection_model(name: str, config: Dict[str, Any]) -> DetectionModel:
    """
    Factory function to get a detection model by name.
    
    Args:
        name: Model name (e.g., "hybrid")
        config: Detection configuration
        
    Returns:
        Instantiated DetectionModel
        
    Raises:
        ValueError: If model name not found
    """
    name_lower = name.lower().replace("-", "_")
    
    if name_lower not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{name}' not found. Available: {available}"
        )
    
    model_class = MODEL_REGISTRY[name_lower]
    return model_class(config)


def list_models() -> Dict[str, str]:
    """List available detection models with descriptions."""
    return {
        "hybrid": "Qwen2.5-VL for labels + GroundingDINO for bboxes (RECOMMENDED)",
    }


__all__ = [
    "DetectionModel",
    "HybridDetector",
    "get_detection_model",
    "list_models",
    "MODEL_REGISTRY",
]
