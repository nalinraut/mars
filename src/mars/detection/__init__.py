"""
Detection module for MARS.

Provides object detection before segmentation.

Model:
- Hybrid: Qwen2.5-VL for labels + GroundingDINO for accurate bboxes
"""

from .base import DetectionModel, Detection
from .factory import get_detection_model, MODEL_REGISTRY
from .hybrid import HybridDetector
from .qwen_direct import QwenDirectDetector

__all__ = [
    "DetectionModel",
    "Detection",
    "HybridDetector",
    "QwenDirectDetector",
    "get_detection_model",
    "MODEL_REGISTRY",
]


def detect_objects(
    image_path: str,
    config_path: str = "config/detection.yaml",
    output_dir: str = None,
) -> dict:
    """
    Detect objects in an image.
    
    Args:
        image_path: Path to input image
        config_path: Path to detection config
        output_dir: Optional output directory for visualizations
        
    Returns:
        Dictionary with detections and metadata
    """
    import yaml
    from pathlib import Path
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    detection_config = config.get('detection', {})
    model_name = detection_config.get('model', {}).get('type', 'hybrid')
    
    # Get detector
    detector = get_detection_model(model_name, detection_config)
    
    # Load model
    checkpoint = detection_config.get('model', {}).get('checkpoint', None)
    device = detection_config.get('model', {}).get('device', 'cuda')
    detector.load_model(checkpoint_path=checkpoint, device=device, config=detection_config)
    
    # Run detection
    detections = detector.detect(image_path, detection_config)
    
    # Save visualization if output_dir provided
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        detector.visualize(image_path, detections, str(output_path / "detections.jpg"))
    
    return {
        "status": "success",
        "image_path": image_path,
        "model": model_name,
        "detections": [d.to_dict() for d in detections],
        "count": len(detections),
    }
