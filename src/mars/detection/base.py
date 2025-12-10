"""
Abstract base class for detection models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path


@dataclass
class Detection:
    """
    Represents a single object detection.
    
    Attributes:
        label: Detected object label/class
        confidence: Detection confidence score (0-1)
        bbox: Bounding box as (x1, y1, x2, y2) in pixels
        bbox_normalized: Bounding box normalized to (0-1) range
        estimated_size: Estimated real-world size in meters (from LLM)
        material: Detected material type (for physics)
        metadata: Additional detection-specific data
    """
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels
    bbox_normalized: Tuple[float, float, float, float] = None  # x1, y1, x2, y2 normalized
    estimated_size: Optional[float] = None  # Real-world size in meters
    material: Optional[str] = None  # Material type for physics
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "bbox_normalized": self.bbox_normalized,
            "estimated_size": self.estimated_size,
            "material": self.material,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Detection":
        """Create from dictionary."""
        return cls(
            label=data["label"],
            confidence=data["confidence"],
            bbox=tuple(data["bbox"]),
            bbox_normalized=tuple(data["bbox_normalized"]) if data.get("bbox_normalized") else None,
            estimated_size=data.get("estimated_size"),
            material=data.get("material"),
            metadata=data.get("metadata", {}),
        )
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    @property
    def area(self) -> int:
        """Get area of bounding box in pixels."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class DetectionModel(ABC):
    """
    Abstract base class for object detection models.
    
    Implementations:
    - GroundingDINODetector: Open-vocabulary detection with text prompts
    - (Future) YOLO: Fast closed-vocabulary detection
    - (Future) OWLv2: Open-vocabulary detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detector.
        
        Args:
            config: Detection configuration dictionary
        """
        self.config = config
        self.model = None
        self.device = "cuda"
    
    @abstractmethod
    def load_model(
        self, 
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda"
    ) -> None:
        """
        Load model weights.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to model config (if separate)
            device: Device to load model on
        """
        pass
    
    @abstractmethod
    def detect(
        self,
        image_path: str,
        config: Dict[str, Any]
    ) -> List[Detection]:
        """
        Detect objects in an image.
        
        Args:
            image_path: Path to input image
            config: Detection configuration
            
        Returns:
            List of Detection objects
        """
        pass
    
    def visualize(
        self,
        image_path: str,
        detections: List[Detection],
        output_path: str
    ) -> None:
        """
        Visualize detections on image.
        
        Args:
            image_path: Path to input image
            detections: List of detections to visualize
            output_path: Path to save visualization
        """
        import cv2
        import numpy as np
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Colors for different labels
        np.random.seed(42)
        colors = {}
        
        for det in detections:
            # Get or create color for this label
            if det.label not in colors:
                colors[det.label] = tuple(map(int, np.random.randint(0, 255, 3)))
            color = colors[det.label]
            
            # Draw bounding box
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label_text = f"{det.label}: {det.confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Background for text
            cv2.rectangle(
                image,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 5, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                image,
                label_text,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        cv2.imwrite(output_path, image)
    
    def filter_by_confidence(
        self,
        detections: List[Detection],
        threshold: float
    ) -> List[Detection]:
        """Filter detections by confidence threshold."""
        return [d for d in detections if d.confidence >= threshold]
    
    def filter_by_labels(
        self,
        detections: List[Detection],
        labels: List[str]
    ) -> List[Detection]:
        """Filter detections to only include specific labels."""
        labels_lower = [l.lower() for l in labels]
        return [d for d in detections if d.label.lower() in labels_lower]
    
    def remove_duplicates(
        self,
        detections: List[Detection],
        iou_threshold: float = 0.5
    ) -> List[Detection]:
        """Remove duplicate detections using NMS."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
        
        kept = []
        for det in detections:
            # Check if overlaps with any kept detection
            should_keep = True
            for kept_det in kept:
                if self._iou(det.bbox, kept_det.bbox) > iou_threshold:
                    should_keep = False
                    break
            if should_keep:
                kept.append(det)
        
        return kept
    
    def _iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

