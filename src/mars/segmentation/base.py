from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from pathlib import Path

class SegmentationModel(ABC):
    """
    Abstract base class for segmentation models.
    Any model (SAM 2, SAM v1, GroundingDINO, etc.) must inherit from this.
    """
    
    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """Load model weights into memory."""
        pass

    @abstractmethod
    def generate_masks(
        self, 
        image_path: str,
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Run segmentation on an image.
        
        Args:
            image_path: Path to input image
            config: Segmentation configuration
            
        Returns:
            List of mask dictionaries with keys: segmentation, area, bbox, predicted_iou, stability_score, etc.
        """
        pass

