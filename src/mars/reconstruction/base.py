from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

class ReconstructionModel(ABC):
    """
    Abstract base class for 3D reconstruction models.
    Any model (SAM 3D, TripoSR, etc.) must inherit from this.
    """
    
    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        """Load model weights into memory."""
        pass

    @abstractmethod
    def reconstruct(
        self, 
        image_path: str, 
        output_dir: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run reconstruction on a single image.
        
        Args:
            image_path: Path to input image (or mask).
            output_dir: Where to save the .obj/.glb files.
            metadata: Additional info (camera, mask, etc.)
            
        Returns:
            Dict containing paths to generated 3D files (mesh, texture) and metrics.
        """
        pass

