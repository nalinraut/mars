"""
MARS - Multi Asset Reconstruction for Simulation
Transforms images into physics-ready 3D scene assets
"""

__version__ = "0.1.0"

from .pipeline import MARSPipeline, PipelineStage, STAGE_MAP

__all__ = ["MARSPipeline", "PipelineStage", "STAGE_MAP"]


def run(
    image_path: str,
    output_dir: str = "output",
    run_until: str = "composition",
    enable_physics: bool = False,
    enable_validation: bool = False,
    enable_storage: bool = False,
):
    """
    Convenience function to run the MARS pipeline.
    
    Args:
        image_path: Path to input image
        output_dir: Output directory for results
        run_until: Stage to stop at (ingestion, detection, segmentation,
                   reconstruction, physics, composition, validation, storage)
        enable_physics: Enable physics estimation
        enable_validation: Enable physics validation
        enable_storage: Enable storage/indexing
        
    Returns:
        Pipeline results dictionary
        
    Example:
        >>> from src.mars import run
        >>> result = run("image.jpg", run_until="composition")
    """
    from pathlib import Path
    
    pipeline = MARSPipeline(
        sam_checkpoint="",  # Loaded from HuggingFace
        sam3d_checkpoint="",  # Loaded from checkpoints/
        output_dir=Path(output_dir),
        enable_physics=enable_physics,
        enable_validation=enable_validation,
        enable_storage=enable_storage,
    )
    
    return pipeline.process(
        image_path=Path(image_path),
        run_until=run_until,
    )

