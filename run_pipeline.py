#!/usr/bin/env python
"""
MARS Pipeline CLI

Run the complete 2D-to-3D scene generation pipeline.

Usage:
    python run_pipeline.py <image_path> [options]
    
Examples:
    # Run full Phase 1 pipeline (up to composition)
    python run_pipeline.py scene.jpg
    
    # Run only detection
    python run_pipeline.py scene.jpg --run-until detection
    
    # Run with physics estimation
    python run_pipeline.py scene.jpg --enable-physics
    
    # Custom output directory
    python run_pipeline.py scene.jpg -o output/my_scene
"""

import argparse
import sys
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging before importing mars (to set up Prefect properly)
from src.mars.utils.logging_config import setup_logging, configure_prefect_logging
configure_prefect_logging()

from src.mars import run, STAGE_MAP


def main():
    parser = argparse.ArgumentParser(
        description="MARS: Multi Asset Reconstruction for Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Stages:
  1. ingestion      - Validate and prepare image
  2. detection      - Detect objects (Qwen2-VL)
  3. segmentation   - Generate masks (SAM 3)
  4. reconstruction - Create 3D meshes (SAM 3D Objects)
  5. physics        - Estimate physical properties
  6. composition    - Compose scene with layout
  7. validation     - Physics simulation validation
  8. storage        - Index and export

Phase 1 (default): Runs stages 1-6 (ingestion → composition)
        """
    )
    
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to input image"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    parser.add_argument(
        "--run-until",
        type=str,
        default="composition",
        choices=list(STAGE_MAP.keys()),
        help="Run pipeline until this stage (default: composition)"
    )
    
    parser.add_argument(
        "--enable-physics",
        action="store_true",
        help="Enable physics estimation stage"
    )
    
    parser.add_argument(
        "--enable-validation",
        action="store_true",
        help="Enable physics validation stage"
    )
    
    parser.add_argument(
        "--enable-storage",
        action="store_true",
        help="Enable storage/indexing stage"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging with Prefect noise suppression
    import logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(
        level=log_level,
        suppress_prefect_noise=True
    )
    
    logger = logging.getLogger("mars")
    
    # Validate image path
    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        sys.exit(1)
    
    # Run pipeline
    logger.info("=" * 60)
    logger.info("MARS Pipeline")
    logger.info("=" * 60)
    logger.info(f"Input:     {image_path}")
    logger.info(f"Output:    {args.output_dir}")
    logger.info(f"Run until: {args.run_until}")
    logger.info("=" * 60)
    
    try:
        result = run(
            image_path=str(image_path),
            output_dir=args.output_dir,
            run_until=args.run_until,
            enable_physics=args.enable_physics,
            enable_validation=args.enable_validation,
            enable_storage=args.enable_storage,
        )
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Status: {result.get('status', 'unknown')}")
        logger.info(f"Stages: {' → '.join(result.get('stages_completed', []))}")
        
        # Print stage-specific info
        if "detection" in result:
            det = result["detection"]
            if det.get("status") == "success":
                logger.info(f"Detected: {det.get('count', 0)} objects")
                for d in det.get("detections", [])[:5]:
                    material = d.get("metadata", {}).get("material", "")
                    logger.info(f"  - {d['label']} ({material})")
        
        if "segmentation" in result:
            seg = result["segmentation"]
            if seg.get("status") == "success":
                logger.info(f"Masks: {seg.get('mask_count', 0)} generated")
        
        if "reconstruction" in result:
            rec = result["reconstruction"]
            if rec.get("status") == "success":
                logger.info(f"Meshes: {len(rec.get('objects', []))} reconstructed")
        
        if "composition" in result:
            comp = result["composition"]
            if comp.get("status") == "success":
                logger.info(f"Scene: {comp.get('scene_path', 'N/A')}")
        
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

