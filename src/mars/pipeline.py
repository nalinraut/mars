"""
Main MARS Pipeline
Orchestrates all stages of scene generation
"""

from pathlib import Path
from typing import Optional, Dict, Any, Literal
import logging
import json
import shutil
from enum import IntEnum

# Configure Prefect logging before importing flows
from src.mars.utils.logging_config import configure_prefect_logging, setup_logging, patch_prefect_emit
configure_prefect_logging()

from src.mars.ingestion import ingest_image
from src.mars.detection import detect_objects
from src.mars.segmentation import segment_image
from src.mars.reconstruction import reconstruct_scene
from src.mars.physics import estimate_physics
from src.mars.composition import compose_scene
from src.mars.validation import validate_scene_flow
from src.mars.storage import store_scene

# Patch Prefect events after all imports (to suppress "stopped service" errors)
patch_prefect_emit()

logger = logging.getLogger(__name__)


class PipelineStage(IntEnum):
    """Pipeline stages in execution order."""
    INGESTION = 1
    DETECTION = 2
    SEGMENTATION = 3
    RECONSTRUCTION = 4
    PHYSICS = 5
    COMPOSITION = 6
    VALIDATION = 7
    STORAGE = 8


# String to enum mapping
STAGE_MAP = {
    "ingestion": PipelineStage.INGESTION,
    "detection": PipelineStage.DETECTION,
    "segmentation": PipelineStage.SEGMENTATION,
    "reconstruction": PipelineStage.RECONSTRUCTION,
    "physics": PipelineStage.PHYSICS,
    "composition": PipelineStage.COMPOSITION,
    "validation": PipelineStage.VALIDATION,
    "storage": PipelineStage.STORAGE,
}


class MARSPipeline:
    """
    Main pipeline for generating 3D scenes from 2D images.
    
    Pipeline Stages:
    1. Ingestion - Image validation and preparation
    2. Detection - Object detection with Grounding DINO (auto or manual prompts)
    3. Segmentation - Precise masks with SAM (using detections)
    4. Reconstruction - 3D mesh generation with SAM 3D Objects
    5. Physics - Material and property estimation (optional)
    6. Composition - Scene layout reconstruction
    7. Validation - Physics simulation (optional)
    8. Storage - Indexing and export (optional)
    """
    
    def __init__(
        self,
        sam_checkpoint: str,
        sam3d_checkpoint: str,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        enable_physics: bool = False,
        enable_validation: bool = True,  # PyBullet stability check
        enable_storage: bool = False,
    ):
        """
        Initialize MARS pipeline.
        
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            sam3d_checkpoint: Path to SAM 3D Objects checkpoint
            output_dir: Directory for output scenes
            config: Optional configuration dictionary
            enable_physics: Enable physics estimation stage
            enable_validation: Enable physics validation stage
            enable_storage: Enable storage/indexing stage
        """
        self.sam_checkpoint = Path(sam_checkpoint)
        self.sam3d_checkpoint = Path(sam3d_checkpoint)
        self.output_dir = Path(output_dir)
        self.config = config or {}
        
        # Optional stages
        self.enable_physics = enable_physics
        self.enable_validation = enable_validation
        self.enable_storage = enable_storage
        
        # Stage output directories (created per-run in _setup_stage_dirs)
        self.stage_dirs = {}
        
        logger.info("Initializing MARS Pipeline...")
        logger.info(f"  Physics: {'enabled' if enable_physics else 'disabled'}")
        logger.info(f"  Validation: {'enabled' if enable_validation else 'disabled'}")
        logger.info(f"  Storage: {'enabled' if enable_storage else 'disabled'}")
    
    def _setup_stage_dirs(self, image_id: str) -> Dict[str, Path]:
        """Create per-stage output directories."""
        base = self.output_dir / image_id
        
        dirs = {
            "ingestion": base / "1_ingestion",
            "detection": base / "2_detection",
            "segmentation": base / "3_segmentation",
            "reconstruction": base / "4_reconstruction",
            "composition": base / "5_composition",
        }
        
        for stage_dir in dirs.values():
            stage_dir.mkdir(parents=True, exist_ok=True)
        
        self.stage_dirs = dirs
        return dirs
    
    def _save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save dictionary as JSON file."""
        # Filter out non-serializable fields
        def clean(obj):
            if isinstance(obj, dict):
                return {k: clean(v) for k, v in obj.items() if not k.startswith('_')}
            elif isinstance(obj, list):
                return [clean(i) for i in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj
        
        with open(path, 'w') as f:
            json.dump(clean(data), f, indent=2, default=str)
        
    def process(
        self,
        image_path: Path,
        task_type: str = "pick_place",
        run_until: str = "composition",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a single image through the pipeline.
        
        Core stages (Phase 1):
        1. Ingestion → 2. Detection → 3. Segmentation → 4. Reconstruction → 5. Composition
        
        Optional stages (Phase 2+):
        - Physics estimation
        - Validation
        - Storage
        
        Args:
            image_path: Path to input image
            task_type: Task type (pick_place, stack, push, etc.)
            run_until: Stop after this stage. Options:
                       ingestion, detection, segmentation, reconstruction,
                       physics, composition, validation, storage
            **kwargs: Additional processing parameters
            
        Returns:
            Dictionary with results from all completed stages
        """
        # Resolve run_until to enum
        run_until_lower = run_until.lower()
        if run_until_lower not in STAGE_MAP:
            valid = list(STAGE_MAP.keys())
            raise ValueError(f"Invalid run_until='{run_until}'. Valid: {valid}")
        
        stop_after = STAGE_MAP[run_until_lower]
        
        logger.info(f"Processing image: {image_path}")
        logger.info(f"Pipeline will run until: {run_until_lower}")
        
        results = {
            "image_path": str(image_path),
            "run_until": run_until_lower,
            "stages_completed": [],
        }
        
        # Stage 1: Ingestion
        ingestion_result = self._ingest(image_path)
        ingestion_result["_original_path"] = str(image_path)
        results["ingestion"] = ingestion_result
        results["stages_completed"].append("ingestion")
        
        # Setup stage directories after we have image_id
        image_id = ingestion_result.get("image_id", "unknown")
        self._setup_stage_dirs(image_id)
        
        # Save ingestion outputs
        self._save_ingestion_outputs(ingestion_result, image_path)
        
        if stop_after == PipelineStage.INGESTION:
            results["status"] = "completed"
            return results
        
        # Get image path for downstream stages
        current_image_path = str(image_path)
        if ingestion_result.get("status") in ["ingested", "duplicate"]:
            if "record" in ingestion_result:
                current_image_path = ingestion_result["record"]["paths"]["staging_path"]
        
        # Stage 2: Detection
        detection_result = self._detect(current_image_path)
        results["detection"] = detection_result
        results["stages_completed"].append("detection")
        
        # Save detection outputs
        self._save_detection_outputs(detection_result, current_image_path)
        
        if stop_after == PipelineStage.DETECTION:
            results["status"] = "completed"
            return results
        
        # Stage 3: Segmentation (uses detections)
        segmentation_result = self._segment(ingestion_result, detection_result)
        results["segmentation"] = segmentation_result
        results["stages_completed"].append("segmentation")
        
        # Save segmentation outputs (masks are saved by segment_image, we just copy/link)
        self._save_segmentation_outputs(segmentation_result)
        
        if stop_after == PipelineStage.SEGMENTATION:
            results["status"] = "completed"
            return results
        
        # Stage 4: 3D Reconstruction (pass detections for labeling)
        reconstruction_result = self._reconstruct_3d(segmentation_result, current_image_path, detection_result)
        results["reconstruction"] = reconstruction_result
        results["stages_completed"].append("reconstruction")
        
        # Save reconstruction outputs
        self._save_reconstruction_outputs(reconstruction_result)
        
        if stop_after == PipelineStage.RECONSTRUCTION:
            results["status"] = "completed"
            return results
        
        # Stage 5: Physics Estimation (optional, or if explicitly requested)
        if self.enable_physics or stop_after >= PipelineStage.PHYSICS:
            if stop_after == PipelineStage.PHYSICS or self.enable_physics:
                physics_result = self._estimate_physics(reconstruction_result)
                results["physics"] = physics_result
                results["stages_completed"].append("physics")
            else:
                physics_result = reconstruction_result.copy()
                physics_result["physics_skipped"] = True
        else:
            physics_result = reconstruction_result.copy()
            physics_result["physics_skipped"] = True
        
        if stop_after == PipelineStage.PHYSICS:
            results["status"] = "completed"
            return results
        
        # Stage 6: Scene Composition (with detection labels for scaling)
        # DEBUG: Log what we're passing to composition
        logger.info(f"DEBUG: physics_result has {len(physics_result.get('objects', []))} objects")
        if physics_result.get('objects'):
            o = physics_result['objects'][0]
            logger.info(f"DEBUG: First object: {o.get('mask_id')}, area={o.get('area')}, bbox={o.get('bbox')}")
        composition_result = self._compose_scene(physics_result, segmentation_result, detection_result, current_image_path)
        results["composition"] = composition_result
        results["stages_completed"].append("composition")
        
        # Save composition outputs
        self._save_composition_outputs(composition_result, ingestion_result)
        
        # Update scene_path to point to the stage directory location
        # (after _save_composition_outputs copies it there)
        stage_dir = self.stage_dirs.get("composition")
        if stage_dir:
            composition_result["scene_path"] = str(stage_dir / "scene.json")
        
        # Cleanup duplicate folders (originals saved by segment/reconstruct modules)
        self._cleanup_duplicate_folders(ingestion_result.get("image_id"))
        
        if stop_after == PipelineStage.COMPOSITION:
            results["status"] = "completed"
            return results
        
        # Stage 7: Validation (optional)
        if self.enable_validation or stop_after >= PipelineStage.VALIDATION:
            if stop_after == PipelineStage.VALIDATION or self.enable_validation:
                validation_result = self._validate(composition_result, segmentation_result, reconstruction_result)
                results["validation"] = validation_result
                results["stages_completed"].append("validation")
            else:
                validation_result = composition_result.copy()
                validation_result["validation_skipped"] = True
        else:
            validation_result = composition_result.copy()
            validation_result["validation_skipped"] = True
        
        if stop_after == PipelineStage.VALIDATION:
            results["status"] = "completed"
            return results
        
        # Stage 8: Storage (optional)
        if self.enable_storage or stop_after == PipelineStage.STORAGE:
            storage_result = self._store(validation_result, composition_result)
            results["storage"] = storage_result
            results["stages_completed"].append("storage")
        
        results["status"] = "completed"
        return results
    
    def _ingest(self, image_path: Path) -> Dict[str, Any]:
        """Stage 1: Image ingestion and validation"""
        return ingest_image(str(image_path))
    
    def _detect(self, image_path: str) -> Dict[str, Any]:
        """Stage 2: Object detection with Grounding DINO"""
        logger.info("Stage 2: Running object detection...")
        
        try:
            result = detect_objects(
                image_path=image_path,
                config_path="config/detection.yaml",
                output_dir=str(self.output_dir)
            )
            
            logger.info(f"Detected {result.get('count', 0)} objects")
            for det in result.get("detections", []):
                logger.info(f"  - {det['label']}: {det['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return {"status": "failed", "error": str(e), "detections": []}
    
    def _segment(self, ingestion_result: Dict[str, Any], detection_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Stage 3: SAM 3 segmentation using text prompts from detection"""
        if ingestion_result.get("status") not in ["ingested", "duplicate"]:
            logger.error(f"Skipping segmentation, ingestion status: {ingestion_result.get('status')}")
            return {"status": "skipped", "reason": "ingestion_failed"}

        image_id = ingestion_result["image_id"]
        
        # Get staging path
        if "record" in ingestion_result and ingestion_result["record"]:
            staging_path = ingestion_result["record"]["paths"]["staging_path"]
        else:
            logger.warning("Record details not available, using original path as fallback")
            staging_path = ingestion_result.get("_original_path")
            if not staging_path:
                raise ValueError("Cannot determine staging path")
        
        # Extract labels from detections to use as SAM 3 text prompts
        text_prompts = []
        if detection_result and detection_result.get("status") == "success":
            raw_detections = detection_result.get("detections", [])
            if raw_detections:
                text_prompts = [d["label"] for d in raw_detections]
                logger.info(f"Using detection labels as SAM 3 text prompts: {text_prompts}")
        
        if not text_prompts:
            logger.warning("No detections available, using fallback prompts")
            text_prompts = ["object"]  # Fallback to generic prompt

        return segment_image(
            image_path=staging_path,
            image_id=image_id,
            output_dir=str(self.output_dir),
            text_prompts=text_prompts,
        )
    
    def _reconstruct_3d(
        self, 
        segmentation_result: Dict[str, Any], 
        image_path: str,
        detection_result: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Stage 3: SAM 3D Objects reconstruction"""
        
        if segmentation_result.get("status") != "success":
             logger.warning("Skipping reconstruction due to segmentation failure/empty")
             return {"status": "skipped", "reason": "segmentation_not_success"}

        # Extract detections for label assignment during reconstruction
        detections = []
        if detection_result and detection_result.get("status") == "success":
            detections = detection_result.get("detections", [])
            logger.info(f"Passing {len(detections)} detections to reconstruction for labeling")

        return reconstruct_scene(
           segmentation_result=segmentation_result,
           image_path=image_path,
           output_dir=str(self.output_dir),
           detections=detections
        )
    
    def _estimate_physics(self, reconstruction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 4: Physics property estimation"""
        
        if reconstruction_result.get("status") != "success":
             logger.warning("Skipping physics due to reconstruction failure")
             return {"status": "skipped", "reason": "reconstruction_not_success"}

        return estimate_physics(
            reconstruction_result=reconstruction_result,
            output_dir=str(self.output_dir)
        )
    
    def _compose_scene(
        self, 
        physics_result: Dict[str, Any], 
        segmentation_result: Dict[str, Any] = None,
        detection_result: Dict[str, Any] = None,
        source_image_path: str = None
    ) -> Dict[str, Any]:
        """Stage 6: Scene composition and layout with real-world scaling"""
        
        # Allow composition even if physics was skipped (uses reconstruction result)
        if physics_result.get("status") not in ["success"] and not physics_result.get("physics_skipped"):
            logger.warning("Skipping composition due to upstream failure")
            return {"status": "skipped", "reason": "upstream_not_success"}

        # Merge segmentation data into physics_result for compose_scene
        if segmentation_result and segmentation_result.get("status") == "success":
            physics_result["segmentation"] = segmentation_result

        # IMPORTANT: Embed detections directly into physics_result
        # This ensures they're passed through Prefect flow correctly
        if detection_result and detection_result.get("status") == "success":
            detections = detection_result.get("detections", [])
            physics_result["detections"] = detections
            logger.info(f"Embedded {len(detections)} detections into physics_result")
        
        # Pass image path for MoGe depth estimation
        # Use source_image_path directly if provided (most reliable)
        if source_image_path:
            physics_result["image_path"] = source_image_path
            logger.info(f"Using source image for MoGe: {source_image_path}")
        elif segmentation_result and segmentation_result.get("status") == "success":
            seg_data = segmentation_result.get("segmentation", segmentation_result)
            # Fallback: try to get from segmentation
            image_path = physics_result.get("segmentation", {}).get("image_path")
            if not image_path:
                # Try to get from first mask's path (derive from mask directory)
                masks = seg_data.get("masks", [])
                if masks and masks[0].get("mask_path"):
                    mask_path = Path(masks[0]["mask_path"])
                    # Mask path is like: .../image_id/masks/mask_0.png
                    # Image might be in staging: .../staging/image_id.jpg
                    staging_dir = mask_path.parent.parent.parent / "staging"
                    image_id = mask_path.parent.parent.name
                    for ext in ['.jpg', '.jpeg', '.png']:
                        candidate = staging_dir / f"{image_id}{ext}"
                        if candidate.exists():
                            image_path = str(candidate)
                            break
            physics_result["image_path"] = image_path

        return compose_scene(
            physics_result=physics_result,
            output_dir=str(self.output_dir)
        )
    
    def _validate(self, composition_result: Dict[str, Any], segmentation_result: Dict[str, Any], reconstruction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 6: MuJoCo physics validation"""
        
        if composition_result.get("status") != "success":
             logger.warning("Skipping validation due to composition failure")
             return {"status": "skipped", "reason": "composition_not_success"}

        # We need geometric metrics from reconstruction stage to be passed here
        # Or we recompute them.
        # Assuming reconstruction_result has 'objects' list with mesh paths.
        # We can re-run geometric validation quickly if needed.
        
        # Quick re-calc of geometric metrics for now
        # Ideally this should be passed from Stage 3 if expensive
        from src.mars.validation import calculate_geometric_quality
        objects = reconstruction_result.get("objects", [])
        geo_metrics = calculate_geometric_quality(objects)
        
        return validate_scene_flow(
            scene_composition_result=composition_result,
            segmentation_result=segmentation_result,
            geometric_metrics=geo_metrics
        )
    
    def _store(self, validation_result: Dict[str, Any], composition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Stage 7: Storage and indexing"""
        
        if validation_result.get("status") != "validated":
             logger.warning("Skipping storage due to validation incomplete")
             return {"status": "skipped", "reason": "validation_not_validated"}

        return store_scene(
            validation_result=validation_result,
            composition_result=composition_result
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Per-Stage Output Saving
    # ─────────────────────────────────────────────────────────────────────────
    
    def _save_ingestion_outputs(self, ingestion_result: Dict[str, Any], original_path: Path) -> None:
        """Save Stage 1 outputs: original image, resized image, metadata."""
        stage_dir = self.stage_dirs.get("ingestion")
        if not stage_dir:
            return
        
        # Copy original image
        original_path = Path(original_path)
        if original_path.exists():
            shutil.copy2(original_path, stage_dir / f"original{original_path.suffix}")
        
        # Copy resized/staging image if available
        if ingestion_result.get("record") and ingestion_result["record"].get("paths"):
            staging_path = ingestion_result["record"]["paths"].get("staging_path")
            if staging_path and Path(staging_path).exists():
                shutil.copy2(staging_path, stage_dir / "processed.jpg")
            
            thumb_path = ingestion_result["record"]["paths"].get("thumbnail_path")
            if thumb_path and Path(thumb_path).exists():
                shutil.copy2(thumb_path, stage_dir / "thumbnail.jpg")
        
        # Save metadata
        metadata = {
            "image_id": ingestion_result.get("image_id"),
            "status": ingestion_result.get("status"),
            "dimensions": ingestion_result.get("record", {}).get("dimensions"),
            "file_size": ingestion_result.get("record", {}).get("file_size"),
            "format": ingestion_result.get("record", {}).get("format"),
        }
        self._save_json(metadata, stage_dir / "metadata.json")
        logger.info(f"Saved ingestion outputs to {stage_dir}")
    
    def _save_detection_outputs(self, detection_result: Dict[str, Any], image_path: str) -> None:
        """Save Stage 2 outputs: detection visualization, detections.json."""
        stage_dir = self.stage_dirs.get("detection")
        if not stage_dir:
            return
        
        # Copy visualization if exists in output_dir
        vis_path = self.output_dir / "detections.jpg"
        if vis_path.exists():
            shutil.copy2(vis_path, stage_dir / "detections.jpg")
        
        # Save detections with full details
        detections_data = {
            "status": detection_result.get("status"),
            "model": detection_result.get("model"),
            "count": detection_result.get("count", 0),
            "objects": []
        }
        
        for det in detection_result.get("detections", []):
            obj = {
                "label": det.get("label"),
                "confidence": det.get("confidence"),
                "bbox": det.get("bbox"),
                "bbox_normalized": det.get("bbox_normalized"),
                "material": det.get("metadata", {}).get("material"),
                "attributes": det.get("metadata", {}).get("attributes", {}),
            }
            detections_data["objects"].append(obj)
        
        self._save_json(detections_data, stage_dir / "detections.json")
        logger.info(f"Saved detection outputs to {stage_dir}")
    
    def _save_segmentation_outputs(self, segmentation_result: Dict[str, Any]) -> None:
        """Save Stage 3 outputs: masks and metadata."""
        stage_dir = self.stage_dirs.get("segmentation")
        if not stage_dir:
            return
        
        masks_dir = stage_dir / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        # Copy masks from original location
        masks_data = {"masks": []}
        for mask in segmentation_result.get("masks", []):
            mask_path = Path(mask.get("mask_path", ""))
            if mask_path.exists():
                shutil.copy2(mask_path, masks_dir / mask_path.name)
                
            mask_info = {
                "mask_id": mask.get("mask_id"),
                "category": mask.get("category"),
                "confidence": mask.get("confidence"),
                "bbox": mask.get("bbox"),
                "area": mask.get("area"),
                "centroid": mask.get("centroid"),
            }
            masks_data["masks"].append(mask_info)
        
        masks_data["count"] = len(masks_data["masks"])
        masks_data["quality_score"] = segmentation_result.get("quality_score")
        masks_data["image_size"] = segmentation_result.get("image_size")
        
        self._save_json(masks_data, stage_dir / "masks.json")
        logger.info(f"Saved segmentation outputs to {stage_dir}")
    
    def _save_reconstruction_outputs(self, reconstruction_result: Dict[str, Any]) -> None:
        """Save Stage 4 outputs: per-object visual.obj and collision.obj."""
        stage_dir = self.stage_dirs.get("reconstruction")
        if not stage_dir:
            return
        
        objects_dir = stage_dir / "objects"
        objects_dir.mkdir(exist_ok=True)
        
        objects_data = {"objects": []}
        
        for obj in reconstruction_result.get("objects", []):
            # Use object_id (labeled) if available, fallback to mask_id
            object_id = obj.get("object_id", obj.get("mask_id", "unknown"))
            obj_dir = objects_dir / object_id
            obj_dir.mkdir(exist_ok=True)
            
            # Copy mesh files (OBJ format)
            visual_path = Path(obj.get("mesh_path", ""))
            if visual_path.exists():
                shutil.copy2(visual_path, obj_dir / "visual.obj")
            
            collision_path = Path(obj.get("collision_path", ""))
            if collision_path.exists():
                shutil.copy2(collision_path, obj_dir / "collision.obj")
            
            # Copy USDC file if available
            usdc_path = Path(obj.get("usdc_path", "") or "")
            if usdc_path.exists():
                shutil.copy2(usdc_path, obj_dir / "visual.usdc")
            
            obj_info = {
                "mask_id": obj.get("mask_id"),
                "object_id": object_id,
                "label": obj.get("label"),
                "vertices_count": obj.get("vertices_count"),
                "is_watertight": obj.get("is_watertight"),
                "volume": obj.get("volume"),
                "bbox_dims": obj.get("bbox_dims"),
                "model_used": obj.get("model_used"),
            }
            objects_data["objects"].append(obj_info)
        
        objects_data["count"] = len(objects_data["objects"])
        objects_data["status"] = reconstruction_result.get("status")
        
        self._save_json(objects_data, stage_dir / "reconstruction.json")
        logger.info(f"Saved reconstruction outputs to {stage_dir}")
    
    def _save_composition_outputs(self, composition_result: Dict[str, Any], ingestion_result: Dict[str, Any]) -> None:
        """Save Stage 5 outputs: scene.usdc, scene.json, preview image."""
        stage_dir = self.stage_dirs.get("composition")
        if not stage_dir:
            return
        
        image_id = ingestion_result.get("image_id", "unknown")
        
        # Copy USD file (prefer USDC for Isaac Sim, fallback to USDA)
        usdc_path = self.output_dir / image_id / f"{image_id}.usdc"
        usda_path = self.output_dir / image_id / f"{image_id}.usda"
        if usdc_path.exists():
            shutil.copy2(usdc_path, stage_dir / "scene.usdc")
        elif usda_path.exists():
            shutil.copy2(usda_path, stage_dir / "scene.usda")
        
        # Copy scene.json
        scene_json_path = self.output_dir / image_id / "scene.json"
        if scene_json_path.exists():
            shutil.copy2(scene_json_path, stage_dir / "scene.json")
        
        # Copy original image as reference
        if ingestion_result.get("record") and ingestion_result["record"].get("paths"):
            staging_path = ingestion_result["record"]["paths"].get("staging_path")
            if staging_path and Path(staging_path).exists():
                shutil.copy2(staging_path, stage_dir / "source_image.jpg")
        
        # Save composition summary
        summary = {
            "status": composition_result.get("status"),
            "scene_id": composition_result.get("scene_id"),
            "object_count": len(composition_result.get("objects", [])),
            "usd_path": str(stage_dir / "scene.usdc"),
            "scene_graph_path": str(stage_dir / "scene.json"),
        }
        self._save_json(summary, stage_dir / "composition.json")
        
        # Generate scene visualization
        self._generate_scene_visualization(composition_result, stage_dir)
        
        logger.info(f"Saved composition outputs to {stage_dir}")
    
    def _cleanup_duplicate_folders(self, image_id: str) -> None:
        """Remove duplicate folders and files created by stage modules."""
        if not image_id:
            return
        
        base = self.output_dir / image_id
        
        # Folders to remove (originals - data now in stage directories)
        duplicate_folders = [
            base / "masks",
            base / "objects",
        ]
        
        for folder in duplicate_folders:
            if folder.exists() and folder.is_dir():
                try:
                    shutil.rmtree(folder)
                    logger.debug(f"Cleaned up duplicate folder: {folder}")
                except Exception as e:
                    logger.warning(f"Could not remove {folder}: {e}")
        
        # Files to remove from image_id folder (now in stage directories)
        duplicate_files = [
            base / "scene.json",           # Now in 5_composition
            base / f"{image_id}.usda",     # Now in 5_composition
        ]
        
        for file in duplicate_files:
            if file.exists():
                try:
                    file.unlink()
                except:
                    pass
        
        # Also remove detections.jpg from output_dir root (now in 2_detection)
        root_detections = self.output_dir / "detections.jpg"
        if root_detections.exists():
            try:
                root_detections.unlink()
            except:
                pass
    
    def _generate_scene_visualization(self, composition_result: Dict[str, Any], stage_dir: Path) -> None:
        """Generate a visualization image of the composed 3D scene."""
        try:
            import trimesh
            import numpy as np
            
            # Load all meshes into a scene
            scene = trimesh.Scene()
            
            for obj in composition_result.get("objects", []):
                mesh_path = obj.get("mesh_path")
                if not mesh_path or not Path(mesh_path).exists():
                    continue
                
                # Load mesh
                mesh = trimesh.load(mesh_path)
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                
                # Apply transform if available
                transform = obj.get("transform", {})
                position = transform.get("position", [0, 0, 0])
                
                # Create transformation matrix
                T = np.eye(4)
                T[:3, 3] = position
                
                # Add to scene with object name
                mask_id = obj.get("mask_id", "object")
                scene.add_geometry(mesh, node_name=mask_id, transform=T)
            
            if len(scene.geometry) == 0:
                logger.warning("No meshes to visualize")
                return
            
            # Try to render the scene
            try:
                # Get scene bounds for camera positioning
                bounds = scene.bounds
                center = (bounds[0] + bounds[1]) / 2
                extent = np.max(bounds[1] - bounds[0])
                
                # Set up camera - isometric view from above
                camera_distance = extent * 2.5
                camera_pos = center + np.array([camera_distance * 0.7, -camera_distance * 0.7, camera_distance * 0.8])
                
                # Create camera transform looking at center
                scene.set_camera(
                    angles=[np.pi/6, 0, np.pi/4],  # pitch, roll, yaw
                    distance=camera_distance,
                    center=center
                )
                
                # Render to image
                try:
                    # Try pyrender/pyglet rendering
                    png_data = scene.save_image(resolution=[1024, 768], visible=False)
                    
                    if png_data:
                        viz_path = stage_dir / "scene_preview.png"
                        with open(viz_path, 'wb') as f:
                            f.write(png_data)
                        logger.info(f"Generated scene visualization: {viz_path}")
                        return
                except Exception as render_err:
                    logger.debug(f"Primary render failed: {render_err}")
                
                # Fallback: save a simple top-down 2D projection plot
                self._generate_2d_projection(composition_result, stage_dir)
                
            except Exception as cam_err:
                logger.debug(f"Camera setup failed: {cam_err}")
                self._generate_2d_projection(composition_result, stage_dir)
                
        except ImportError as e:
            logger.warning(f"Cannot generate visualization (missing dependency): {e}")
        except Exception as e:
            logger.warning(f"Scene visualization failed: {e}")
    
    def _generate_2d_projection(self, composition_result: Dict[str, Any], stage_dir: Path) -> None:
        """Generate a simple 2D top-down projection of the scene using actual transforms."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle, Circle
            import numpy as np
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            
            colors = plt.cm.tab10.colors
            objects = composition_result.get("objects", [])
            
            for idx, obj in enumerate(objects):
                transform = obj.get('transform', {})
                position = transform.get('position', [0, 0, 0])
                scale = transform.get('scale', [1, 1, 1])[0]  # Uniform scale
                label = obj.get('label', obj.get('mask_id', f'obj_{idx}'))
                
                # Get mesh dimensions (bbox_dims from reconstruction)
                bbox_dims = obj.get('bbox_dims', [0.1, 0.1, 0.1])
                
                # Calculate scaled dimensions
                width = bbox_dims[0] * scale  # X dimension
                depth = bbox_dims[2] * scale  # Z dimension (shown as Y in top-down)
                
                # Position is center, so calculate corners
                x = position[0]  # X position
                z = position[2]  # Z position (depth in 3D, Y in top-down view)
                
                x_min = x - width / 2
                z_min = z - depth / 2
                
                color = colors[idx % len(colors)]
                rect = Rectangle(
                    (x_min, z_min), 
                    width, 
                    depth,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=(*color[:3], 0.3),
                    label=f"{label[:12]} ({scale:.2f})"
                )
                ax.add_patch(rect)
                
                # Add center marker
                ax.plot(x, z, 'o', color=color, markersize=4)
                
                # Add label at center
                short_label = label[:10] if len(label) > 10 else label
                ax.text(x, z, short_label, ha='center', va='center', fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            ax.set_aspect('equal')
            ax.autoscale()
            
            # Add some padding
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            padding = 0.1
            ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
            ax.set_ylim(ylim[0] - padding, ylim[1] + padding)
            
            ax.set_xlabel('X (meters)')
            ax.set_ylabel('Z / Depth (meters)')
            ax.set_title('Scene Layout (Top-Down View)\nObjects positioned by 2D centroids')
            ax.legend(loc='upper right', fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)
            
            # Add scale reference
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
            
            viz_path = stage_dir / "scene_layout.png"
            plt.savefig(viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Generated 2D scene layout: {viz_path}")
            
        except Exception as e:
            logger.warning(f"2D projection failed: {e}")

