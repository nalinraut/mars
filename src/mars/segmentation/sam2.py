import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import SegmentationModel

logger = logging.getLogger(__name__)

class Sam2Segmentation(SegmentationModel):
    """
    Segmentation using Meta's SAM 2.1 (Segment Anything Model 2).
    Uses Hugging Face Transformers >= 4.45.0 for native SAM 2 support.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.is_sam2 = False  # Track which SAM version is loaded

    def load_model(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """Load SAM 2.1 model using Hugging Face Transformers."""
        
        try:
            import torch
            from transformers import SamModel, SamProcessor, Sam2Model, Sam2Processor
            
            # Check transformers version
            import transformers
            version_parts = transformers.__version__.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            # Use GPU if available, fallback to CPU
            if device == "cuda" and torch.cuda.is_available():
                try:
                    # Test CUDA initialization
                    torch.zeros(1).cuda()
                    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                    self.device = "cuda"
                except Exception as cuda_error:
                    logger.warning(f"CUDA initialization failed: {cuda_error}, falling back to CPU")
                    device = "cpu"
                    self.device = "cpu"
            else:
                logger.info("Using CPU mode (CUDA not available or not requested)")
                device = "cpu"
                self.device = "cpu"
            
            # Get model name from config
            model_config = self.config.get('segmentation', {}).get('model', {})
            requested_model = model_config.get('huggingface_model', 'facebook/sam2.1-hiera-large')
            
            # Try SAM 2.1 first if transformers version supports it
            use_sam2 = major > 4 or (major == 4 and minor >= 45)
            
            if use_sam2 and 'sam2' in requested_model.lower():
                try:
                    logger.info(f"Attempting to load SAM 2.1: {requested_model}")
                    self.processor = Sam2Processor.from_pretrained(requested_model)
                    self.model = Sam2Model.from_pretrained(requested_model).to(self.device)
                    self.model.eval()
                    self.is_sam2 = True
                    logger.info(f"SAM 2.1 loaded successfully on {self.device}")
                    
                except Exception as sam2_error:
                    logger.warning(f"SAM 2.1 load failed: {sam2_error}")
                    logger.info("Falling back to SAM 1.0")
                    use_sam2 = False
            
            # Fallback to SAM 1.0 if SAM 2 not requested or failed
            if not use_sam2 or 'sam2' not in requested_model.lower():
                model_name = "facebook/sam-vit-huge" if not use_sam2 else requested_model
                logger.info(f"Loading SAM 1.0: {model_name}")
                self.processor = SamProcessor.from_pretrained(model_name)
                self.model = SamModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                self.is_sam2 = False
                logger.info(f"SAM 1.0 loaded successfully on {self.device}")
                
        except ImportError as e:
            raise RuntimeError(
                f"Transformers library is required. "
                f"Please install: pip install 'transformers>=4.47.0'\nError: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load SAM model: {e}")

    def generate_masks(self, image_path: str, config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate segmentation masks using SAM's automatic mask generation.
        
        For SAM 2.1, this uses a grid-based point sampling approach since
        the transformers implementation doesn't yet have a dedicated AMG pipeline.
        
        Args:
            image_path: Path to input image
            config: Optional configuration overrides
            
        Returns:
            List of mask dictionaries with keys: segmentation, bbox, area, stability_score
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        from PIL import Image
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)
        h, w = image_np.shape[:2]
        
        logger.info(f"Generating masks for image: {image_path} ({image.size})")
        
        # Get processing config
        processing_config = config or self.config.get('segmentation', {}).get('processing', {})
        
        try:
            # Grid-based automatic mask generation
            points_per_side = processing_config.get('points_per_side', 32)
            
            # Create grid of sample points
            grid_points = []
            for i in range(points_per_side):
                for j in range(points_per_side):
                    x = int((j + 0.5) * w / points_per_side)
                    y = int((i + 0.5) * h / points_per_side)
                    grid_points.append([[x, y]])
            
            logger.info(f"Sampling {len(grid_points)} grid points for mask generation")
            
            # Process points ONE AT A TIME to avoid batching issues
            # This is slower but works reliably with both SAM 1.0 and SAM 2
            all_masks = []
            
            logger.info(f"Processing {len(grid_points)} points individually (this may take a while)...")
            
            for point_idx, point in enumerate(grid_points):
                if point_idx % 100 == 0:
                    logger.info(f"Progress: {point_idx}/{len(grid_points)} points processed, {len(all_masks)} masks found")
                
                try:
                    # Process single point
                    inputs = self.processor(
                        image,
                        input_points=[point],  # Single point [[x, y]]
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Generate masks for this point
                    with torch.no_grad():
                        outputs = self.model(**inputs)
                    
                    # Extract predictions
                    pred_masks = outputs.pred_masks.squeeze(0).cpu().numpy()  # (num_pred, H, W)
                    iou_scores = outputs.iou_scores.squeeze(0).cpu().numpy()  # (num_pred,)
                    
                    # Take the best prediction for this point
                    best_idx = np.argmax(iou_scores)
                    mask = pred_masks[best_idx]
                    score = float(iou_scores[best_idx])
                    
                    # Threshold mask
                    mask_binary = (mask > 0.0).astype(np.uint8)
                    
                    # Calculate area
                    area = int(np.sum(mask_binary))
                    
                    # Skip tiny masks
                    if area < 100:
                        continue
                    
                    # Calculate bbox
                    rows = np.any(mask_binary, axis=1)
                    cols = np.any(mask_binary, axis=0)
                    if not np.any(rows) or not np.any(cols):
                        continue
                    
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
                    
                    # Calculate stability (mask solidity)
                    bbox_area = bbox[2] * bbox[3]
                    stability = float(area / bbox_area) if bbox_area > 0 else 0.5
                    
                    all_masks.append({
                        'segmentation': mask_binary,
                        'bbox': bbox,
                        'area': area,
                        'predicted_iou': score,
                        'stability_score': stability,
                        'crop_box': bbox
                    })
                    
                except Exception as e:
                    # Skip problematic points
                    if point_idx < 5:  # Only log first few failures
                        logger.warning(f"Point {point_idx} failed: {e}")
                    continue
            
            logger.info(f"Generated {len(all_masks)} candidate masks from {len(grid_points)} sample points")
            
            if len(all_masks) < 10:
                logger.warning(f"Very few masks generated! This usually indicates:")
                logger.warning(f"  - Too strict filtering (area < 100, etc.)")
                logger.warning(f"  - All points landing on same object")
                logger.warning(f"  - Model not producing diverse masks")
            
            # Post-processing: NMS to remove overlapping masks
            logger.info(f"Applying NMS with IoU threshold: 0.7")
            masks = self._apply_nms(all_masks, iou_threshold=0.7)
            logger.info(f"After NMS: {len(masks)} masks")
            
            # Filter by area and stability
            min_area = processing_config.get('min_mask_region_area', 100)
            min_stability = processing_config.get('stability_score_thresh', 0.85)
            
            filtered_masks = []
            for mask_dict in masks:
                if (mask_dict['area'] >= min_area and 
                    mask_dict['stability_score'] >= min_stability):
                    filtered_masks.append(mask_dict)
            
            logger.info(f"After NMS and filtering: {len(filtered_masks)} masks")
            
            # Sort by area (largest first)
            filtered_masks = sorted(filtered_masks, key=lambda x: x['area'], reverse=True)
            
            return filtered_masks
            
        except Exception as e:
            logger.error(f"Mask generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to generate masks: {e}")
    
    def _apply_nms(self, masks: List[Dict], iou_threshold: float = 0.7) -> List[Dict]:
        """Apply Non-Maximum Suppression to remove overlapping masks."""
        if len(masks) == 0:
            return []
        
        # Sort by stability score (descending)
        sorted_masks = sorted(masks, key=lambda x: x['stability_score'], reverse=True)
        
        keep_masks = []
        for current_mask in sorted_masks:
            # Check overlap with already kept masks
            should_keep = True
            current_seg = current_mask['segmentation']
            
            for kept_mask in keep_masks:
                kept_seg = kept_mask['segmentation']
                
                # Compute IoU
                intersection = np.logical_and(current_seg, kept_seg).sum()
                union = np.logical_or(current_seg, kept_seg).sum()
                
                if union > 0:
                    iou = intersection / union
                    if iou > iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep_masks.append(current_mask)
        
        return keep_masks
