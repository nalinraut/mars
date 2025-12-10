import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from scipy.ndimage import binary_dilation
import cv2

from prefect import task, flow, get_run_logger
from omegaconf import OmegaConf
from PIL import Image

from .base import SegmentationModel
from .factory import get_segmentation_model

logger = logging.getLogger(__name__)

@dataclass
class MaskMetadata:
    mask_id: str
    category: str
    object_type: str  # interactive, static, ignore
    confidence: float
    bbox: List[int]  # [x_min, y_min, x_max, y_max]
    area: int
    centroid: List[int] # [x, y]
    stability_score: float

@task
def load_segmentation_config(config_path: str = "config/segmentation.yaml") -> Dict[str, Any]:
    """Load segmentation configuration."""
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)

@task
def initialize_segmentation_model(config: Dict[str, Any]) -> SegmentationModel:
    """Factory task to create and load the segmentation model."""
    logger = get_run_logger()
    
    # Get model name from config (defaults to "sam2")
    model_name = config.get('segmentation', {}).get('model', {}).get('type', 'sam2')
    logger.info(f"Initializing segmentation model: {model_name}")
    
    model = get_segmentation_model(model_name, config)
    
    # Get model-specific config
    model_config = config.get('segmentation', {}).get('model', {})
    checkpoint = model_config.get('checkpoint')
    device = model_config.get('device', 'cuda')
    
    model.load_model(checkpoint_path=checkpoint, device=device)
    return model

@task
def generate_masks_task(
    segmentation_model: SegmentationModel,
    image_path: str,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Generate masks using the segmentation model."""
    return segmentation_model.generate_masks(image_path, config)

@task
def filter_and_classify_masks(
    raw_masks: List[Dict[str, Any]], 
    config: Dict[str, Any],
    image_shape: Tuple[int, int]
) -> Tuple[List[MaskMetadata], List[Dict[str, Any]], Dict[str, float]]:
    """
    Filter masks by confidence/area and classify them.
    Returns both processed masks AND their corresponding raw masks to keep them in sync.
    """
    logger = get_run_logger()
    processed_masks = []
    filtered_raw_masks = []  # Keep raw masks in sync with processed masks
    
    # Get thresholds from config (use get() with defaults for robustness)
    processing_config = config['segmentation']['processing']
    classification_config = config['segmentation'].get('classification', {})
    
    conf_threshold = processing_config.get('pred_iou_thresh', 0.88)
    min_area_ratio = classification_config.get('min_area_ratio', 0.001)
    max_area_ratio = processing_config.get('max_area_ratio', 0.50)  # Filter out background masks >50% of image
    total_pixels = image_shape[0] * image_shape[1]
    
    # Categories lists (optional, use empty sets if not defined)
    interactive_cats = set(classification_config.get('interactive_categories', []))
    static_cats = set(classification_config.get('static_categories', []))
    
    scores = []
    
    logger.info(f"Filtering {len(raw_masks)} raw masks with thresholds: IoU={conf_threshold}, area_ratio={min_area_ratio}")
    
    for i, mask_data in enumerate(raw_masks):
        # SAM output keys: segmentation, area, bbox, predicted_iou, point_coords, stability_score, crop_box
        
        score = mask_data['predicted_iou']
        scores.append(score)
        
        # Debug first few masks
        if i < 3:
            logger.info(f"Mask {i}: score={score:.3f}, area={mask_data['area']}")
        
        # Filter by confidence (IoU)
        if score < conf_threshold:
            continue
            
        # Filter by area - both min and max
        area = mask_data['area']
        area_ratio = area / total_pixels
        if area_ratio < min_area_ratio:
            continue
        if area_ratio > max_area_ratio:
            logger.info(f"  Skipping mask {i}: area ratio {area_ratio:.1%} > {max_area_ratio:.1%} (likely background)")
            continue
            
        # Determine Category & Type
        category = "unknown" 
        obj_type = "interactive"
        
        # Calculate Centroid
        bbox = mask_data['bbox'] # [x, y, w, h]
        cx = bbox[0] + bbox[2]/2
        cy = bbox[1] + bbox[3]/2
        
        meta = MaskMetadata(
            mask_id=f"mask_{i}",
            category=category,
            object_type=obj_type,
            confidence=score,
            bbox=[bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]], # convert to xyxy
            area=area,
            centroid=[int(cx), int(cy)],
            stability_score=mask_data['stability_score']
        )
        processed_masks.append(meta)
        filtered_raw_masks.append(mask_data)  # Keep raw mask in sync!

    # Compute Metrics
    metrics = {
        "avg_confidence": float(np.mean(scores)) if scores else 0.0,
        "mask_count": len(processed_masks),
    }
    
    logger.info(f"Filtered to {len(processed_masks)} high-quality masks.")
    return processed_masks, filtered_raw_masks, metrics


@task
def split_disconnected_instances(
    processed_masks: List[MaskMetadata],
    raw_masks: List[Dict[str, Any]],
    min_component_area: int = 500,
    min_component_ratio: float = 0.1
) -> Tuple[List[MaskMetadata], List[Dict[str, Any]]]:
    """
    Split masks with multiple disconnected regions into separate instances.
    
    This handles cases like "blue wooden blocks" where one detection covers
    2 separate blocks - each block becomes its own mask/mesh.
    
    Args:
        processed_masks: List of mask metadata
        raw_masks: Corresponding raw mask data with binary masks
        min_component_area: Minimum pixels for a component to be kept
        min_component_ratio: Min ratio of component area to total mask area
        
    Returns:
        Expanded lists with split masks
    """
    logger = get_run_logger()
    logger.info(f"Checking {len(processed_masks)} masks for multi-instance splitting")
    
    new_masks = []
    new_raw = []
    
    for idx, (meta, raw) in enumerate(zip(processed_masks, raw_masks)):
        # Get binary mask
        if isinstance(raw['segmentation'], dict):
            from pycocotools import mask as mask_utils
            binary_mask = mask_utils.decode(raw['segmentation'])
        else:
            binary_mask = raw['segmentation'].astype(np.uint8)
        
        # Ensure binary (0 or 1)
        if binary_mask.max() > 1:
            binary_mask = (binary_mask > 127).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        
        # Label 0 is background, so actual components start at 1
        num_components = num_labels - 1
        
        if num_components <= 1:
            # Single component - keep as is
            new_masks.append(meta)
            new_raw.append(raw)
            continue
        
        # Multiple components - split them
        total_area = binary_mask.sum()
        components_kept = 0
        
        for comp_id in range(1, num_labels):
            comp_area = stats[comp_id, cv2.CC_STAT_AREA]
            
            # Filter small components
            if comp_area < min_component_area:
                continue
            if comp_area / total_area < min_component_ratio:
                continue
            
            # Extract this component
            comp_mask = (labels == comp_id).astype(np.uint8)
            
            # Compute new bbox and centroid
            x, y, w, h = stats[comp_id, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            cx, cy = centroids[comp_id]
            
            # Create new metadata with unique ID
            suffix = chr(ord('a') + components_kept)  # a, b, c, ...
            new_meta = MaskMetadata(
                mask_id=f"{meta.mask_id}_{suffix}",
                category=meta.category,
                object_type=meta.object_type,
                confidence=meta.confidence,
                bbox=[int(x), int(y), int(x + w), int(y + h)],
                area=int(comp_area),
                centroid=[int(cx), int(cy)],
                stability_score=meta.stability_score
            )
            
            # Create new raw mask data
            new_raw_data = raw.copy()
            new_raw_data['segmentation'] = comp_mask
            new_raw_data['area'] = comp_area
            new_raw_data['bbox'] = [x, y, w, h]
            
            new_masks.append(new_meta)
            new_raw.append(new_raw_data)
            components_kept += 1
        
        if components_kept > 1:
            logger.info(f"  Split {meta.mask_id} into {components_kept} instances")
    
    logger.info(f"Instance splitting: {len(processed_masks)} -> {len(new_masks)} masks")
    return new_masks, new_raw


@task
def deduplicate_overlapping_masks(
    processed_masks: List[MaskMetadata],
    raw_masks: List[Dict[str, Any]],
    containment_threshold: float = 0.7,
    iou_threshold: float = 0.5
) -> Tuple[List[MaskMetadata], List[Dict[str, Any]]]:
    """
    Remove duplicate masks using two criteria:
    1. Containment: If mask A is largely contained within mask B
    2. IoU: If two masks have high IoU (same object detected twice)
    
    Args:
        processed_masks: List of filtered mask metadata
        raw_masks: Corresponding raw mask data with binary masks
        containment_threshold: If mask A overlaps with B by this ratio of A's area, A is a subset
        iou_threshold: If IoU between two masks exceeds this, keep only the higher-confidence one
        
    Returns:
        Deduplicated list of masks
    """
    logger = get_run_logger()
    
    if len(processed_masks) <= 1:
        return processed_masks, raw_masks
    
    logger.info(f"Deduplicating {len(processed_masks)} masks (containment={containment_threshold}, iou={iou_threshold})")
    
    # Create binary mask images
    mask_images = []
    for mask_data in raw_masks:
        if isinstance(mask_data['segmentation'], dict):
            from pycocotools import mask as mask_utils
            mask_img = mask_utils.decode(mask_data['segmentation'])
        else:
            mask_img = mask_data['segmentation'].astype(np.uint8)
        mask_images.append(mask_img)
    
    n = len(processed_masks)
    to_remove = set()
    
    # Check each pair of masks
    for i in range(n):
        if i in to_remove:
            continue
            
        for j in range(n):
            if i == j or j in to_remove:
                continue
            
            mask_i = mask_images[i]
            mask_j = mask_images[j]
            
            # Calculate overlap metrics
            intersection = np.logical_and(mask_i, mask_j).sum()
            area_i = mask_i.sum()
            area_j = mask_j.sum()
            
            if area_j == 0 or area_i == 0:
                continue
            
            # Calculate IoU
            union = np.logical_or(mask_i, mask_j).sum()
            iou = intersection / union if union > 0 else 0
            
            # Calculate containment ratio
            containment_ratio = intersection / area_j
            
            # Check IoU first - if high IoU, these are likely the same object
            if iou > iou_threshold:
                # Keep the one with higher confidence score
                score_i = processed_masks[i].confidence
                score_j = processed_masks[j].confidence
                if score_j < score_i:
                    to_remove.add(j)
                    logger.info(f"  Removing {processed_masks[j].mask_id} (IoU={iou:.2f} with {processed_masks[i].mask_id}, lower score)")
                else:
                    to_remove.add(i)
                    logger.info(f"  Removing {processed_masks[i].mask_id} (IoU={iou:.2f} with {processed_masks[j].mask_id}, lower score)")
                    break
            # Then check containment
            elif containment_ratio > containment_threshold:
                if area_j < area_i:
                    # j is smaller and contained in i -> remove j
                    to_remove.add(j)
                    logger.info(f"  Removing {processed_masks[j].mask_id} (contained in {processed_masks[i].mask_id}, {containment_ratio:.1%} overlap)")
                else:
                    # i is smaller and contained in j -> remove i
                    to_remove.add(i)
                    logger.info(f"  Removing {processed_masks[i].mask_id} (contained in {processed_masks[j].mask_id}, {containment_ratio:.1%} overlap)")
                    break
    
    # Filter out removed masks
    kept_masks = []
    kept_raw = []
    for idx in range(n):
        if idx not in to_remove:
            kept_masks.append(processed_masks[idx])
            kept_raw.append(raw_masks[idx])
    
    logger.info(f"Deduplication: {len(processed_masks)} -> {len(kept_masks)} masks ({len(to_remove)} removed)")
    
    return kept_masks, kept_raw


@task
def merge_related_masks(
    processed_masks: List[MaskMetadata],
    raw_masks: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    iou_threshold: float = 0.3,
    distance_threshold: float = 50
) -> Tuple[List[MaskMetadata], List[Dict[str, Any]]]:
    """
    Merge masks that are spatially close or overlapping.
    This combines fragmented parts (like stool legs, cup handle) into complete objects.
    
    Args:
        processed_masks: List of filtered mask metadata
        raw_masks: Corresponding raw mask data with binary masks
        image_shape: (height, width) of the image
        iou_threshold: IoU threshold for considering masks as related
        distance_threshold: Max distance (pixels) between mask centroids to consider merging
        
    Returns:
        Merged list of masks and their raw data
    """
    logger = get_run_logger()
    
    if len(processed_masks) == 0:
        return processed_masks, raw_masks
        
    logger.info(f"Merging {len(processed_masks)} masks (IoU threshold={iou_threshold}, distance={distance_threshold}px)")
    
    # Create binary mask images for each mask
    mask_images = []
    for mask_data in raw_masks:
        if isinstance(mask_data['segmentation'], dict):
            # RLE format (from SAM)
            from pycocotools import mask as mask_utils
            mask_img = mask_utils.decode(mask_data['segmentation'])
        else:
            # Already binary mask
            mask_img = mask_data['segmentation'].astype(np.uint8)
        mask_images.append(mask_img)
    
    # Build adjacency graph of overlapping/close masks
    n = len(processed_masks)
    merged = [False] * n
    merge_groups = []
    
    for i in range(n):
        if merged[i]:
            continue
            
        # Start a new group with mask i
        group = [i]
        merged[i] = True
        
        # Find all masks that should merge with i
        for j in range(i+1, n):
            if merged[j]:
                continue
                
            # Check IoU
            mask_i = mask_images[i]
            mask_j = mask_images[j]
            intersection = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            iou = intersection / union if union > 0 else 0
            
            # Check centroid distance
            cx_i, cy_i = processed_masks[i].centroid
            cx_j, cy_j = processed_masks[j].centroid
            distance = np.sqrt((cx_i - cx_j)**2 + (cy_i - cy_j)**2)
            
            # Merge if overlapping or close
            if iou > iou_threshold or distance < distance_threshold:
                group.append(j)
                merged[j] = True
        
        merge_groups.append(group)
    
    # Create merged masks
    merged_masks_meta = []
    merged_masks_raw = []
    
    for group_idx, group in enumerate(merge_groups):
        if len(group) == 1:
            # No merge needed, keep original
            idx = group[0]
            merged_masks_meta.append(processed_masks[idx])
            merged_masks_raw.append(raw_masks[idx])
        else:
            # Merge multiple masks
            merged_mask = np.zeros(image_shape, dtype=np.uint8)
            total_area = 0
            max_score = 0
            max_stability = 0
            
            for idx in group:
                merged_mask = np.logical_or(merged_mask, mask_images[idx])
                total_area += processed_masks[idx].area
                max_score = max(max_score, processed_masks[idx].confidence)
                max_stability = max(max_stability, processed_masks[idx].stability_score)
            
            merged_mask = merged_mask.astype(np.uint8)
            
            # Compute new bbox and centroid
            ys, xs = np.where(merged_mask > 0)
            if len(xs) == 0:
                continue
                
            x_min, x_max = int(xs.min()), int(xs.max())
            y_min, y_max = int(ys.min()), int(ys.max())
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)
            actual_area = int(merged_mask.sum())
            
            # Create merged metadata
            meta = MaskMetadata(
                mask_id=f"merged_{group_idx}",
                category="unknown",
                object_type="interactive",
                confidence=float(max_score),
                bbox=[x_min, y_min, x_max, y_max],
                area=actual_area,
                centroid=[cx, cy],
                stability_score=float(max_stability)
            )
            merged_masks_meta.append(meta)
            
            # Create merged raw mask
            raw_merged = {
                'segmentation': merged_mask,
                'area': actual_area,
                'bbox': [x_min, y_min, x_max - x_min, y_max - y_min],
                'predicted_iou': max_score,
                'stability_score': max_stability
            }
            merged_masks_raw.append(raw_merged)
    
    logger.info(f"Merged {len(processed_masks)} masks into {len(merged_masks_meta)} complete objects")
    logger.info(f"  - {len([g for g in merge_groups if len(g) > 1])} groups merged")
    logger.info(f"  - {len([g for g in merge_groups if len(g) == 1])} masks kept as-is")
    
    return merged_masks_meta, merged_masks_raw


@task
def save_masks(
    processed_masks: List[MaskMetadata],
    raw_masks: List[Dict[str, Any]],
    output_dir: str,
    image_id: str
) -> List[Dict[str, Any]]:
    """
    Save binary masks to disk and return serializable records.
    """
    out_path = Path(output_dir) / image_id / "masks"
    out_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # After merging, raw_masks and processed_masks are aligned by index
    for idx, meta in enumerate(processed_masks):
        binary_mask = raw_masks[idx]['segmentation']
        
        # Save as PNG
        mask_filename = f"{meta.mask_id}.png"
        mask_path = out_path / mask_filename
        
        # Handle both 0/1 and 0/255 masks
        if binary_mask.max() <= 1:
            # Boolean or 0/1 mask - scale to 0-255
            mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8))
        else:
            # Already 0-255 (from SAM3)
            mask_img = Image.fromarray(binary_mask.astype(np.uint8))
        mask_img.save(mask_path)
        
        # Create record
        record = asdict(meta)
        record['mask_path'] = str(mask_path)
        results.append(record)
        
    return results

@flow(name="Segment Scene Objects", retries=2, retry_delay_seconds=5)
def segment_image(
    image_path: str,
    image_id: str,
    config_path: str = "config/segmentation.yaml",
    output_dir: str = "data/processed",
    text_prompts: List[str] = None,
) -> Dict[str, Any]:
    """
    Main segmentation flow using SAM 3 with text prompts.
    
    Args:
        image_path: Path to image file
        image_id: Unique identifier for this image
        config_path: Path to segmentation config
        output_dir: Directory for output files
        text_prompts: Optional list of text prompts from detection (e.g., ["coffee cup", "saucer"])
    """
    logger = get_run_logger()
    logger.info(f"Starting segmentation for {image_id}")
    
    # 1. Load Config
    config = load_segmentation_config(config_path)
    
    # Use text prompts with SAM 3
    if text_prompts:
        # Apply label mapping from config
        label_mapping = config.get('segmentation', {}).get('label_mapping', {})
        
        mapped_prompts = []
        for prompt in text_prompts:
            # Check if label needs mapping
            mapped = label_mapping.get(prompt.lower(), prompt)
            if mapped is None:
                logger.info(f"  Skipping '{prompt}' (mapped to null)")
                continue
            if mapped != prompt:
                logger.info(f"  Mapping '{prompt}' → '{mapped}'")
            mapped_prompts.append(mapped)
        
        if mapped_prompts:
            logger.info(f"Using text prompts for SAM 3: {mapped_prompts}")
            if 'segmentation' not in config:
                config['segmentation'] = {}
            if 'prompts' not in config['segmentation']:
                config['segmentation']['prompts'] = {}
            config['segmentation']['prompts']['text_prompts'] = mapped_prompts
            config['segmentation']['model'] = {'type': 'sam3'}
        else:
            logger.warning("All prompts were filtered out by label mapping")
    
    # 2. Initialize Model (using factory)
    try:
        segmentation_model = initialize_segmentation_model(config)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        raise e  # Trigger Prefect retry
        
    # 3. Generate Masks
    # Check image dimensions
    with Image.open(image_path) as img:
        w, h = img.size
        
    try:
        raw_masks = generate_masks_task(segmentation_model, image_path, config)
    except Exception as e:
        logger.error(f"Mask generation failed: {e}")
        return {"status": "failed", "error": str(e)}
        
    if not raw_masks:
        logger.warning("No masks found. Marking scene as empty.")
        return {
            "status": "empty",
            "image_id": image_id,
            "quality_score": 0.0,
            "masks": []
        }
        
    # 4. Filter & Classify
    filtered_masks, filtered_raw, metrics = filter_and_classify_masks(raw_masks, config, (h, w))
    
    # 5. Deduplicate overlapping masks (remove masks that are subsets of larger masks)
    dedup_config = config.get('segmentation', {}).get('deduplication', {})
    dedup_enabled = dedup_config.get('enabled', True)  # Enabled by default
    
    if dedup_enabled and len(filtered_masks) > 1:
        containment_thresh = dedup_config.get('containment_threshold', 0.7)
        iou_thresh = dedup_config.get('iou_threshold', 0.5)
        deduped_masks, deduped_raw = deduplicate_overlapping_masks(
            filtered_masks,
            filtered_raw,  # Use the synced raw masks, not a slice!
            containment_threshold=containment_thresh,
            iou_threshold=iou_thresh
        )
    else:
        deduped_masks = filtered_masks
        deduped_raw = filtered_raw  # Use the synced raw masks!
    
    # 5.5. Split disconnected instances (e.g., "2 blue blocks" in one mask -> 2 separate masks)
    instance_config = config.get('segmentation', {}).get('instance_splitting', {})
    splitting_enabled = instance_config.get('enabled', True)  # Enabled by default
    
    if splitting_enabled and len(deduped_masks) > 0:
        min_comp_area = instance_config.get('min_component_area', 500)
        min_comp_ratio = instance_config.get('min_component_ratio', 0.1)
        split_masks, split_raw = split_disconnected_instances(
            deduped_masks,
            deduped_raw,
            min_component_area=min_comp_area,
            min_component_ratio=min_comp_ratio
        )
    else:
        split_masks = deduped_masks
        split_raw = deduped_raw
    
    # 6. Merge Related Masks (optional - SAM 3 generates complete objects, so merging often not needed)
    merging_config = config.get('segmentation', {}).get('merging', {})
    merge_enabled = merging_config.get('enabled', False)
    
    if merge_enabled and len(split_masks) > 0:
        logger.info("Merging enabled - combining spatially close masks")
        iou_thresh = merging_config.get('iou_threshold', 0.05)
        dist_thresh = merging_config.get('distance_threshold', 200)
        keep_top_n = merging_config.get('keep_top_n', 0)  # 0 = keep all
        
        final_masks, final_raw_masks = merge_related_masks(
            split_masks, 
            split_raw,
            (h, w),
            iou_threshold=iou_thresh,
            distance_threshold=dist_thresh
        )
        
        # Apply top-N filter if configured
        if keep_top_n > 0 and len(final_masks) > keep_top_n:
            sorted_indices = sorted(range(len(final_masks)), 
                                  key=lambda i: final_masks[i].area, 
                                  reverse=True)
            final_masks = [final_masks[i] for i in sorted_indices[:keep_top_n]]
            final_raw_masks = [final_raw_masks[i] for i in sorted_indices[:keep_top_n]]
            logger.info(f"Kept top {keep_top_n} largest objects after merging")
    else:
        logger.info("Merging disabled - using split masks")
        final_masks = split_masks
        final_raw_masks = split_raw
    
    # 6.5 Post-split filter: remove any masks still too large (background pieces)
    processing_config = config.get('segmentation', {}).get('processing', {})
    max_area_ratio = processing_config.get('max_area_ratio', 0.35)
    total_pixels = h * w
    
    filtered_final = []
    filtered_final_raw = []
    for mask_meta, mask_raw in zip(final_masks, final_raw_masks):
        area_ratio = mask_meta.area / total_pixels
        if area_ratio > max_area_ratio:
            logger.info(f"  Post-split filter: removing {mask_meta.mask_id} (area ratio {area_ratio:.1%} > {max_area_ratio:.1%})")
            continue
        filtered_final.append(mask_meta)
        filtered_final_raw.append(mask_raw)
    
    if len(filtered_final) < len(final_masks):
        logger.info(f"Post-split filter: {len(final_masks)} -> {len(filtered_final)} masks")
        final_masks = filtered_final
        final_raw_masks = filtered_final_raw
    
    # 7. Save Results
    saved_masks = save_masks(final_masks, final_raw_masks, output_dir, image_id)
    
    result = {
        "status": "success",
        "image_id": image_id,
        "masks": saved_masks,
        "metrics": metrics,
        "quality_score": metrics['avg_confidence'],
        "image_size": (int(w), int(h)),  # (width, height) for downstream stages
    }
    
    return result

if __name__ == "__main__":
    # Test run
    import sys
    if len(sys.argv) > 2:
        segment_image(sys.argv[1], sys.argv[2])
