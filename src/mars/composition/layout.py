"""
Layout Engine for 3D scene composition.

Approach:
1. Map image X directly to scene X (left-to-right preserved)
2. Use Depth Anything V2 for scene Z positioning (depth-based)
3. Fallback: Map image Y to scene Z if depth unavailable
4. Use category scales for accurate real-world sizes
5. Detect stacking from same-label objects that overlap
6. Place table first, objects on top
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import trimesh

logger = logging.getLogger(__name__)

# Objects covering >15% of image are background/table
BACKGROUND_AREA_THRESHOLD = 0.15


@dataclass
class Transform:
    position: List[float]  # [x, y, z]
    rotation: List[float]  # [x, y, z, w] quaternion
    scale: List[float]     # [x, y, z]


class LayoutEngine:
    """
    Scene layout engine with depth-based positioning.
    
    Maps 2D image positions to 3D scene positions:
    - Image X → Scene X (left=negative, right=positive)
    - Depth Anything V2 → Scene Z (depth-based positioning)
    - Fallback: Image Y → Scene Z if depth unavailable
    - Objects placed on table surface (Y > 0)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        layout_cfg = config['composition']['layout']
        self.scene_width = layout_cfg.get('scene_width', 0.6)
        self.scene_depth = layout_cfg.get('scene_depth', 0.5)
        self.min_separation = layout_cfg.get('object_separation', 0.02)
        
        scaling_cfg = config['composition'].get('scaling', {})
        self.default_scale = scaling_cfg.get('default_scale', 0.1)
        self.category_scales = scaling_cfg.get('categories', {})
        
        # Depth estimation config
        depth_cfg = config['composition'].get('depth', {})
        self.use_depth = depth_cfg.get('enabled', True)
        self.depth_model = depth_cfg.get('model', 'depth_anything_v2')  # or 'moge'
        self.depth_model_size = depth_cfg.get('model_size', 'large')
        self.depth_metric = depth_cfg.get('metric', False)
        self.depth_scene_type = depth_cfg.get('scene_type', 'indoor')
        
        self._mesh_cache = {}
        self._depth_estimator = None
    
    def _get_depth_estimator(self):
        """Get or create depth estimator based on config."""
        if self._depth_estimator is not None:
            return self._depth_estimator
        
        if not self.use_depth:
            return None
        
        try:
            if self.depth_model == 'depth_anything_v2':
                from .depth_anything import DepthAnythingV2Estimator
                self._depth_estimator = DepthAnythingV2Estimator({
                    'model_size': self.depth_model_size,
                    'metric': self.depth_metric,
                    'scene_type': self.depth_scene_type,
                })
                logger.info(f"Using Depth Anything V2 ({self.depth_model_size}, metric={self.depth_metric})")
            elif self.depth_model == 'moge':
                from .depth import DepthEstimator
                self._depth_estimator = DepthEstimator(self.config)
                logger.info("Using MoGe depth estimator")
            else:
                logger.warning(f"Unknown depth model: {self.depth_model}")
                return None
            
            return self._depth_estimator
            
        except Exception as e:
            logger.warning(f"Failed to load depth estimator: {e}")
            return None
    
    def _get_mesh_extents(self, mesh_path: str) -> np.ndarray:
        """Get mesh bounding box extents."""
        if mesh_path in self._mesh_cache:
            return self._mesh_cache[mesh_path]
        
        extents = np.array([1.0, 1.0, 1.0])
        if mesh_path and Path(mesh_path).exists():
            try:
                mesh = trimesh.load(mesh_path)
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                extents = mesh.extents
                self._mesh_cache[mesh_path] = extents
            except Exception as e:
                logger.warning(f"Could not load mesh {mesh_path}: {e}")
        
        return extents
    
    def _get_category_scale(self, label: str, estimated_size: float = None) -> float:
        """
        Get real-world size for object category.
        
        Priority:
        1. LLM-estimated size from detection (if provided)
        2. Category scale from config (exact match)
        3. Category scale from config (partial match)
        4. Default scale
        """
        # Priority 1: LLM-estimated size from detection
        if estimated_size is not None and 0.01 <= estimated_size <= 5.0:
            return estimated_size
        
        if not label:
            return self.default_scale
        
        label_lower = label.lower().strip()
        
        # Priority 2: Exact match from config
        if label_lower in self.category_scales:
            return self.category_scales[label_lower]
        
        # Priority 3: Partial match from config
        for category, size in self.category_scales.items():
            if category in label_lower:
                return size
        
        # Priority 4: Default
        return self.default_scale
    
    def _is_background(self, obj: Dict, image_size: Tuple[int, int]) -> bool:
        """Check if object is background (table/surface)."""
        area = obj.get('area', 0)
        total_area = image_size[0] * image_size[1]
        return (area / total_area) > BACKGROUND_AREA_THRESHOLD if total_area > 0 else False
    
    def _deduplicate_fragments(
        self, 
        objects: List[Dict], 
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Remove fragmented masks that match the same detection.
        
        SAM sometimes splits one object into multiple masks (e.g., striped socks).
        For each detection, keep only the LARGEST overlapping mask.
        """
        if not detections:
            return objects
        
        # Group masks by which detection they match
        detection_to_masks = {}  # det_idx -> list of (mask, iou)
        unmatched_masks = []
        
        for obj in objects:
            mask_bbox = obj.get('bbox', [0, 0, 100, 100])
            if len(mask_bbox) == 4:
                # Mask bbox is in [x1, y1, x2, y2] format (xyxy)
                mx1, my1, mx2, my2 = [int(float(v)) for v in mask_bbox]
            else:
                unmatched_masks.append(obj)
                continue
            
            best_det_idx = None
            best_iou = 0
            
            for det_idx, det in enumerate(detections):
                det_bbox = det.get('bbox', [0, 0, 0, 0])
                dx1, dy1, dx2, dy2 = [float(x) for x in det_bbox[:4]]
                
                # Calculate IoU
                inter_x1 = max(mx1, dx1)
                inter_y1 = max(my1, dy1)
                inter_x2 = min(mx2, dx2)
                inter_y2 = min(my2, dy2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    mask_area = (mx2 - mx1) * (my2 - my1)
                    det_area = (dx2 - dx1) * (dy2 - dy1)
                    union_area = mask_area + det_area - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    
                    if iou > best_iou and iou > 0.1:
                        best_iou = iou
                        best_det_idx = det_idx
            
            if best_det_idx is not None:
                if best_det_idx not in detection_to_masks:
                    detection_to_masks[best_det_idx] = []
                detection_to_masks[best_det_idx].append((obj, best_iou))
            else:
                unmatched_masks.append(obj)
        
        # For each detection, keep only the largest mask
        deduplicated = []
        for det_idx, mask_list in detection_to_masks.items():
            if len(mask_list) == 1:
                deduplicated.append(mask_list[0][0])
            else:
                # Multiple masks for same detection - keep largest by area
                mask_list.sort(key=lambda x: x[0].get('area', 0), reverse=True)
                largest = mask_list[0][0]
                deduplicated.append(largest)
                
                det_label = detections[det_idx].get('label', 'unknown')
                logger.info(f"Deduplicated {len(mask_list)} fragments for '{det_label}' "
                           f"-> keeping {largest.get('mask_id')} (area={largest.get('area')})")
        
        # Add unmatched masks back
        deduplicated.extend(unmatched_masks)
        
        return deduplicated
    
    def _image_to_scene(
        self, 
        centroid: List[int], 
        image_size: Tuple[int, int],
        depth_value: float = None
    ) -> Tuple[float, float]:
        """
        Convert image coordinates to scene coordinates.
        
        Image: (0,0) = top-left, (W,H) = bottom-right
        Scene: X left-to-right, Z based on depth (or fallback to Y)
        
        Args:
            centroid: [x, y] pixel coordinates
            image_size: (width, height)
            depth_value: Normalized depth (0=front, 1=back) if available
        """
        width, height = image_size
        cx, cy = centroid
        
        # X: left of image = negative X, right = positive X
        x = (cx / width - 0.5) * self.scene_width
        
        if depth_value is not None:
            # Use depth for Z positioning
            # depth_value is 0-1, map to scene depth range
            # 0 (close) -> negative Z (front), 1 (far) -> positive Z (back)
            z = (depth_value - 0.5) * self.scene_depth
        else:
            # Fallback: top of image = back, bottom = front
            z = (0.5 - cy / height) * self.scene_depth
        
        return x, z
    
    def project_to_3d(
        self, 
        objects: List[Dict], 
        image_size: Tuple[int, int],
        camera_config: Dict,
        detections: List[Dict] = None,
        image_path: str = None
    ) -> List[Dict]:
        """
        Project objects from 2D image to 3D scene.
        
        Approach:
        1. Table is placed at origin, flat
        2. Image X → Scene X
        3. Depth Anything V2 → Scene Z (or fallback to Image Y)
        4. Category scales for sizing
        5. Same-label overlapping objects = stacking
        """
        width, height = image_size
        placed_objects = []
        
        # Get depth values for all objects
        depth_values = {}
        if image_path and self.use_depth:
            depth_estimator = self._get_depth_estimator()
            if depth_estimator:
                try:
                    depth_values = depth_estimator.get_depths_for_objects(
                        image_path, objects, image_size
                    )
                    logger.info(f"Got depth values for {len(depth_values)} objects using {self.depth_model}")
                    
                    # Log depth ordering for debugging
                    if depth_values:
                        sorted_by_depth = sorted(depth_values.items(), key=lambda x: x[1])
                        logger.info("Depth ordering (front to back):")
                        for mask_id, d in sorted_by_depth[:5]:  # Show top 5
                            logger.info(f"  {mask_id}: depth={d:.3f}")
                except Exception as e:
                    logger.warning(f"Depth estimation failed: {e}")
        
        # Assign labels to masks using detection-centric approach
        # This must happen BEFORE separating foreground/background
        objects = self._assign_labels_to_masks(objects, detections)
        
        # Separate background (table) from foreground
        table_objects = []
        foreground_objects = []
        
        for obj in objects:
            # Check if this is a background/table object
            is_bg = self._is_background(obj, image_size)
            
            # Also check label - anything with "table" in the name is background
            label = self._get_label(obj, detections).lower()
            is_table_label = 'table' in label
            
            mask_id = obj.get('mask_id', 'unknown')
            
            if is_bg or is_table_label:
                table_objects.append(obj)
                if is_table_label and not is_bg:
                    logger.info(f"Filtering table fragment by label: {mask_id} ({label})")
                elif is_bg:
                    logger.info(f"Background object by area: {mask_id} ({label})")
            else:
                foreground_objects.append(obj)
        
        logger.info(f"Layout: {len(table_objects)} table pieces, {len(foreground_objects)} foreground objects")
        
        # === Place table (largest background piece only) ===
        table_surface_height = 0.02  # 2cm default table height
        
        if table_objects:
            table_objects.sort(key=lambda o: o.get('area', 0), reverse=True)
            table = table_objects[0]
            
            mesh_path = table.get('mesh_path')
            mesh_extents = self._get_mesh_extents(mesh_path)
            mesh_max = max(mesh_extents) if len(mesh_extents) > 0 else 1.0
            
            # Get label from detection (could be "stool", "table", etc.)
            detected_label = self._get_label(table, detections) or 'table'
            
            # Use LLM-estimated size if available, then category scale, then fill scene
            estimated_size = table.get('estimated_size')
            category_size = self._get_category_scale(detected_label, estimated_size)
            
            if estimated_size and 0.01 <= estimated_size <= 5.0:
                # LLM-estimated size
                table_scale = estimated_size / mesh_max
                logger.info(f"Background '{detected_label}': using LLM-estimated size {estimated_size}m")
            elif category_size != self.default_scale:
                # Category scale found - use it
                table_scale = category_size / mesh_max
                logger.info(f"Background '{detected_label}': using category scale {category_size}m")
            else:
                # No specific scale - scale to fill scene (like a table)
                table_scale = max(self.scene_width, self.scene_depth) / mesh_max
                logger.info(f"Background '{detected_label}': scaling to fill scene")
            
            table_height = mesh_extents[1] * table_scale if len(mesh_extents) > 1 else 0.02
            
            table_transform = Transform(
                position=[0.0, table_height / 2, 0.0],
                rotation=[0, 0, 0, 1],
                scale=[table_scale, table_scale, table_scale]
            )
            
            table_obj = table.copy()
            table_obj['transform'] = asdict(table_transform)
            table_obj['label'] = detected_label  # Use actual label, not hardcoded 'table'
            table_obj['is_surface'] = True
            
            # Create labeled object_id for table
            table_mask_id = table.get('mask_id', 'table')
            safe_label = detected_label.lower().replace(' ', '_').replace('-', '_')
            table_obj['object_id'] = f"{safe_label}_{table_mask_id}"
            
            placed_objects.append(table_obj)
            
            table_surface_height = table_height
            logger.info(f"Placed {table_obj['object_id']}: scale={table_scale:.3f}, height={table_height:.3f}m")
        
        # === Detect stacking relationships ===
        # VERTICAL stacking: objects aligned horizontally (similar X) but different Y
        # Side-by-side: objects at similar Y but different X (NOT stacking)
        stacking = {}  # base_mask_id -> [stacked_mask_ids]
        
        logger.info(f"Checking stacking for {len(foreground_objects)} objects...")
        
        for i, obj_a in enumerate(foreground_objects):
            label_a = self._get_label(obj_a, detections)
            bbox_a = [float(x) for x in obj_a.get('bbox', [0, 0, 0, 0])]
            centroid_a = [float(x) for x in obj_a.get('centroid', [0, 0])]
            
            for j, obj_b in enumerate(foreground_objects):
                if i >= j:  # Only check each pair once
                    continue
                
                label_b = self._get_label(obj_b, detections)
                bbox_b = [float(x) for x in obj_b.get('bbox', [0, 0, 0, 0])]
                centroid_b = [float(x) for x in obj_b.get('centroid', [0, 0])]
                
                # Check for similar labels (partial match)
                similar_label = False
                if label_a and label_b:
                    la, lb = label_a.lower(), label_b.lower()
                    similar_label = (la == lb or 
                                    any(word in lb for word in la.split() if len(word) > 3) or
                                    any(word in la for word in lb.split() if len(word) > 3))
                
                if not similar_label:
                    continue
                
                # Check spatial relationship
                dx = abs(centroid_a[0] - centroid_b[0])  # Horizontal distance
                dy = abs(centroid_a[1] - centroid_b[1])  # Vertical distance in image
                
                # Get object sizes
                width_a = abs(bbox_a[2] - bbox_a[0]) if len(bbox_a) >= 4 else 100
                width_b = abs(bbox_b[2] - bbox_b[0]) if len(bbox_b) >= 4 else 100
                height_a = abs(bbox_a[3] - bbox_a[1]) if len(bbox_a) >= 4 else 100
                height_b = abs(bbox_b[3] - bbox_b[1]) if len(bbox_b) >= 4 else 100
                avg_width = (width_a + width_b) / 2
                avg_height = (height_a + height_b) / 2
                
                # STACKING criteria:
                # 1. Horizontally aligned (dx < 0.8 * width) - objects are roughly above/below each other
                # 2. Vertically separated (dy > 0.3 * height) - there's vertical distance
                # 3. Not too far apart (dy < 2 * height)
                horizontally_aligned = dx < avg_width * 0.8
                vertically_separated = dy > avg_height * 0.3
                not_too_far = dy < avg_height * 2.0
                
                is_stacked = horizontally_aligned and vertically_separated and not_too_far
                
                # SIDE-BY-SIDE criteria (NOT stacking):
                # Horizontally separated (dx > width) and similar Y level
                side_by_side = dx > avg_width * 0.8 and dy < avg_height * 0.5
                
                logger.info(f"  {label_a} vs {label_b}: dx={dx:.0f}, dy={dy:.0f}, "
                           f"h_aligned={horizontally_aligned}, v_sep={vertically_separated}, "
                           f"stacked={is_stacked}, side_by_side={side_by_side}")
                
                if is_stacked and not side_by_side:
                    # Determine which is on top (smaller Y = higher in image = physically on top)
                    if centroid_a[1] < centroid_b[1]:
                        # A is higher in image = A is ON TOP of B
                        base_id = obj_b.get('mask_id')
                        stacked_id = obj_a.get('mask_id')
                    else:
                        # B is higher in image = B is ON TOP of A
                        base_id = obj_a.get('mask_id')
                        stacked_id = obj_b.get('mask_id')
                    
                    if base_id not in stacking:
                        stacking[base_id] = []
                    if stacked_id not in stacking[base_id]:
                        stacking[base_id].append(stacked_id)
                        logger.info(f"  => STACKING: {stacked_id} ON TOP of {base_id}")
        
        # NOTE: Composition-level deduplication disabled
        # The segmentation stage already handles duplicate masks (IoU-based)
        # Keeping all masks here since one detection can cover multiple real objects
        # (e.g., "crayons" detection bbox covers 2 separate crayons)
        logger.info(f"Foreground objects: {len(foreground_objects)}")
        
        # === Place foreground objects ===
        # Sort by area (larger = base, smaller = stacked on top)
        foreground_objects.sort(key=lambda o: o.get('area', 0), reverse=True)
        
        object_tops = {}  # mask_id -> top surface height
        
        for obj in foreground_objects:
            mask_id = obj.get('mask_id', 'unknown')
            centroid = [float(x) for x in obj.get('centroid', [width // 2, height // 2])]
            mesh_path = obj.get('mesh_path')
            
            # Get label
            label = self._get_label(obj, detections)
            
            # Get scale from category (with LLM-estimated size if available)
            estimated_size = obj.get('estimated_size')
            category_size = self._get_category_scale(label, estimated_size)
            mesh_extents = self._get_mesh_extents(mesh_path)
            mesh_max = max(mesh_extents) if len(mesh_extents) > 0 else 1.0
            scale_factor = category_size / mesh_max
            scale_factor = max(0.01, min(scale_factor, 1.0))
            
            if estimated_size:
                logger.debug(f"Using LLM-estimated size {estimated_size}m for '{label}'")
            
            scaled_height = mesh_extents[1] * scale_factor if len(mesh_extents) > 1 else category_size
            
            # Position from image coordinates + depth
            obj_depth = depth_values.get(mask_id)  # None if not available
            x, z = self._image_to_scene(centroid, image_size, depth_value=obj_depth)
            
            # Y position: on table or stacked
            base_height = table_surface_height
            
            # Check if stacked on another object
            for base_id, stacked_ids in stacking.items():
                if mask_id in stacked_ids and base_id in object_tops:
                    base_height = object_tops[base_id]
                    logger.info(f"  {mask_id} stacked on {base_id} at {base_height:.3f}m")
                    break
            
            y = base_height + scaled_height / 2
            
            # Record top surface
            object_tops[mask_id] = base_height + scaled_height
            
            transform = Transform(
                position=[float(x), float(y), float(z)],
                rotation=[0, 0, 0, 1],
                scale=[scale_factor, scale_factor, scale_factor]
            )
            
            obj_result = obj.copy()
            obj_result['transform'] = asdict(transform)
            obj_result['label'] = label
            
            # Create labeled object_id: {label}_{mask_id}
            if label:
                safe_label = label.lower().replace(' ', '_').replace('-', '_')
                obj_result['object_id'] = f"{safe_label}_{mask_id}"
            else:
                obj_result['object_id'] = mask_id
            
            placed_objects.append(obj_result)
            
            logger.info(f"Placed {obj_result['object_id']} ({label}): pos=({x:.3f}, {y:.3f}, {z:.3f}), scale={scale_factor:.3f}")
        
        return placed_objects
    
    def _get_label(self, obj: Dict, detections: List[Dict]) -> str:
        """
        Get label for object using detection-centric matching.
        
        For each detection, we pre-compute which mask has the most area inside it.
        Then we look up this mask's assigned label.
        """
        # First check if object already has label (assigned by _assign_labels_to_masks)
        if obj.get('label'):
            return obj['label']
        
        # Fallback to empty if no pre-assigned label
        return ''
    
    def _assign_labels_to_masks(
        self, 
        objects: List[Dict], 
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Assign labels to masks using detection-centric approach.
        
        For each mask:
        1. Find which detection bbox contains the most of this mask
        2. Assign that detection's label to the mask
        
        This allows multiple masks to get the same label if they're 
        both inside the same detection bbox (e.g., 2 crayons in one bbox).
        """
        if not detections:
            return objects
        
        # For each mask, find the best matching detection
        for obj in objects:
            mask_id = obj.get('mask_id', 'unknown')
            bbox = obj.get('bbox', [0, 0, 100, 100])
            
            if len(bbox) != 4:
                continue
                
            mx1, my1, mx2, my2 = [int(float(v)) for v in bbox]
            mask_area = (mx2 - mx1) * (my2 - my1)
            
            if mask_area <= 0:
                continue
            
            best_label = ''
            best_score = 0
            best_estimated_size = None
            best_material = None
            
            for det in detections:
                label = det.get('label', '')
                det_bbox = det.get('bbox', [0, 0, 0, 0])
                dx1, dy1, dx2, dy2 = [float(x) for x in det_bbox[:4]]
                det_area = (dx2 - dx1) * (dy2 - dy1)
                
                if det_area <= 0:
                    continue
                
                # Calculate intersection
                inter_x1 = max(mx1, dx1)
                inter_y1 = max(my1, dy1)
                inter_x2 = min(mx2, dx2)
                inter_y2 = min(my2, dy2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    
                    # Score: what fraction of the MASK is inside this detection?
                    # Higher = mask is more contained within detection
                    containment = inter_area / mask_area
                    
                    # Also consider: what fraction of the DETECTION does this mask fill?
                    # Higher = mask is a good match for this detection's size
                    fill_ratio = inter_area / det_area
                    
                    # Combined score: prefer high containment AND reasonable fill
                    # This helps when a huge detection covers many objects
                    score = containment * (0.5 + 0.5 * min(fill_ratio * 2, 1.0))
                    
                    if score > best_score and containment > 0.3:
                        best_score = score
                        best_label = label
                        # Also capture LLM-estimated size and material from detection
                        best_estimated_size = det.get('estimated_size')
                        best_material = det.get('material')
            
            if best_label:
                obj['label'] = best_label
                # Store LLM-estimated size for use in scaling
                if best_estimated_size is not None:
                    obj['estimated_size'] = best_estimated_size
                if best_material:
                    obj['material'] = best_material
                logger.debug(f"Label: {mask_id} -> '{best_label}' (score={best_score:.2f}, size={best_estimated_size})")
        
        # Log summary
        from collections import Counter
        labels = [obj.get('label', '') for obj in objects if obj.get('label')]
        logger.info(f"Label assignment: {dict(Counter(labels))}")
        
        return objects

    def resolve_collisions(self, objects: List[Dict]) -> List[Dict]:
        """Push apart overlapping objects."""
        if len(objects) <= 1:
            return objects
        
        for i, obj_a in enumerate(objects):
            if obj_a.get('is_surface'):
                continue
            
            pos_a = list(obj_a.get('transform', {}).get('position', [0, 0, 0]))
            scale_a = obj_a.get('transform', {}).get('scale', [0.1])[0]
            
            for j, obj_b in enumerate(objects):
                if i >= j or obj_b.get('is_surface'):
                    continue
                
                pos_b = list(obj_b.get('transform', {}).get('position', [0, 0, 0]))
                scale_b = obj_b.get('transform', {}).get('scale', [0.1])[0]
                
                dx = pos_a[0] - pos_b[0]
                dz = pos_a[2] - pos_b[2]
                dist = np.sqrt(dx * dx + dz * dz)
                
                min_dist = (scale_a + scale_b) / 2 + self.min_separation
                
                if dist < min_dist and dist > 0.001:
                    push = (min_dist - dist) / 2
                    nx, nz = dx / dist, dz / dist
                    
                    pos_a[0] += nx * push
                    pos_a[2] += nz * push
                    pos_b[0] -= nx * push
                    pos_b[2] -= nz * push
                    
                    obj_a['transform']['position'] = pos_a
                    obj_b['transform']['position'] = pos_b
        
        return objects

    def compute_scene_bounds(self, objects: List[Dict]) -> Dict:
        """Compute scene bounding box."""
        if not objects:
            return {'min': [0, 0, 0], 'max': [1, 1, 1]}
        
        positions = np.array([
            obj.get('transform', {}).get('position', [0, 0, 0])
            for obj in objects
        ])
        
        return {
            'min': positions.min(axis=0).tolist(),
            'max': positions.max(axis=0).tolist()
        }
