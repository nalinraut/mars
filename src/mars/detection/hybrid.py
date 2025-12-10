"""
Hybrid detector combining Qwen2-VL (labels) + GroundingDINO (bboxes).

Workflow:
1. Qwen2-VL analyzes image and returns object labels (no bboxes needed)
2. GroundingDINO uses those labels as prompts for accurate bboxes
3. Results have both rich labels AND accurate localization
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from PIL import Image

from .base import DetectionModel, Detection

logger = logging.getLogger(__name__)


class HybridDetector(DetectionModel):
    """
    Hybrid detector: Qwen2-VL for labels + GroundingDINO for bboxes.
    
    Why:
    - Qwen2-VL is great at understanding scene content but poor at localization
    - GroundingDINO is great at localization but needs text prompts
    - Together they provide accurate labeled detections
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.qwen_model = None
        self.qwen_processor = None
        self.gdino_model = None
        self.gdino_processor = None
        self._model_loaded = False
        self.device = "cuda"
    
    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load both Qwen VL and GroundingDINO models."""
        if self._model_loaded:
            return
        
        self.device = device
        
        # Get model_id from config if provided
        model_id = None
        if config:
            model_id = config.get("model", {}).get("huggingface_model")
        
        # Load Qwen VL for label extraction
        self._load_qwen(device, model_id=model_id)
        
        # Load GroundingDINO for bbox detection
        self._load_gdino(device)
        
        self._model_loaded = True
        logger.info("Hybrid detector loaded successfully")
    
    def _load_qwen(self, device: str, model_id: str = None):
        """Load Qwen VL model (supports 2.0 and 2.5 variants)."""
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
            
            # Use provided model_id or default
            if not model_id:
                model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
            
            self.qwen_model_id = model_id
            logger.info(f"Loading Qwen VL: {model_id}")
            
            self.qwen_processor = AutoProcessor.from_pretrained(model_id)
            self.qwen_model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            logger.info(f"Qwen VL loaded: {model_id}")
        except Exception as e:
            logger.error(f"Failed to load Qwen VL: {e}")
            raise
    
    def _load_gdino(self, device: str):
        """Load GroundingDINO model."""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            model_id = "IDEA-Research/grounding-dino-tiny"
            logger.info(f"Loading GroundingDINO: {model_id}")
            
            self.gdino_processor = AutoProcessor.from_pretrained(model_id)
            self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.gdino_model.to(device)
            self.gdino_model.eval()
            logger.info("GroundingDINO loaded")
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO: {e}")
            raise
    
    def detect(
        self,
        image_path: str,
        config: Dict[str, Any]
    ) -> List[Detection]:
        """
        Detect objects using hybrid approach.
        
        Steps:
        1. Use Qwen2-VL to get object labels from image
        2. Use GroundingDINO with those labels to get accurate bboxes
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        
        # Resize large images to prevent OOM (attention scales quadratically)
        max_dimension = config.get("detection", {}).get("processing", {}).get("max_image_dimension", 1280)
        scale_factor = 1.0
        
        if max(original_width, original_height) > max_dimension:
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
        
        width, height = image.size
        
        logger.info(f"Hybrid detection on {image_path} ({width}x{height})")
        
        # Step 1: Get labels from Qwen2-VL
        labels = self._get_labels_from_qwen(image, config)
        logger.info(f"Qwen2-VL detected labels: {labels}")
        
        if not labels:
            logger.warning("No labels detected by Qwen2-VL")
            return []
        
        # Step 2: Get bboxes from GroundingDINO using UNIQUE labels as prompts
        # (Qwen might list "blue block, blue block" for 2 instances - we only need unique labels for DINO)
        unique_labels = list(dict.fromkeys(labels))  # Preserve order, remove duplicates
        logger.info(f"Unique labels for GroundingDINO: {unique_labels}")
        detections = self._get_bboxes_from_gdino(image, unique_labels, width, height, config)
        
        # Scale bboxes back to original image coordinates if we resized
        if scale_factor != 1.0:
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                det.bbox = (
                    int(x1 / scale_factor),
                    int(y1 / scale_factor),
                    int(x2 / scale_factor),
                    int(y2 / scale_factor)
                )
                det.bbox_normalized = (
                    det.bbox[0] / original_width,
                    det.bbox[1] / original_height,
                    det.bbox[2] / original_width,
                    det.bbox[3] / original_height
                )
                det.metadata["image_size"] = (original_width, original_height)
                det.metadata["scale_factor"] = scale_factor
        
        logger.info(f"Hybrid detection: {len(detections)} objects")
        
        # Step 3: Estimate real-world sizes using Qwen with scene context
        if detections and config.get("detection", {}).get("estimate_sizes", True):
            detections = self._estimate_sizes_with_context(image, detections, config)
        
        return detections
    
    def _get_labels_from_qwen(self, image: Image.Image, config: Dict[str, Any]) -> List[str]:
        """Use Qwen2-VL to get object labels (no bboxes)."""
        import torch
        
        # Get prompt from config or use default
        # Keep it simple - let the model do its job
        default_prompt = """List all objects visible in this image.
Return a comma-separated list of object names with colors/materials.
Objects:"""
        
        # Try multiple config paths (config structure varies)
        prompt = (
            config.get("detection", {}).get("prompts", {}).get("qwen_prompt") or
            config.get("prompts", {}).get("qwen_prompt") or
            default_prompt
        )
        logger.debug(f"Using Qwen prompt: {prompt[:80]}...")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        inputs = self.qwen_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.qwen_model.device)
        
        with torch.no_grad():
            output_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.qwen_processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        
        # Parse comma-separated labels
        labels = [l.strip() for l in response.split(",") if l.strip()]
        
        # Filter out common non-object words
        skip_words = {"the", "a", "an", "and", "or", "in", "on", "at", "to", "for"}
        labels = [l for l in labels if l.lower() not in skip_words and len(l) > 2]
        
        # Clean up duplicated words in labels (e.g., "blue block blue block" -> "blue block")
        cleaned_labels = []
        for label in labels:
            words = label.split()
            # Check if label is duplicated (same words repeated)
            half = len(words) // 2
            if half > 0 and words[:half] == words[half:2*half]:
                # Deduplicate: keep only first half
                label = " ".join(words[:half])
            cleaned_labels.append(label)
        
        logger.info(f"Qwen2-VL labels: {cleaned_labels}")
        
        return cleaned_labels
    
    def _get_bboxes_from_gdino(
        self,
        image: Image.Image,
        labels: List[str],
        width: int,
        height: int,
        config: Dict[str, Any]
    ) -> List[Detection]:
        """Use GroundingDINO to get accurate bboxes for each label."""
        import torch
        
        detections = []
        # Read threshold from config (try multiple paths for compatibility)
        conf_threshold = (
            config.get("detection", {}).get("processing", {}).get("box_threshold") or
            config.get("detection", {}).get("confidence_threshold") or
            config.get("processing", {}).get("box_threshold") or
            0.3
        )
        
        # Create prompt string for GroundingDINO (period-separated)
        prompt = ". ".join(labels) + "."
        
        logger.info(f"GroundingDINO prompt: {prompt[:100]}...")
        
        inputs = self.gdino_processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.gdino_model(**inputs)
        
        # Process outputs - use lower threshold to see all candidates, filter later
        raw_threshold = 0.1  # Low threshold to get all candidates
        results = self.gdino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=raw_threshold,
            text_threshold=raw_threshold,
            target_sizes=[(height, width)]
        )[0]
        
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        result_labels = results["labels"]
        
        logger.debug(f"GroundingDINO raw candidates: {len(boxes)} (threshold={raw_threshold})")
        
        for box, score, label in zip(boxes, scores, result_labels):
            # Apply configured threshold
            if score < conf_threshold:
                continue
            
            # Clean up merged/malformed labels from GroundingDINO
            # e.g., "blue wooden blocks white wooden blocks" -> find matching labels
            label_lower = label.lower().strip()
            
            # Find all matching original labels
            matching_labels = []
            for orig in labels:
                if orig.lower() in label_lower:
                    matching_labels.append(orig)
            
            # Determine which label(s) to use
            if not matching_labels:
                labels_to_add = [label]  # Keep original if no match
            elif len(matching_labels) == 1:
                labels_to_add = matching_labels
            else:
                # Multiple matches - create detection for each (segmentation will sort out)
                labels_to_add = matching_labels
                
            x1, y1, x2, y2 = map(int, box)
            
            # Clamp to image bounds
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Create detection for each matched label
            # (merged detections like "blue blocks white blocks" create multiple detections)
            for det_label in labels_to_add:
                detection = Detection(
                    label=det_label,
                    confidence=float(score),
                    bbox=(x1, y1, x2, y2),
                    bbox_normalized=(
                        x1 / width,
                        y1 / height,
                        x2 / width,
                        y2 / height
                    ),
                    metadata={
                        "model": "hybrid_qwen_gdino",
                        "image_size": (width, height),
                    }
                )
                detections.append(detection)
                logger.info(f"  {det_label}: bbox=[{x1},{y1},{x2},{y2}] conf={score:.2f}")
        
        return detections
    
    def detect_with_prompts(
        self,
        image_path: str,
        prompts: List[str],
        config: Dict[str, Any]
    ) -> List[Detection]:
        """
        Detect objects using provided prompts (skip Qwen2-VL label extraction).
        Useful when you already know what objects to detect.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        return self._get_bboxes_from_gdino(image, prompts, width, height, config)
    
    def _estimate_sizes_with_context(
        self, 
        image: Image.Image, 
        detections: List[Detection],
        config: Dict[str, Any]
    ) -> List[Detection]:
        """
        Use Qwen to estimate real-world sizes with scene context.
        
        Smart sizing:
        - For supporting surfaces (table, desk), considers objects on top
        - For regular objects, uses typical size knowledge
        - Returns sizes in meters
        """
        import torch
        import re
        
        if not detections:
            return detections
        
        # Get all labels for context
        all_labels = [d.label for d in detections]
        labels_str = ", ".join(all_labels)
        
        # Identify supporting surfaces (tables, desks, shelves)
        surface_keywords = ['table', 'desk', 'shelf', 'counter', 'bench', 'stand']
        surfaces = []
        objects_on_surface = []
        
        for det in detections:
            label_lower = det.label.lower()
            if any(kw in label_lower for kw in surface_keywords):
                surfaces.append(det)
            else:
                objects_on_surface.append(det)
        
        # Build context-aware prompt from config
        default_size_prompt = """For each object, estimate its TYPICAL real-world size (largest dimension in METERS).
Be precise - use exact measurements for common objects.

Objects detected: {objects}
{surface_context}

Return ONLY in this format (one per line):
object_name: size_in_meters

Sizes:"""
        
        size_prompt_template = (
            config.get("detection", {}).get("prompts", {}).get("size_prompt") or
            config.get("prompts", {}).get("size_prompt") or
            default_size_prompt
        )
        
        # Build surface context
        if surfaces and objects_on_surface:
            obj_labels = [d.label for d in objects_on_surface]
            surface_labels = [d.label for d in surfaces]
            surface_context = f"Context: {', '.join(surface_labels)} supports {', '.join(obj_labels)}"
        else:
            surface_context = ""
        
        # Format prompt
        prompt = size_prompt_template.format(
            objects=labels_str,
            surface_context=surface_context
        )
        
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            text = self.qwen_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = self.qwen_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.qwen_model.device)
            
            with torch.no_grad():
                outputs = self.qwen_model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                )
            
            response = self.qwen_processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )[0]
            
            logger.debug(f"Size estimation response: {response}")
            
            # Parse response: "object_name: size_in_meters"
            size_map = {}
            for line in response.strip().split('\n'):
                line = line.strip()
                if ':' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        obj_name = parts[0].strip().lower()
                        size_str = parts[1].strip()
                        # Extract number from size string
                        match = re.search(r'([\d.]+)', size_str)
                        if match:
                            try:
                                size = float(match.group(1))
                                # Sanity check: size should be between 1cm and 5m
                                if 0.01 <= size <= 5.0:
                                    size_map[obj_name] = size
                            except ValueError:
                                pass
            
            logger.info(f"Estimated sizes: {size_map}")
            
            # Apply sizes to detections with fuzzy matching
            for det in detections:
                label_lower = det.label.lower()
                # Normalize: remove extra spaces, handle dashes
                label_norm = ' '.join(label_lower.split()).replace(' - ', '-')
                
                # Try exact match first
                if label_norm in size_map:
                    det.estimated_size = size_map[label_norm]
                    continue
                
                # Try normalized key match
                for key, size in size_map.items():
                    key_norm = ' '.join(key.split()).replace(' - ', '-')
                    if key_norm == label_norm:
                        det.estimated_size = size
                        break
                    # Partial/substring match
                    if key_norm in label_norm or label_norm in key_norm:
                        det.estimated_size = size
                        break
                    # Word overlap match (for "blue blocks" vs "block")
                    label_words = set(label_norm.replace('-', ' ').split())
                    key_words = set(key_norm.replace('-', ' ').split())
                    # Match if significant word overlap (>50%)
                    common = label_words & key_words
                    if len(common) >= max(1, min(len(label_words), len(key_words)) // 2):
                        det.estimated_size = size
                        break
            
        except Exception as e:
            logger.warning(f"Size estimation failed: {e}")
        
        return detections

