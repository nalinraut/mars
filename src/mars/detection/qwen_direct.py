"""
Direct Qwen 2.5 VL detector - labels AND bboxes from single model.

Simpler and faster than hybrid approach (no GroundingDINO needed).
"""

import logging
from typing import Dict, Any, List, Optional
from PIL import Image

from .base import DetectionModel, Detection

logger = logging.getLogger(__name__)


class QwenDirectDetector(DetectionModel):
    """
    Direct detection using Qwen 2.5 VL.
    
    Gets both labels and bounding boxes from Qwen in a single pass.
    Simpler and faster than hybrid approach.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.model = None
        self.processor = None
        self.device = "cuda"
        self._model_loaded = False
        self.model_id = None
    
    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda",
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Load Qwen 2.5 VL model."""
        if self._model_loaded:
            return
        
        self.device = device
        
        # Get model_id from config
        model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        if config:
            model_id = config.get("model", {}).get("huggingface_model", model_id)
        
        self.model_id = model_id
        
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            import torch
            
            logger.info(f"Loading Qwen VL (direct mode): {model_id}")
            
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            
            self._model_loaded = True
            logger.info(f"Qwen VL loaded: {model_id}")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen VL: {e}")
            raise
    
    def detect(
        self,
        image_path: str,
        config: Dict[str, Any]
    ) -> List[Detection]:
        """
        Detect objects with labels and bboxes directly from Qwen.
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_width, original_height = image.size
        
        # Resize large images to prevent OOM
        max_dimension = config.get("detection", {}).get("processing", {}).get("max_image_dimension", 1280)
        scale_factor = 1.0
        
        if max(original_width, original_height) > max_dimension:
            scale_factor = max_dimension / max(original_width, original_height)
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
            logger.info(f"Resized image from {original_width}x{original_height} to {new_width}x{new_height}")
        
        width, height = image.size
        
        logger.info(f"Qwen direct detection on {image_path} ({width}x{height})")
        
        # Prompt for object detection with bounding boxes
        # Qwen 2.5 VL supports grounding with normalized coordinates
        prompt = """Detect all objects in this image. For each object, provide the label and bounding box coordinates.

Output format (one object per line):
label: [x1, y1, x2, y2]

Where coordinates are normalized to 0-1000 range.
Example:
cup: [100, 200, 300, 400]
table: [0, 500, 1000, 1000]

Objects in image:"""

        # Prepare messages for Qwen
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                temperature=0.01,
                top_p=0.001,
                top_k=1,
            )
        
        # Decode response
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Qwen response: {response[:200]}...")
        
        # Parse response to extract detections
        detections = self._parse_response(response, width, height, original_width, original_height, scale_factor, config)
        
        logger.info(f"Qwen direct detection: {len(detections)} objects")
        return detections
    
    def _parse_response(
        self,
        response: str,
        width: int,
        height: int,
        original_width: int,
        original_height: int,
        scale_factor: float,
        config: Dict[str, Any]
    ) -> List[Detection]:
        """Parse Qwen response to extract detections."""
        import re
        
        detections = []
        conf_threshold = config.get("detection", {}).get("processing", {}).get("box_threshold", 0.35)
        
        # Pattern to match "label: [x1, y1, x2, y2]" format
        pattern = r'([^:\[\]]+):\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
        
        for match in re.finditer(pattern, response):
            label = match.group(1).strip()
            x1 = int(match.group(2))
            y1 = int(match.group(3))
            x2 = int(match.group(4))
            y2 = int(match.group(5))
            
            # Skip background/environment labels
            label_lower = label.lower()
            if any(skip in label_lower for skip in ['background', 'floor', 'wall', 'ceiling', 'sky']):
                continue
            
            # Convert from 0-1000 normalized to pixel coordinates
            x1_px = int(x1 * width / 1000)
            y1_px = int(y1 * height / 1000)
            x2_px = int(x2 * width / 1000)
            y2_px = int(y2 * height / 1000)
            
            # Clamp to image bounds
            x1_px = max(0, min(x1_px, width))
            y1_px = max(0, min(y1_px, height))
            x2_px = max(0, min(x2_px, width))
            y2_px = max(0, min(y2_px, height))
            
            # Skip invalid boxes
            if x2_px <= x1_px or y2_px <= y1_px:
                continue
            
            # Scale back to original image coordinates
            if scale_factor != 1.0:
                x1_px = int(x1_px / scale_factor)
                y1_px = int(y1_px / scale_factor)
                x2_px = int(x2_px / scale_factor)
                y2_px = int(y2_px / scale_factor)
            
            # Create detection (assign confidence based on order - earlier = more confident)
            # Qwen doesn't output confidence, so we estimate based on position
            confidence = max(0.5, 0.95 - len(detections) * 0.05)
            
            detection = Detection(
                label=label,
                confidence=confidence,
                bbox=(x1_px, y1_px, x2_px, y2_px),
                bbox_normalized=(
                    x1_px / original_width,
                    y1_px / original_height,
                    x2_px / original_width,
                    y2_px / original_height
                ),
                metadata={
                    "model": "qwen_direct",
                    "model_id": self.model_id,
                    "image_size": (original_width, original_height),
                }
            )
            detections.append(detection)
            logger.info(f"  {label}: bbox=[{x1_px},{y1_px},{x2_px},{y2_px}]")
        
        return detections

