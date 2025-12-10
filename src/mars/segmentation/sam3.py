"""
SAM 3 segmentation implementation.
Uses Meta's SAM 3 (Segment Anything Model 3) from Hugging Face Transformers.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from PIL import Image

from .base import SegmentationModel

logger = logging.getLogger(__name__)


class Sam3Segmentation(SegmentationModel):
    """
    Segmentation using Meta's SAM 3 (Segment Anything Model 3).
    
    Features:
    - Advanced automatic mask generation
    - Natural language prompts support
    - Better accuracy than SAM 2.1
    - Released November 2025
    
    Reference: https://ai.meta.com/sam3/
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.device = "cpu"
        
    def load_model(self, checkpoint_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, device: str = "cuda"):
        """Load SAM 3 model using Hugging Face Transformers."""
        
        try:
            import torch
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Check CUDA availability
            if device == "cuda" and torch.cuda.is_available():
                try:
                    torch.zeros(1).cuda()
                    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
                    self.device = "cuda"
                except Exception as cuda_error:
                    logger.warning(f"CUDA initialization failed: {cuda_error}, falling back to CPU")
                    device = "cpu"
                    self.device = "cpu"
            else:
                logger.info("Using CPU mode")
                device = "cpu"
                self.device = "cpu"
            
            # Get model name from config
            model_config = self.config.get('segmentation', {}).get('model', {})
            model_name = model_config.get('huggingface_model', 'facebook/sam3')
            
            logger.info(f"Loading SAM 3 model from official sam3 package")
            
            # Load model using official sam3 API
            self.model = build_sam3_image_model()
            
            # Move model to device
            if device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(device)
                logger.info(f"Model moved to CUDA")
            else:
                self.model = self.model.to("cpu")
                device = "cpu"
                logger.info(f"Model on CPU")
            
            self.device = device
            
            # Get confidence threshold from config (lower = more masks/instances)
            # Default 0.3 to get multiple instances per text prompt
            seg_config = self.config.get('segmentation', {})
            confidence_threshold = seg_config.get('confidence_threshold', 0.3)
            
            self.processor = Sam3Processor(
                self.model, 
                device=device,
                confidence_threshold=confidence_threshold
            )
            logger.info(f"SAM 3 confidence threshold: {confidence_threshold}")
            
            logger.info(f"SAM 3 loaded successfully on {self.device}")
            
        except ImportError as e:
            raise RuntimeError(
                "SAM 3 requires transformers with SAM 3 support. Install with:\n"
                "pip install transformers>=4.46.0\n"
                f"Error: {e}"
            )
    
    def generate_masks(self, image_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate segmentation masks using SAM 3 with text prompts.
        
        Args:
            image_path: Path to input image file
            config: Segmentation configuration with text_prompts
            
        Returns:
            List of mask dictionaries with keys:
                - segmentation: binary mask (H, W) as uint8
                - area: number of pixels
                - bbox: [x, y, w, h]
                - predicted_iou: confidence score
                - stability_score: mask quality score
                - prompt: the text prompt that generated this mask
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        import torch
        from PIL import Image as PILImage
        
        # Load image
        image_pil = PILImage.open(image_path).convert("RGB")
        h, w = image_pil.size[1], image_pil.size[0]  # PIL size is (width, height)
        
        logger.info(f"Generating masks for image {image_path} ({w}x{h})")
        
        # Get text prompts from config
        prompts_config = config.get('segmentation', {}).get('prompts', {})
        text_prompts = prompts_config.get('text_prompts', [])
        
        all_masks = []
        
        # Use text prompts for segmentation
        if text_prompts:
            # Use text prompts to find all instances of each concept
            logger.info(f"Using {len(text_prompts)} text prompts: {text_prompts}")
            
            for prompt in text_prompts:
                try:
                    # Initialize inference session with image
                    inference_state = self.processor.set_image(image_pil)
                    
                    # Use text prompt
                    output = self.processor.set_text_prompt(state=inference_state, prompt=prompt)
                    
                    # Get masks, boxes, and scores from output
                    # SAM 3 returns: masks [N, 1, H, W], boxes [N, 4], scores [N]
                    masks_tensor = output["masks"]  # [num_masks, 1, H, W]
                    boxes_tensor = output["boxes"]   # [num_masks, 4]
                    scores_tensor = output["scores"]  # [num_masks]
                    
                    # Determine number of instances found
                    num_masks = masks_tensor.shape[0]
                    
                    logger.info(f"  Prompt '{prompt}': SAM 3 returned {num_masks} instances")
                    
                    # Process each mask instance
                    for i in range(num_masks):
                        # Extract mask - remove channel dimension [1, H, W] -> [H, W]
                        mask = masks_tensor[i].squeeze(0)  # [H, W]
                        
                        # Convert boolean mask to uint8
                        mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                        
                        # Extract box (xyxy format)
                        box = boxes_tensor[i].cpu().numpy()  # [4]
                        x1, y1, x2, y2 = box.astype(int)
                        bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to xywh
                        
                        # Extract score
                        score = float(scores_tensor[i].cpu())
                        
                        # Calculate area
                        area = int(mask_np.sum() / 255)  # Count non-zero pixels
                        
                        if area > 0:  # Only add non-empty masks
                            all_masks.append({
                                'segmentation': mask_np,
                                'area': area,
                                'bbox': bbox,
                                'predicted_iou': score,
                                'stability_score': score,
                                'prompt': prompt,  # Track which prompt generated this
                            })
                            logger.debug(f"    Instance {i}: score={score:.3f}, area={area}")
                    
                    count = len([m for m in all_masks if m.get('prompt') == prompt])
                    logger.info(f"  Prompt '{prompt}': kept {count} instances")
                
                except Exception as e:
                    logger.warning(f"Error processing prompt '{prompt}': {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    continue
        
        else:
            # No text prompts - try with generic "object" prompt
            logger.warning("No text prompts provided. Using generic 'object' prompt...")
            
            try:
                inference_state = self.processor.set_image(image_pil)
                output = self.processor.set_text_prompt(state=inference_state, prompt="object")
                
                # SAM 3 returns: masks [N, 1, H, W], boxes [N, 4], scores [N]
                masks_tensor = output["masks"]
                boxes_tensor = output["boxes"]
                scores_tensor = output["scores"]
                
                num_masks = masks_tensor.shape[0]
                logger.info(f"Generic 'object' prompt returned {num_masks} instances")
                
                for i in range(num_masks):
                    mask = masks_tensor[i].squeeze(0)  # [H, W]
                    mask_np = mask.cpu().numpy().astype(np.uint8) * 255
                    
                    box = boxes_tensor[i].cpu().numpy()
                    x1, y1, x2, y2 = box.astype(int)
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    
                    score = float(scores_tensor[i].cpu())
                    area = int(mask_np.sum() / 255)
                    
                    if area > 0:
                        all_masks.append({
                            'segmentation': mask_np,
                            'area': area,
                            'bbox': bbox,
                            'predicted_iou': score,
                            'stability_score': score,
                        })
            except Exception as e:
                logger.error(f"Generic segmentation failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info(f"Generated {len(all_masks)} masks total with SAM 3")
        return all_masks
    
    def unload_model(self):
        """Release model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        logger.info("SAM 3 model unloaded")

