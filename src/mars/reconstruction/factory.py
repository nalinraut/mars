from typing import Dict, Any, Type
from .base import ReconstructionModel
from .sam3d import Sam3DReconstructor

# Registry of available models
MODEL_REGISTRY: Dict[str, Type[ReconstructionModel]] = {
    "sam3d": Sam3DReconstructor,
    # "cosmos": CosmosReconstructor,  # Future addition
    # "triposr": TripoSRReconstructor # Future addition
}

def get_reconstruction_model(name: str, config: Dict[str, Any]) -> ReconstructionModel:
    """Factory function to get the requested reconstruction model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[name]
    return model_class(config)

