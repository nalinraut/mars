from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class QualityWeights:
    geometric: float = 0.25
    physical: float = 0.25
    completeness: float = 0.20
    semantic: float = 0.15
    utility: float = 0.15

def compute_composite_score(
    metrics: Dict[str, float], 
    weights: QualityWeights = QualityWeights()
) -> float:
    """
    Compute weighted overall quality score.
    metrics dict must contain keys matching the 5 tiers (averages).
    """
    score = (
        weights.geometric * metrics.get("geometric_avg", 0.0) +
        weights.physical * metrics.get("physical_avg", 0.0) +
        weights.completeness * metrics.get("completeness_avg", 0.0) +
        weights.semantic * metrics.get("semantic_avg", 0.0) +
        weights.utility * metrics.get("utility_avg", 0.0)
    )
    return min(100.0, max(0.0, score))

def compute_quick_score(
    geometric_metrics: Dict[str, float],
    segmentation_metrics: Dict[str, float]
) -> float:
    """
    Fast approximation using only cheap-to-compute metrics.
    """
    # Simple average of what we have so far
    score = (
        geometric_metrics.get("mesh_integrity", 0.0) * 0.4 +
        geometric_metrics.get("reconstruction_confidence", 0.0) * 0.3 +
        segmentation_metrics.get("avg_confidence", 0.0) * 0.3
    )
    return min(100.0, max(0.0, score))

