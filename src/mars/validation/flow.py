import logging
import json
from typing import Dict, Any

from prefect import task, flow, get_run_logger
from omegaconf import OmegaConf

from .simulation import SimulationValidator
from .scorer import compute_composite_score, compute_quick_score

@task
def load_validation_config(config_path: str = "config/validation.yaml") -> Dict[str, Any]:
    conf = OmegaConf.load(config_path)
    return OmegaConf.to_container(conf, resolve=True)

@task
def run_simulation_check(
    scene_graph: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Run PyBullet simulation validation.
    """
    validator = SimulationValidator(config)
    return validator.validate_scene(scene_graph)

@task
def compute_final_metrics(
    sim_metrics: Dict[str, float],
    geometric_metrics: Dict[str, float],
    segmentation_metrics: Dict[str, float],
    scene_graph: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aggregate all scores into the final 5-tier record.
    """
    
    # Tier 1: Geometric
    # (passed in from Stage 3 accumulation ideally, or recomputed)
    # For now, we use what we have.
    
    # Tier 2: Physical
    physical = {
        "stability": sim_metrics.get("stability", 0.0),
        "mass_plausibility": 80.0, # Placeholder logic
        "contact_quality": 85.0 # Placeholder
    }
    physical["physical_avg"] = sum(physical.values()) / len(physical)
    
    # Tier 3: Completeness
    obj_count = len(scene_graph.get('objects', []))
    completeness = {
        "task_readiness": 100.0 if obj_count > 0 else 0.0,
        "object_count_score": min(100.0, obj_count * 10.0),
        "spatial_layout": 90.0 # Placeholder
    }
    completeness["completeness_avg"] = sum(completeness.values()) / len(completeness)
    
    # Tier 4: Semantic
    semantic = {
        "segmentation_confidence": segmentation_metrics.get("avg_confidence", 0.0) * 100,
        "category_confidence": 80.0,
        "scene_coherence": 80.0
    }
    semantic["semantic_avg"] = sum(semantic.values()) / len(semantic)
    
    # Tier 5: Utility
    utility = {
        "diversity": 70.0,
        "difficulty_estimate": 50.0,
        "randomization_potential": 90.0
    }
    utility["utility_avg"] = sum(utility.values()) / len(utility)
    
    # Prep metrics for scorer
    all_metrics = {}
    all_metrics.update(geometric_metrics) # assumes it has geometric_avg or components
    # Add manual avg for geometric if missing
    if "geometric_avg" not in all_metrics:
        all_metrics["geometric_avg"] = geometric_metrics.get("reconstruction_confidence", 0.0)
        
    all_metrics.update(physical)
    all_metrics.update(completeness)
    all_metrics.update(semantic)
    all_metrics.update(utility)
    
    # Composite
    overall = compute_composite_score(all_metrics)
    quick = compute_quick_score(geometric_metrics, segmentation_metrics)
    
    return {
        "tier_1_geometric": geometric_metrics,
        "tier_2_physical": physical,
        "tier_3_completeness": completeness,
        "tier_4_semantic": semantic,
        "tier_5_utility": utility,
        "composite": {
            "overall": overall,
            "quick": quick,
            "simulation": sim_metrics.get("stability", 0.0)
        }
    }

@flow(name="Validate Scene")
def validate_scene_flow(
    scene_composition_result: Dict[str, Any],
    segmentation_result: Dict[str, Any],
    geometric_metrics: Dict[str, float],
    config_path: str = "config/validation.yaml"
) -> Dict[str, Any]:
    """
    Main flow for Stage 6: Validation.
    """
    logger = get_run_logger()
    
    image_id = scene_composition_result.get("image_id")
    if not image_id:
         return {"status": "skipped", "reason": "missing_id"}

    # 1. Load Config
    config = load_validation_config(config_path)
    
    # 2. Load Scene Graph
    # We need the full scene graph dict, either passed in or loaded from disk.
    # Assuming it's in 'details' of composition result or we load from path.
    scene_graph_path = scene_composition_result.get("scene_path")
    if not scene_graph_path:
         return {"status": "failed", "reason": "no_scene_graph"}
         
    with open(scene_graph_path, 'r') as f:
        scene_graph = json.load(f)
    
    # 3. Run Simulation
    sim_metrics = run_simulation_check(scene_graph, config)
    logger.info(f"Simulation Stability: {sim_metrics.get('stability')}%")
    
    # 4. Compute All Metrics
    final_quality = compute_final_metrics(
        sim_metrics,
        geometric_metrics,
        segmentation_result.get("metrics", {}),
        scene_graph
    )
    
    return {
        "status": "validated",
        "image_id": image_id,
        "quality_record": final_quality,
        "passed": sim_metrics.get("stability", 0) > 80.0 # Pass/Fail threshold
    }

