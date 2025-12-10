"""
Visualization utilities for MARS pipeline.

Provides tools to visualize:
1. Segmentation masks overlaid on images
2. 3D scenes with meshes, physics, and zones
3. Quality metrics and scene graphs
"""

import logging
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2

logger = logging.getLogger(__name__)

def visualize_segmentation(
    image_path: str,
    masks: List[Dict],
    output_path: Optional[str] = None,
    show_boxes: bool = True,
    show_scores: bool = True,
    alpha: float = 0.5
) -> str:
    """
    Visualize segmentation masks overlaid on the original image.
    
    Args:
        image_path: Path to the original image
        masks: List of mask dictionaries with 'segmentation', 'bbox', 'predicted_iou'
        output_path: Where to save the visualization (auto-generated if None)
        show_boxes: Whether to draw bounding boxes
        show_scores: Whether to display IoU scores
        alpha: Transparency of mask overlay (0-1)
        
    Returns:
        Path to the saved visualization
    """
    from PIL import Image
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np)
    ax.set_title(f"Segmentation Results: {len(masks)} masks", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Generate distinct colors for each mask
    colors = plt.cm.rainbow(np.linspace(0, 1, len(masks)))
    
    # Create overlay for all masks
    overlay = np.zeros((*image_np.shape[:2], 4))
    
    for idx, (mask_data, color) in enumerate(zip(masks, colors)):
        mask = mask_data['segmentation']
        bbox = mask_data['bbox']
        score = mask_data.get('predicted_iou', mask_data.get('stability_score', 0))
        
        # Add mask to overlay with unique color
        mask_rgba = np.zeros((*mask.shape, 4))
        mask_rgba[mask > 0] = [*color[:3], alpha]
        overlay = np.maximum(overlay, mask_rgba)
        
        # Draw bounding box
        if show_boxes:
            x, y, w, h = bbox
            rect = patches.Rectangle(
                (x, y), w, h,
                linewidth=2,
                edgecolor=color,
                facecolor='none',
                linestyle='--'
            )
            ax.add_patch(rect)
        
        # Add score label
        if show_scores:
            x, y, w, h = bbox
            label = f"#{idx} ({score:.2f})"
            ax.text(
                x + w/2, y - 5,
                label,
                color='white',
                fontsize=10,
                fontweight='bold',
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7, edgecolor='none')
            )
    
    # Apply overlay
    ax.imshow(overlay)
    
    # Generate output path if not provided
    if output_path is None:
        input_path = Path(image_path)
        output_path = input_path.parent / f"{input_path.stem}_segmentation.png"
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Segmentation visualization saved to: {output_path}")
    return str(output_path)


def visualize_scene_3d(
    scene_json_path: str,
    output_path: Optional[str] = None,
    show_zones: bool = True,
    show_axes: bool = True,
    camera_view: Tuple[float, float] = (30, 45)
) -> str:
    """
    Visualize the 3D scene with meshes, zones, and camera.
    
    Args:
        scene_json_path: Path to scene.json
        output_path: Where to save the visualization
        show_zones: Whether to visualize zones
        show_axes: Whether to show coordinate axes
        camera_view: (elevation, azimuth) for camera view in degrees
        
    Returns:
        Path to the saved visualization
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        logger.error("mpl_toolkits.mplot3d not available. Install matplotlib with 3D support.")
        return None
    
    # Load scene data
    with open(scene_json_path, 'r') as f:
        scene = json.load(f)
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot objects - load actual meshes
    try:
        import trimesh
        import trimesh.transformations
    except ImportError:
        logger.warning("trimesh not available, using sphere placeholders")
        trimesh = None
    
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    scene_dir = Path(scene_json_path).parent
    colors = plt.cm.tab10(np.linspace(0, 1, len(scene.get('objects', []))))
    
    for idx, (obj, color) in enumerate(zip(scene.get('objects', []), colors)):
        mask_id = obj.get('mask_id', f'mask_{idx}')
        pos = obj['transform']['position']
        rot = obj['transform'].get('rotation', [0, 0, 0, 1])  # quaternion [x, y, z, w]
        scale = obj['transform'].get('scale', [1, 1, 1])
        material = obj.get('material', 'unknown')
        mass = obj.get('mass', 0)
        
        mesh_loaded = False
        
        # Try to load actual mesh if trimesh is available
        if trimesh is not None:
            mesh_path = scene_dir / "objects" / mask_id / "visual.obj"
            if mesh_path.exists():
                try:
                    mesh = trimesh.load(str(mesh_path))
                    
                    # Apply transform
                    if isinstance(mesh, trimesh.Trimesh):
                        # Create transform matrix
                        # Scale
                        scale_matrix = np.eye(4)
                        scale_matrix[0, 0] = scale[0]
                        scale_matrix[1, 1] = scale[1]
                        scale_matrix[2, 2] = scale[2]
                        
                        # Rotation (quaternion [x, y, z, w] -> [w, x, y, z] for trimesh)
                        if len(rot) == 4:
                            q_trimesh = [rot[3], rot[0], rot[1], rot[2]]  # [w, x, y, z]
                            rot_matrix = trimesh.transformations.quaternion_matrix(q_trimesh)
                        else:
                            rot_matrix = np.eye(4)
                        
                        # Translation
                        trans_matrix = np.eye(4)
                        trans_matrix[:3, 3] = pos
                        
                        # Combine transforms: T * R * S
                        transform = trans_matrix @ rot_matrix @ scale_matrix
                        mesh.apply_transform(transform)
                        
                        # Extract vertices and faces for matplotlib
                        vertices = mesh.vertices
                        faces = mesh.faces
                        
                        # Create Poly3DCollection
                        triangles = vertices[faces]
                        collection = Poly3DCollection(triangles, alpha=0.7, facecolor=color[:3], 
                                                     edgecolor='black', linewidth=0.3)
                        ax.add_collection3d(collection)
                        
                        mesh_loaded = True
                        logger.info(f"Loaded mesh for {mask_id}: {len(vertices)} vertices, {len(faces)} faces")
                        
                except Exception as e:
                    logger.warning(f"Could not load mesh {mesh_path}: {e}")
        
        # Fallback to sphere if mesh not loaded
        if not mesh_loaded:
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            radius = 0.05
            x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
            y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
            z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
            ax.plot_surface(x, y, z, color=color, alpha=0.7)
        
        # Add label
        ax.text(pos[0], pos[1], pos[2] + 0.08, 
                f"{mask_id}\n{material}\n{mass:.1f}kg",
                fontsize=8, ha='center')
    
    # Plot zones
    if show_zones:
        zone_colors = {'pick': 'green', 'place': 'blue', 'dropzone': 'red', 'storage': 'orange'}
        zone_alpha = 0.1
        
        for zone in scene.get('zones', []):
            name = zone['name']
            bounds = zone['bounds']
            x_min, y_min, z_min, x_max, y_max, z_max = bounds
            
            # Draw wireframe box for zone
            vertices = [
                [x_min, y_min, z_min], [x_max, y_min, z_min],
                [x_max, y_max, z_min], [x_min, y_max, z_min],
                [x_min, y_min, z_max], [x_max, y_min, z_max],
                [x_max, y_max, z_max], [x_min, y_max, z_max]
            ]
            
            # Define the 6 faces
            faces = [
                [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
            ]
            
            color = zone_colors.get(name, 'gray')
            face_collection = Poly3DCollection(faces, alpha=zone_alpha, facecolor=color, edgecolor=color, linewidth=2)
            ax.add_collection3d(face_collection)
            
            # Add zone label
            center = [(x_min + x_max)/2, (y_min + y_max)/2, z_max + 0.05]
            ax.text(*center, name.upper(), fontsize=10, fontweight='bold', color=color)
    
    # Plot camera (as a marker)
    camera_pos = scene.get('camera', {}).get('position', [0, 0, 1])
    ax.scatter(*camera_pos, color='purple', s=100, marker='^', label='Camera')
    
    # Plot ground plane
    ground_size = 1.0
    xx, yy = np.meshgrid(
        np.linspace(-ground_size, ground_size, 2),
        np.linspace(-ground_size, ground_size, 2)
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
    
    # Set labels and title
    ax.set_xlabel('X (m)', fontsize=10)
    ax.set_ylabel('Y (m)', fontsize=10)
    ax.set_zlabel('Z (m)', fontsize=10)
    ax.set_title(f"Scene: {scene['scene_id']}\n{len(scene.get('objects', []))} objects", 
                 fontsize=12, fontweight='bold')
    
    # Set viewing angle
    ax.view_init(elev=camera_view[0], azim=camera_view[1])
    
    # Set axis limits
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 0.5])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Generate output path if not provided
    if output_path is None:
        scene_path = Path(scene_json_path)
        output_path = scene_path.parent / f"{scene_path.stem}_3d_view.png"
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"3D scene visualization saved to: {output_path}")
    return str(output_path)


def visualize_scene_3d_multi_view(
    scene_json_path: str,
    output_path: Optional[str] = None,
    views: Optional[List[Tuple[str, Tuple[float, float]]]] = None
) -> str:
    """
    Create a multi-view 3D visualization showing perspective, top, and side views.
    
    Args:
        scene_json_path: Path to scene.json
        output_path: Where to save the visualization
        views: List of (name, (elevation, azimuth)) tuples. Default: perspective, top, side
        
    Returns:
        Path to the saved visualization
    """
    try:
        import trimesh
        import trimesh.transformations
    except ImportError:
        logger.warning("trimesh not available, using sphere placeholders")
        trimesh = None
    
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Load scene data
    with open(scene_json_path, 'r') as f:
        scene = json.load(f)
    
    scene_dir = Path(scene_json_path).parent
    scene_id = scene.get('scene_id', 'unknown')
    objects = scene.get('objects', [])
    
    # Default views
    if views is None:
        views = [
            ("Perspective", (30, 45)),
            ("Top View", (90, 0)),
            ("Side View", (0, 0))
        ]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(6 * len(views), 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(objects)))
    
    for view_idx, (view_name, (elev, azim)) in enumerate(views):
        ax = fig.add_subplot(1, len(views), view_idx + 1, projection='3d')
        
        # Load and plot meshes
        for idx, (obj, color) in enumerate(zip(objects, colors)):
            mask_id = obj.get('mask_id', f'mask_{idx}')
            pos = obj['transform']['position']
            rot = obj['transform'].get('rotation', [0, 0, 0, 1])
            scale = obj['transform'].get('scale', [1, 1, 1])
            material = obj.get('material', 'unknown')
            
            mesh_loaded = False
            
            if trimesh is not None:
                mesh_path = scene_dir / "objects" / mask_id / "visual.obj"
                if mesh_path.exists():
                    try:
                        mesh = trimesh.load(str(mesh_path))
                        if isinstance(mesh, trimesh.Trimesh):
                            # Apply transforms
                            scale_matrix = np.eye(4)
                            scale_matrix[0, 0] = scale[0]
                            scale_matrix[1, 1] = scale[1]
                            scale_matrix[2, 2] = scale[2]
                            
                            if len(rot) == 4:
                                q_trimesh = [rot[3], rot[0], rot[1], rot[2]]
                                rot_matrix = trimesh.transformations.quaternion_matrix(q_trimesh)
                            else:
                                rot_matrix = np.eye(4)
                            
                            trans_matrix = np.eye(4)
                            trans_matrix[:3, 3] = pos
                            
                            transform = trans_matrix @ rot_matrix @ scale_matrix
                            mesh.apply_transform(transform)
                            
                            vertices = mesh.vertices
                            faces = mesh.faces
                            
                            triangles = vertices[faces]
                            collection = Poly3DCollection(triangles, alpha=0.7, facecolor=color[:3], 
                                                         edgecolor='black', linewidth=0.2)
                            ax.add_collection3d(collection)
                            mesh_loaded = True
                    except Exception as e:
                        logger.debug(f"Could not load mesh {mask_id}: {e}")
            
            # Fallback to sphere
            if not mesh_loaded:
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 15)
                radius = 0.05
                x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
                y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
                z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
                ax.plot_surface(x, y, z, color=color, alpha=0.7)
        
        # Ground plane
        ground_size = 0.6
        xx, yy = np.meshgrid(np.linspace(-ground_size, ground_size, 2),
                             np.linspace(-ground_size, ground_size, 2))
        zz = np.zeros_like(xx)
        ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray')
        
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        ax.set_title(view_name, fontsize=12, fontweight='bold')
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim([-0.4, 0.4])
        ax.set_ylim([-0.4, 0.4])
        ax.set_zlim([0, 0.4])
        ax.grid(True, alpha=0.3)
    
    # Set overall title
    num_objects = len(objects)
    plt.suptitle(f'3D Scene: {scene_id} ({num_objects} objects)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Generate output path if not provided
    if output_path is None:
        scene_path = Path(scene_json_path)
        output_path = scene_path.parent / f"{scene_path.stem}_3d_multi_view.png"
    
    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Multi-view 3D visualization saved to: {output_path}")
    return str(output_path)


def visualize_quality_metrics(
    scene_json_path: str,
    database_path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Visualize quality metrics from the database.
    
    Args:
        scene_json_path: Path to scene.json to get scene_id
        database_path: Path to SQLite database
        output_path: Where to save the visualization
        
    Returns:
        Path to the saved visualization
    """
    import sqlite3
    
    # Load scene to get ID
    with open(scene_json_path, 'r') as f:
        scene = json.load(f)
    scene_id = scene['scene_id']
    
    # Query database for quality metrics
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT geometric_score, physical_score, completeness_score, 
               semantic_score, training_utility_score,
               overall_score, quick_score, simulation_score
        FROM scene_quality
        WHERE image_id = (SELECT id FROM ingestion WHERE unique_id = ?)
    """, (scene_id,))
    
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        logger.warning(f"No quality metrics found for scene {scene_id}")
        return None
    
    # Unpack scores
    geo, phys, comp, sem, train, overall, quick, sim = result
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tier scores (bar chart)
    tiers = ['Geometric', 'Physical', 'Completeness', 'Semantic', 'Training']
    tier_scores = [geo, phys, comp, sem, train]
    colors_tier = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    
    ax1.barh(tiers, tier_scores, color=colors_tier, alpha=0.8)
    ax1.set_xlim([0, 1])
    ax1.set_xlabel('Score', fontsize=12)
    ax1.set_title('Quality Metrics by Tier', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add score labels
    for i, score in enumerate(tier_scores):
        ax1.text(score + 0.02, i, f'{score:.3f}', va='center', fontsize=10)
    
    # Composite scores (radar chart would be better, but using bars for simplicity)
    composites = ['Overall', 'Quick\n(Fast Check)', 'Simulation\n(Physics)']
    composite_scores = [overall, quick, sim]
    colors_comp = ['#1abc9c', '#3498db', '#e67e22']
    
    ax2.bar(composites, composite_scores, color=colors_comp, alpha=0.8, width=0.6)
    ax2.set_ylim([0, 1])
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Composite Quality Scores', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add score labels
    for i, score in enumerate(composite_scores):
        ax2.text(i, score + 0.02, f'{score:.3f}', ha='center', fontsize=10, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'Scene Quality Assessment: {scene_id[:8]}...', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Generate output path if not provided
    if output_path is None:
        scene_path = Path(scene_json_path)
        output_path = scene_path.parent / f"{scene_path.stem}_quality_metrics.png"
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Quality metrics visualization saved to: {output_path}")
    return str(output_path)


def create_visualization_summary(
    scene_dir: str,
    database_path: str = "/workspace/data/scene_index.db"
) -> Dict[str, str]:
    """
    Create all visualizations for a completed scene.
    
    Args:
        scene_dir: Directory containing scene.json and related files
        database_path: Path to quality metrics database
        
    Returns:
        Dictionary mapping visualization type to file path
    """
    scene_dir = Path(scene_dir)
    scene_json = scene_dir / "scene.json"
    
    if not scene_json.exists():
        logger.error(f"scene.json not found in {scene_dir}")
        return {}
    
    visualizations = {}
    
    # 3D scene visualization
    try:
        vis_path = visualize_scene_3d(str(scene_json))
        if vis_path:
            visualizations['3d_scene'] = vis_path
    except Exception as e:
        logger.error(f"Failed to create 3D visualization: {e}")
    
    # Quality metrics visualization
    try:
        vis_path = visualize_quality_metrics(str(scene_json), database_path)
        if vis_path:
            visualizations['quality_metrics'] = vis_path
    except Exception as e:
        logger.error(f"Failed to create quality metrics visualization: {e}")
    
    return visualizations


def copy_visualizations_to_host(scene_dir: str, scene_id: str) -> Dict[str, str]:
    """
    Copy visualization files from scene directory to host-mounted visualizations directory.
    
    Args:
        scene_dir: Path to the scene directory inside container
        scene_id: Scene ID for organizing visualizations
        
    Returns:
        Dictionary mapping visualization types to host paths
    """
    import shutil
    from datetime import datetime
    
    scene_path = Path(scene_dir)
    host_viz_dir = Path("/workspace/visualizations")
    host_viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create scene-specific directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    host_scene_dir = host_viz_dir / f"{scene_id}_{timestamp}"
    host_scene_dir.mkdir(parents=True, exist_ok=True)
    
    copied_files = {}
    
    # Find and copy all visualization files
    viz_patterns = [
        "*.png",
        "*.jpg",
        "*.jpeg",
        "sam3*.png",
        "*visualization*.png",
        "*overlay*.png",
        "*results*.png"
    ]
    
    for pattern in viz_patterns:
        for viz_file in scene_path.glob(pattern):
            if viz_file.is_file():
                dest_path = host_scene_dir / viz_file.name
                shutil.copy2(viz_file, dest_path)
                copied_files[viz_file.name] = str(dest_path)
                logger.info(f"Copied {viz_file.name} to {dest_path}")
    
    # Also copy scene.json for reference
    scene_json = scene_path / "scene.json"
    if scene_json.exists():
        dest_json = host_scene_dir / "scene.json"
        shutil.copy2(scene_json, dest_json)
        copied_files["scene.json"] = str(dest_json)
    
    logger.info(f"Visualizations copied to {host_scene_dir}")
    return copied_files

