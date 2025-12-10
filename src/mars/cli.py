import click
import logging
import sys
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler

# Import Pipeline
from src.mars.pipeline import MARSPipeline

# Setup Rich Console
console = Console()

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )

@click.group()
def cli():
    """MARS: Multi Asset Reconstruction for Simulation"""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.option('--output-dir', '-o', default="data/scenes", help="Output directory for scenes")
@click.option('--sam-checkpoint', default="checkpoints/sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
@click.option('--sam3d-checkpoint', default="checkpoints/sam3d.pth", help="Path to SAM 3D checkpoint")
@click.option('--debug', is_flag=True, help="Enable debug logging")
def process(image_path, output_dir, sam_checkpoint, sam3d_checkpoint, debug):
    """Process a single image through the full pipeline."""
    setup_logging(logging.DEBUG if debug else logging.INFO)
    
    console.rule("[bold green]Starting MARS Pipeline")
    console.print(f"Input: {image_path}")
    console.print(f"Output: {output_dir}")
    
    # Initialize Pipeline
    try:
        pipeline = MARSPipeline(
            sam_checkpoint=sam_checkpoint,
            sam3d_checkpoint=sam3d_checkpoint,
            output_dir=output_dir
        )
        
        # Run
        result = pipeline.process(Path(image_path))
        
        if result.get("status") == "complete":
            console.rule("[bold green]Success")
            console.print(f"Scene ID: {result['image_id']}")
            console.print(f"Final Path: {result['final_path']}")
        else:
            console.rule("[bold red]Failed")
            console.print(f"Status: {result.get('status')}")
            console.print(f"Reason: {result.get('reason')}")
            sys.exit(1)
            
    except Exception as e:
        console.print_exception()
        sys.exit(1)

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
def ingest(image_path):
    """Run only the ingestion stage."""
    setup_logging()
    from src.mars.ingestion import ingest_image
    result = ingest_image(image_path)
    console.print(result)

@cli.command()
@click.argument('scene_dir', type=click.Path(exists=True))
@click.option('--database', '-db', default="/workspace/data/scene_index.db", help="Path to scene database")
@click.option('--show', is_flag=True, help="Open visualizations after creating them")
def visualize(scene_dir, database, show):
    """Create visualizations for a completed scene."""
    setup_logging()
    from src.mars.utils.visualize import create_visualization_summary, visualize_scene_3d, visualize_quality_metrics
    import json
    
    scene_dir = Path(scene_dir)
    scene_json = scene_dir / "scene.json"
    
    if not scene_json.exists():
        console.print(f"[red]Error: scene.json not found in {scene_dir}[/red]")
        sys.exit(1)
    
    console.rule("[bold cyan]Creating Visualizations")
    console.print(f"Scene directory: {scene_dir}")
    
    try:
        # Load scene info
        with open(scene_json, 'r') as f:
            scene = json.load(f)
        
        scene_id = scene['scene_id']
        console.print(f"Scene ID: {scene_id}")
        console.print(f"Objects: {len(scene.get('objects', []))}")
        
        # Create 3D visualization
        console.print("\n[cyan]Generating 3D scene visualization...[/cyan]")
        vis_3d = visualize_scene_3d(str(scene_json))
        if vis_3d:
            console.print(f"✓ 3D view: {vis_3d}")
        
        # Create quality metrics visualization
        if Path(database).exists():
            console.print("\n[cyan]Generating quality metrics visualization...[/cyan]")
            vis_quality = visualize_quality_metrics(str(scene_json), database)
            if vis_quality:
                console.print(f"✓ Quality metrics: {vis_quality}")
        else:
            console.print(f"[yellow]Warning: Database not found at {database}[/yellow]")
        
        # Copy visualizations to host-mounted directory
        from src.mars.utils.visualize import copy_visualizations_to_host
        console.print("\n[cyan]Copying visualizations to host...[/cyan]")
        try:
            copied = copy_visualizations_to_host(str(scene_dir), scene_id)
            if copied:
                host_path = list(copied.values())[0]
                host_dir = Path(host_path).parent
                console.print(f"[green]✓ Visualizations copied to: visualizations/{host_dir.name}/[/green]")
                console.print(f"   Host path: {host_dir}")
            else:
                console.print("[yellow]No visualization files found to copy[/yellow]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not copy to host: {e}[/yellow]")
        
        console.rule("[bold green]Visualizations Complete")
        
        if show:
            import subprocess
            if vis_3d:
                subprocess.run(['xdg-open', vis_3d], check=False)
            if vis_quality:
                subprocess.run(['xdg-open', vis_quality], check=False)
                
    except Exception as e:
        console.print_exception()
        sys.exit(1)

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
@click.argument('masks_file', type=click.Path(exists=True))
@click.option('--output', '-o', help="Output path for visualization")
@click.option('--show', is_flag=True, help="Open visualization after creating it")
def visualize_masks(image_path, masks_file, output, show):
    """Visualize segmentation masks on an image."""
    setup_logging()
    from src.mars.utils.visualize import visualize_segmentation
    import json
    
    console.rule("[bold cyan]Visualizing Segmentation Masks")
    console.print(f"Image: {image_path}")
    console.print(f"Masks: {masks_file}")
    
    try:
        # Load masks
        with open(masks_file, 'r') as f:
            data = json.load(f)
        
        masks = data.get('masks', [])
        console.print(f"Found {len(masks)} masks")
        
        # Create visualization
        vis_path = visualize_segmentation(image_path, masks, output_path=output)
        console.print(f"[green]✓ Visualization saved: {vis_path}[/green]")
        
        if show:
            import subprocess
            subprocess.run(['xdg-open', vis_path], check=False)
            
    except Exception as e:
        console.print_exception()
        sys.exit(1)

if __name__ == '__main__':
    cli()

