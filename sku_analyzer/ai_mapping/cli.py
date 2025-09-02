"""Command-line interface for AI mapping operations."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .models import AIProcessingConfig
from .processor import AIMappingProcessor


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose: bool) -> None:
    """AI Mapping CLI for product data intelligence."""
    setup_logging(verbose)


@cli.command()
@click.argument('output_dir', type=click.Path(exists=True, path_type=Path))
@click.option('--parent', '-p', default='4301', help='Starting parent SKU')
@click.option('--single', '-s', is_flag=True, help='Process single parent only')
@click.option('--batch-size', '-b', default=10, help='Batch size for processing')
@click.option('--max-concurrent', '-c', default=5, help='Max concurrent requests')
@click.option('--temperature', '-t', default=0.1, help='AI model temperature')
def process(
    output_dir: Path,
    parent: str,
    single: bool,
    batch_size: int,
    max_concurrent: int,
    temperature: float
) -> None:
    """Process AI mapping for parent directories.
    
    OUTPUT_DIR: Directory containing parent folders and step3_mandatory_fields.json
    """
    
    async def run_processing():
        # Configure AI processing
        config = AIProcessingConfig(
            model_name="gemini-2.5-flash",
            temperature=temperature,
            batch_size=batch_size,
            max_concurrent=max_concurrent
        )
        
        # Initialize processor
        processor = AIMappingProcessor(config)
        
        try:
            if single:
                # Process single parent
                step3_path = output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
                step2_path = output_dir / f"parent_{parent}" / "step2_compressed.json"
                parent_output_dir = output_dir / f"parent_{parent}"
                
                result = await processor.process_parent_directory(
                    parent, step3_path, step2_path, parent_output_dir
                )
                
                click.echo(f"✅ Processed parent {parent}")
                click.echo(f"   Mapped: {result.get('mapped_fields_count', 0)} fields")
                click.echo(f"   Confidence: {result.get('confidence', 0.0):.2f}")
                click.echo(f"   Time: {result.get('processing_time_ms', 0):.1f}ms")
            else:
                # Process all parents
                summary = await processor.process_all_parents(output_dir, parent)
                
                click.echo(f"✅ AI Mapping Summary:")
                click.echo(f"   Total parents: {summary['summary']['total_parents']}")
                click.echo(f"   Successful: {summary['summary']['successful']}")
                click.echo(f"   Success rate: {summary['summary']['success_rate']:.1%}")
                click.echo(f"   Avg confidence: {summary['summary']['average_confidence']:.2f}")
                click.echo(f"   Total time: {summary['performance']['total_processing_time']:.1f}ms")
        
        except Exception as e:
            click.echo(f"❌ Processing failed: {e}", err=True)
            sys.exit(1)
    
    # Run async processing
    asyncio.run(run_processing())


@cli.command()
@click.argument('parent_sku')
@click.argument('output_dir', type=click.Path(exists=True, path_type=Path))
def test(parent_sku: str, output_dir: Path) -> None:
    """Test AI mapping with a specific parent SKU.
    
    PARENT_SKU: Parent SKU to test (e.g., 4301)
    OUTPUT_DIR: Directory containing required input files
    """
    
    async def run_test():
        config = AIProcessingConfig(temperature=0.1)
        processor = AIMappingProcessor(config)
        
        try:
            step3_path = output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
            step2_path = output_dir / f"parent_{parent_sku}" / "step2_compressed.json"
            parent_output_dir = output_dir / f"parent_{parent_sku}"
            
            result = await processor.process_parent_directory(
                parent_sku, step3_path, step2_path, parent_output_dir
            )
            
            if result["success"]:
                click.echo(f"✅ Test successful for parent {parent_sku}")
                click.echo(f"   Output: {result['output_file']}")
            else:
                click.echo(f"❌ Test failed: {result.get('error', 'Unknown error')}")
                sys.exit(1)
        
        except Exception as e:
            click.echo(f"❌ Test error: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_test())


if __name__ == '__main__':
    cli()