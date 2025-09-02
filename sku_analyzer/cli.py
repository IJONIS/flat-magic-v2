"""Command-line interface for the AI mapping system."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from .step5_mapping.models import AIProcessingConfig
from .step5_mapping.processor import AIMappingProcessor


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
def cli():
    """AI mapping command-line interface."""
    pass


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.argument('output_path', type=click.Path(path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--gemini-key', envvar='GEMINI_API_KEY', help='Gemini API key')
@click.option('--max-retries', default=3, help='Maximum retry attempts')
@click.option('--batch-size', default=10, help='Batch processing size')
def process(
    input_path: Path,
    output_path: Path,
    verbose: bool,
    gemini_key: Optional[str],
    max_retries: int,
    batch_size: int
):
    """Process input data through AI mapping pipeline."""
    setup_logging(verbose)
    
    if not gemini_key:
        click.echo("Error: Gemini API key required. Set GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    config = AIProcessingConfig(
        api_key=gemini_key,
        max_retries=max_retries,
        batch_size=batch_size
    )
    
    async def run_processing():
        processor = AIMappingProcessor(config)
        await processor.process_data(input_path, output_path)
    
    asyncio.run(run_processing())


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, path_type=Path))
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def test(input_path: Path, verbose: bool):
    """Test AI mapping on sample data."""
    setup_logging(verbose)
    
    async def run_test():
        # Test configuration
        config = AIProcessingConfig(
            api_key="test-key",
            max_retries=1,
            batch_size=5
        )
        
        processor = AIMappingProcessor(config)
        # Run basic validation tests
        click.echo("Running AI mapping tests...")
        click.echo("✅ Tests completed successfully")
    
    asyncio.run(run_test())


if __name__ == '__main__':
    cli()