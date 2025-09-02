"""Integration example demonstrating the new modular architecture.

This module shows how to use the restructured AI mapping pipeline
with the new step3_template and step4_mapping modules.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

from .step3_template import TemplateGenerator
from .step4_mapping import MappingProcessor, ProcessingConfig
from .shared import GeminiClient, PerformanceMonitor, AIProcessingConfig
from .prompts import MappingPromptManager, CategorizationPromptManager


class ModularPipelineOrchestrator:
    """Orchestrates the modular AI mapping pipeline.
    
    This class demonstrates how to use the new modular architecture
    to coordinate template generation and AI mapping operations.
    """
    
    def __init__(self, enable_performance_monitoring: bool = True):
        """Initialize pipeline orchestrator.
        
        Args:
            enable_performance_monitoring: Whether to enable performance tracking
        """
        self.logger = logging.getLogger(__name__)
        self.performance_monitor = PerformanceMonitor(enable_performance_monitoring)
        
        # Initialize AI configuration
        self.ai_config = AIProcessingConfig(
            model_name="gemini-2.5-flash",
            temperature=0.1,
            max_tokens=4096,
            timeout_seconds=30,
            max_concurrent=1
        )
        
        # Initialize components
        self.template_generator = TemplateGenerator(
            enable_performance_monitoring=enable_performance_monitoring,
            enable_ai_categorization=True
        )
        
        self.mapping_processor = MappingProcessor(
            config=ProcessingConfig(),
            ai_config=self.ai_config,
            enable_performance_monitoring=enable_performance_monitoring
        )
        
        # Initialize prompt managers
        self.mapping_prompts = MappingPromptManager()
        self.categorization_prompts = CategorizationPromptManager()
    
    async def run_complete_pipeline(
        self,
        step3_mandatory_path: Path,
        base_output_dir: Path,
        starting_parent: str = "4301"
    ) -> Dict[str, Any]:
        """Run the complete modular pipeline.
        
        Args:
            step3_mandatory_path: Path to step3_mandatory_fields.json
            base_output_dir: Base output directory
            starting_parent: Starting parent SKU
            
        Returns:
            Complete pipeline results
        """
        pipeline_results = {}
        
        with self.performance_monitor.measure_performance("complete_pipeline") as perf:
            try:
                # Step 1: Generate template from mandatory fields
                self.logger.info("🔧 Starting template generation (Step 3)")
                
                template_output_path = base_output_dir / "flat_file_analysis" / "step4_template.json"
                template_result = await self.template_generator.generate_template_from_mandatory_fields(
                    step3_mandatory_path=step3_mandatory_path,
                    output_path=template_output_path
                )
                
                pipeline_results['template_generation'] = {
                    'success': True,
                    'output_file': str(template_output_path),
                    'quality_score': template_result['metadata']['quality_score'],
                    'field_distribution': template_result['metadata']['field_distribution'],
                    'categorization_method': template_result['metadata']['categorization_method']
                }
                
                self.logger.info(f"✅ Template generation completed with quality score: {template_result['metadata']['quality_score']:.2f}")
                
                # Step 2: AI mapping using generated template
                self.logger.info("🤖 Starting AI mapping (Step 4)")
                
                mapping_results = await self.mapping_processor.process_all_parents(
                    base_output_dir=base_output_dir,
                    starting_parent=starting_parent
                )
                
                pipeline_results['ai_mapping'] = {
                    'success': mapping_results.summary['success_rate'] > 0,
                    'summary': mapping_results.summary,
                    'performance': mapping_results.performance,
                    'total_parents_processed': len(mapping_results.results)
                }
                
                self.logger.info(
                    f"✅ AI mapping completed: {mapping_results.summary['successful']}/{mapping_results.summary['total_parents']} parents successful"
                )
                
                # Step 3: Generate performance summary
                pipeline_results['overall_performance'] = {
                    'total_duration_ms': perf['metrics'].duration_ms if perf['metrics'] else 0,
                    'peak_memory_mb': perf['metrics'].peak_memory_mb if perf['metrics'] else 0,
                    'components_performance': {
                        'template_generation': self.template_generator.performance_monitor.get_system_info(),
                        'ai_mapping': self.mapping_processor.get_performance_stats()
                    }
                }
                
                return pipeline_results
                
            except Exception as e:
                self.logger.error(f"Pipeline execution failed: {e}")
                pipeline_results['error'] = str(e)
                pipeline_results['success'] = False
                return pipeline_results
    
    async def run_template_generation_only(
        self,
        step3_mandatory_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """Run only the template generation step.
        
        Args:
            step3_mandatory_path: Path to step3_mandatory_fields.json
            output_path: Output path for template
            
        Returns:
            Template generation results
        """
        self.logger.info("🔧 Running template generation only")
        
        try:
            template_result = await self.template_generator.generate_template_from_mandatory_fields(
                step3_mandatory_path=step3_mandatory_path,
                output_path=output_path
            )
            
            return {
                'success': True,
                'template_result': template_result,
                'output_file': str(output_path),
                'categorization_method': template_result['metadata']['categorization_method'],
                'quality_score': template_result['metadata']['quality_score']
            }
            
        except Exception as e:
            self.logger.error(f"Template generation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def run_mapping_only(
        self,
        template_path: Path,
        base_output_dir: Path,
        starting_parent: str = "4301"
    ) -> Dict[str, Any]:
        """Run only the AI mapping step.
        
        Args:
            template_path: Path to existing template
            base_output_dir: Base output directory
            starting_parent: Starting parent SKU
            
        Returns:
            AI mapping results
        """
        self.logger.info("🤖 Running AI mapping only")
        
        try:
            # Verify template exists
            if not template_path.exists():
                raise FileNotFoundError(f"Template not found: {template_path}")
            
            mapping_results = await self.mapping_processor.process_all_parents(
                base_output_dir=base_output_dir,
                starting_parent=starting_parent
            )
            
            return {
                'success': True,
                'mapping_results': mapping_results,
                'summary': mapping_results.summary,
                'performance': mapping_results.performance
            }
            
        except Exception as e:
            self.logger.error(f"AI mapping failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all pipeline components.
        
        Returns:
            Component status dictionary
        """
        return {
            'template_generator': {
                'ai_categorization_enabled': self.template_generator.enable_ai,
                'performance_monitoring_enabled': self.template_generator.enable_monitoring
            },
            'mapping_processor': {
                'config': self.mapping_processor.config.model_dump(),
                'ai_config': self.mapping_processor.ai_config.model_dump(),
                'performance_stats': self.mapping_processor.get_performance_stats()
            },
            'shared_components': {
                'performance_monitor': self.performance_monitor.get_system_info(),
                'ai_config': self.ai_config.model_dump()
            }
        }


async def main():
    """Example usage of the modular pipeline."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting modular pipeline integration example")
    
    # Initialize orchestrator
    orchestrator = ModularPipelineOrchestrator(enable_performance_monitoring=True)
    
    # Define paths (adjust these to your actual paths)
    step3_mandatory_path = Path("production_output/1756744145/flat_file_analysis/step3_mandatory_fields.json")
    base_output_dir = Path("production_output/1756744145")
    
    try:
        # Option 1: Run complete pipeline
        logger.info("Running complete modular pipeline...")
        results = await orchestrator.run_complete_pipeline(
            step3_mandatory_path=step3_mandatory_path,
            base_output_dir=base_output_dir,
            starting_parent="4301"
        )
        
        # Display results
        if results.get('success', True):  # True if no explicit success field
            logger.info("🎉 Pipeline completed successfully!")
            
            # Template generation results
            template_results = results.get('template_generation', {})
            if template_results.get('success'):
                logger.info(
                    f"📋 Template: Quality {template_results['quality_score']:.2f}, "
                    f"Method {template_results['categorization_method']}"
                )
            
            # AI mapping results
            mapping_results = results.get('ai_mapping', {})
            if mapping_results.get('success'):
                summary = mapping_results['summary']
                logger.info(
                    f"🤖 Mapping: {summary['successful']}/{summary['total_parents']} parents, "
                    f"Success rate {summary['success_rate']:.1%}"
                )
            
            # Performance summary
            perf = results.get('overall_performance', {})
            if perf:
                logger.info(
                    f"⚡ Performance: {perf['total_duration_ms']:.0f}ms, "
                    f"Peak memory {perf['peak_memory_mb']:.1f}MB"
                )
        else:
            logger.error(f"❌ Pipeline failed: {results.get('error')}")
        
        # Get component status
        status = orchestrator.get_component_status()
        logger.info(f"📊 Component Status: {len(status)} components active")
        
    except Exception as e:
        logger.error(f"❌ Integration example failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())