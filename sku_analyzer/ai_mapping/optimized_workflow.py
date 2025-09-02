"""Optimized AI mapping workflow with template reuse for 60-80% token reduction."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .template_generator import AITemplateGenerator, OptimizedDataMapper
from .models import AIProcessingConfig


class OptimizedAIMappingWorkflow:
    """Token-optimized AI mapping workflow using template reuse strategy."""
    
    def __init__(self, config: Optional[AIProcessingConfig] = None):
        """Initialize optimized workflow.
        
        Args:
            config: AI processing configuration
        """
        self.config = config or AIProcessingConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.template_generator = AITemplateGenerator(self.config)
        self.data_mapper = OptimizedDataMapper(self.config)
        
        # Performance tracking
        self.workflow_stats = {
            "total_execution_time": 0.0,
            "template_generation_time": 0.0,
            "data_mapping_time": 0.0,
            "token_optimization": {
                "baseline_tokens_estimated": 0,
                "optimized_tokens_actual": 0,
                "reduction_achieved_percent": 0.0
            },
            "parents_processed": 0
        }
    
    async def execute_optimized_mapping(
        self,
        base_output_dir: Path,
        starting_parent: str = "4301"
    ) -> Dict[str, Any]:
        """Execute token-optimized AI mapping workflow.
        
        Args:
            base_output_dir: Base output directory with parent folders
            starting_parent: Starting parent for validation
            
        Returns:
            Complete workflow results with optimization metrics
        """
        workflow_start = time.time()
        self.logger.info("🚀 Starting optimized AI mapping workflow")
        
        try:
            # Step 3: Generate reusable template (ONE TIME)
            template_result = await self._step3_generate_template(base_output_dir)
            
            # Step 4: Map all parent data using template (REUSE)
            mapping_results = await self._step4_map_all_parents(
                base_output_dir, template_result["template_path"], starting_parent
            )
            
            # Calculate final optimization metrics
            optimization_metrics = self._calculate_optimization_metrics(
                template_result, mapping_results
            )
            
            # Generate comprehensive workflow summary
            total_time = time.time() - workflow_start
            workflow_summary = self._generate_workflow_summary(
                total_time, template_result, mapping_results, optimization_metrics
            )
            
            self.logger.info(
                f"✅ Optimized workflow complete: {optimization_metrics['token_reduction_achieved']:.1f}% token reduction"
            )
            
            return workflow_summary
            
        except Exception as e:
            self.logger.error(f"Optimized workflow failed: {e}")
            raise
    
    async def _step3_generate_template(self, base_output_dir: Path) -> Dict[str, Any]:
        """Step 3: Generate reusable AI mapping template.
        
        Args:
            base_output_dir: Base output directory
            
        Returns:
            Template generation result
        """
        step3_start = time.time()
        self.logger.info("📋 Step 3: Generating reusable AI mapping template")
        
        # Load mandatory fields from flat file analysis
        step3_mandatory_path = base_output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
        
        if not step3_mandatory_path.exists():
            raise FileNotFoundError(f"Missing mandatory fields: {step3_mandatory_path}")
        
        with step3_mandatory_path.open('r', encoding='utf-8') as f:
            mandatory_fields = json.load(f)
        
        # Load flat file metadata for context
        step2_values_path = base_output_dir / "flat_file_analysis" / "step2_valid_values.json"
        flat_file_metadata = {}
        
        if step2_values_path.exists():
            with step2_values_path.open('r', encoding='utf-8') as f:
                step2_data = json.load(f)
                flat_file_metadata = {
                    "extraction_metadata": step2_data.get("extraction_metadata", {}),
                    "field_count": len(step2_data.get("field_validations", {}))
                }
        
        # Generate template
        template_output_path = base_output_dir / "ai_optimization" / "mapping_template.json"
        
        template_result = await self.template_generator.generate_mapping_template(
            mandatory_fields, flat_file_metadata, template_output_path
        )
        
        # Update workflow stats
        self.workflow_stats["template_generation_time"] = time.time() - step3_start
        
        self.logger.info(
            f"✅ Step 3 complete: Template generated in {self.workflow_stats['template_generation_time']:.2f}s"
        )
        
        return template_result
    
    async def _step4_map_all_parents(
        self,
        base_output_dir: Path,
        template_path: str,
        starting_parent: str
    ) -> List[Dict[str, Any]]:
        """Step 4: Map all parent data using generated template.
        
        Args:
            base_output_dir: Base output directory
            template_path: Path to generated template
            starting_parent: Starting parent SKU
            
        Returns:
            List of mapping results for all parents
        """
        step4_start = time.time()
        self.logger.info("🎯 Step 4: Mapping all parents with template reuse")
        
        # Load mandatory fields for validation
        step3_mandatory_path = base_output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
        with step3_mandatory_path.open('r', encoding='utf-8') as f:
            mandatory_fields = json.load(f)
        
        # Find all parent directories
        parent_dirs = self._find_parent_directories(base_output_dir)
        
        if not parent_dirs:
            raise ValueError(f"No parent directories found in {base_output_dir}")
        
        # Process starting parent first
        results = []
        if starting_parent in parent_dirs:
            self.logger.info(f"🎯 Processing starting parent {starting_parent}")
            
            result = await self._map_single_parent(
                starting_parent, base_output_dir, Path(template_path), mandatory_fields
            )
            results.append(result)
            
            # Process remaining parents if successful
            if result["success"]:
                remaining_parents = [p for p in parent_dirs if p != starting_parent]
                remaining_results = await self._map_parents_batch(
                    remaining_parents, base_output_dir, Path(template_path), mandatory_fields
                )
                results.extend(remaining_results)
        else:
            # Process all parents
            all_results = await self._map_parents_batch(
                parent_dirs, base_output_dir, Path(template_path), mandatory_fields
            )
            results.extend(all_results)
        
        # Update workflow stats
        self.workflow_stats["data_mapping_time"] = time.time() - step4_start
        self.workflow_stats["parents_processed"] = len(results)
        
        self.logger.info(
            f"✅ Step 4 complete: {len(results)} parents mapped in {self.workflow_stats['data_mapping_time']:.2f}s"
        )
        
        return results
    
    async def _map_single_parent(
        self,
        parent_sku: str,
        base_output_dir: Path,
        template_path: Path,
        mandatory_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map single parent using template.
        
        Args:
            parent_sku: Parent SKU to process
            base_output_dir: Base output directory
            template_path: Template file path
            mandatory_fields: Field constraints
            
        Returns:
            Mapping result for parent
        """
        # Load parent data
        step2_path = base_output_dir / f"parent_{parent_sku}" / "step2_compressed.json"
        
        if not step2_path.exists():
            return {
                "parent_sku": parent_sku,
                "success": False,
                "error": f"Missing compressed data: {step2_path}"
            }
        
        with step2_path.open('r', encoding='utf-8') as f:
            product_data = json.load(f)
        
        # Map using template
        output_path = base_output_dir / f"parent_{parent_sku}" / "step4_optimized_mapping.json"
        
        return await self.data_mapper.map_parent_data_with_template(
            parent_sku, product_data, template_path, mandatory_fields, output_path
        )
    
    async def _map_parents_batch(
        self,
        parent_skus: List[str],
        base_output_dir: Path,
        template_path: Path,
        mandatory_fields: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Map multiple parents in controlled batches.
        
        Args:
            parent_skus: List of parent SKUs to process
            base_output_dir: Base output directory
            template_path: Template file path
            mandatory_fields: Field constraints
            
        Returns:
            List of mapping results
        """
        # Control concurrency for API rate limits
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def map_with_semaphore(parent_sku: str) -> Dict[str, Any]:
            async with semaphore:
                return await self._map_single_parent(
                    parent_sku, base_output_dir, template_path, mandatory_fields
                )
        
        # Execute all mappings
        tasks = [map_with_semaphore(sku) for sku in parent_skus]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _find_parent_directories(self, base_dir: Path) -> List[str]:
        """Find parent directories with required files.
        
        Args:
            base_dir: Base directory to search
            
        Returns:
            List of parent SKU identifiers
        """
        parent_skus = []
        
        for parent_dir in base_dir.glob("parent_*"):
            if parent_dir.is_dir():
                step2_file = parent_dir / "step2_compressed.json"
                if step2_file.exists():
                    parent_sku = parent_dir.name.replace("parent_", "")
                    parent_skus.append(parent_sku)
        
        return parent_skus
    
    def _calculate_optimization_metrics(
        self,
        template_result: Dict[str, Any],
        mapping_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate comprehensive optimization metrics.
        
        Args:
            template_result: Template generation result
            mapping_results: List of mapping results
            
        Returns:
            Optimization metrics summary
        """
        successful_mappings = [r for r in mapping_results if r.get("success", False)]
        
        # Get performance stats from components
        template_stats = self.template_generator.get_performance_stats()
        mapping_stats = self.data_mapper.get_performance_stats()
        
        # Calculate token optimization
        baseline_tokens_estimated = len(successful_mappings) * 28000  # Current approach estimate
        optimized_tokens_actual = (
            template_stats["token_usage"]["template_generation_tokens"] +
            mapping_stats["token_usage"]["total_tokens_used"]
        )
        
        token_reduction = 0.0
        if baseline_tokens_estimated > 0:
            token_reduction = (
                (baseline_tokens_estimated - optimized_tokens_actual) / baseline_tokens_estimated
            ) * 100
        
        return {
            "token_reduction_achieved": token_reduction,
            "baseline_tokens_estimated": baseline_tokens_estimated,
            "optimized_tokens_actual": optimized_tokens_actual,
            "tokens_saved": baseline_tokens_estimated - optimized_tokens_actual,
            "successful_mappings": len(successful_mappings),
            "failed_mappings": len(mapping_results) - len(successful_mappings),
            "template_generation": {
                "time_ms": template_result["generation_time_ms"],
                "tokens_used": template_stats["token_usage"]["template_generation_tokens"]
            },
            "data_mapping": {
                "average_time_per_parent_ms": mapping_stats["average_mapping_time"] * 1000,
                "average_tokens_per_parent": mapping_stats["token_usage"]["average_tokens_per_parent"]
            }
        }
    
    def _generate_workflow_summary(
        self,
        total_time: float,
        template_result: Dict[str, Any],
        mapping_results: List[Dict[str, Any]],
        optimization_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive workflow summary.
        
        Args:
            total_time: Total workflow execution time
            template_result: Template generation result
            mapping_results: List of mapping results
            optimization_metrics: Calculated optimization metrics
            
        Returns:
            Complete workflow summary
        """
        successful_mappings = [r for r in mapping_results if r.get("success", False)]
        
        return {
            "workflow_metadata": {
                "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_execution_time_seconds": total_time,
                "optimization_strategy": "template_reuse"
            },
            "performance_summary": {
                "parents_processed": len(mapping_results),
                "successful_mappings": len(successful_mappings),
                "success_rate": len(successful_mappings) / len(mapping_results) if mapping_results else 0.0,
                "average_mapping_time_seconds": optimization_metrics["data_mapping"]["average_time_per_parent_ms"] / 1000
            },
            "token_optimization": {
                "target_reduction_percent": 70.0,  # Target 60-80%
                "achieved_reduction_percent": optimization_metrics["token_reduction_achieved"],
                "target_met": optimization_metrics["token_reduction_achieved"] >= 60.0,
                "baseline_tokens_estimated": optimization_metrics["baseline_tokens_estimated"],
                "optimized_tokens_actual": optimization_metrics["optimized_tokens_actual"],
                "total_tokens_saved": optimization_metrics["tokens_saved"]
            },
            "step_breakdown": {
                "step3_template_generation": {
                    "duration_seconds": self.workflow_stats["template_generation_time"],
                    "tokens_used": template_result["optimization_metrics"]["tokens_saved_per_parent"],
                    "templates_created": 1,
                    "reuse_factor": len(successful_mappings)
                },
                "step4_data_mapping": {
                    "duration_seconds": self.workflow_stats["data_mapping_time"],
                    "average_tokens_per_parent": optimization_metrics["data_mapping"]["average_tokens_per_parent"],
                    "parents_mapped": len(successful_mappings),
                    "template_reuse_efficiency": optimization_metrics["token_reduction_achieved"]
                }
            },
            "optimization_validation": {
                "performance_targets": {
                    "token_reduction_60_percent": optimization_metrics["token_reduction_achieved"] >= 60.0,
                    "average_mapping_time_5s": optimization_metrics["data_mapping"]["average_time_per_parent_ms"] < 5000,
                    "template_generation_30s": self.workflow_stats["template_generation_time"] < 30.0,
                    "overall_workflow_90s": total_time < 90.0
                },
                "bottleneck_analysis": self._identify_workflow_bottlenecks(optimization_metrics),
                "recommendations": self._generate_optimization_recommendations(optimization_metrics)
            },
            "detailed_results": {
                "template_generation": template_result,
                "parent_mappings": mapping_results,
                "component_stats": {
                    "template_generator": self.template_generator.get_performance_stats(),
                    "data_mapper": self.data_mapper.get_performance_stats()
                }
            }
        }
    
    def _identify_workflow_bottlenecks(
        self,
        optimization_metrics: Dict[str, Any]
    ) -> List[str]:
        """Identify performance bottlenecks in optimized workflow.
        
        Args:
            optimization_metrics: Calculated optimization metrics
            
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Template generation bottlenecks
        if self.workflow_stats["template_generation_time"] > 20.0:
            bottlenecks.append(
                f"Template generation slow: {self.workflow_stats['template_generation_time']:.1f}s"
            )
        
        # Data mapping bottlenecks
        avg_mapping_time = optimization_metrics["data_mapping"]["average_time_per_parent_ms"]
        if avg_mapping_time > 4000:  # >4s per parent
            bottlenecks.append(
                f"Data mapping per parent slow: {avg_mapping_time:.0f}ms average"
            )
        
        # Token optimization bottlenecks
        token_reduction = optimization_metrics["token_reduction_achieved"]
        if token_reduction < 60.0:
            bottlenecks.append(
                f"Token reduction below target: {token_reduction:.1f}% (target: 60-80%)"
            )
        
        # API performance bottlenecks
        mapper_stats = self.data_mapper.get_performance_stats()
        total_tokens = mapper_stats["token_usage"]["total_tokens_used"]
        if total_tokens > 50000:  # Arbitrary threshold
            bottlenecks.append(
                f"High total token usage: {total_tokens:,} tokens"
            )
        
        return bottlenecks
    
    def _generate_optimization_recommendations(
        self,
        optimization_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate optimization recommendations.
        
        Args:
            optimization_metrics: Calculated metrics
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Token optimization recommendations
        token_reduction = optimization_metrics["token_reduction_achieved"]
        if token_reduction < 70.0:
            recommendations.append("Further compress template structure to increase token savings")
            recommendations.append("Implement constraint lookup tables instead of inline validation")
        
        # Performance recommendations
        if self.workflow_stats["template_generation_time"] > 15.0:
            recommendations.append("Optimize template generation prompt for faster AI response")
            
        avg_mapping_time = optimization_metrics["data_mapping"]["average_time_per_parent_ms"]
        if avg_mapping_time > 3000:
            recommendations.append("Implement parallel processing for data mapping step")
            recommendations.append("Cache common field transformations to reduce AI processing")
        
        # Scalability recommendations
        if optimization_metrics["successful_mappings"] < 6:
            recommendations.append("Add error recovery strategies for failed mappings")
            
        return recommendations
    
    async def benchmark_optimization_effectiveness(
        self,
        base_output_dir: Path,
        baseline_approach_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Benchmark optimization effectiveness against baseline.
        
        Args:
            base_output_dir: Base output directory for testing
            baseline_approach_results: Previous baseline results for comparison
            
        Returns:
            Comprehensive optimization benchmark
        """
        self.logger.info("📊 Benchmarking optimization effectiveness")
        
        # Run optimized workflow
        optimized_results = await self.execute_optimized_mapping(base_output_dir)
        
        # Compare with baseline if available
        comparison_metrics = {}
        if baseline_approach_results:
            comparison_metrics = self._compare_with_baseline(
                optimized_results, baseline_approach_results
            )
        
        benchmark_results = {
            "benchmark_metadata": {
                "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "optimization_strategy": "template_reuse",
                "baseline_available": baseline_approach_results is not None
            },
            "optimization_effectiveness": {
                "token_reduction_achieved": optimized_results["token_optimization"]["achieved_reduction_percent"],
                "performance_improvement": optimized_results["performance_summary"],
                "targets_achieved": optimized_results["optimization_validation"]["performance_targets"]
            },
            "detailed_comparison": comparison_metrics,
            "optimization_results": optimized_results
        }
        
        self.logger.info(
            f"📊 Benchmark complete: {optimized_results['token_optimization']['achieved_reduction_percent']:.1f}% optimization achieved"
        )
        
        return benchmark_results
    
    def _compare_with_baseline(
        self,
        optimized_results: Dict[str, Any],
        baseline_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare optimized results with baseline approach.
        
        Args:
            optimized_results: Results from optimized workflow
            baseline_results: Results from baseline approach
            
        Returns:
            Comparison metrics
        """
        # Extract key metrics for comparison
        optimized_time = optimized_results["workflow_metadata"]["total_execution_time_seconds"]
        baseline_time = baseline_results.get("total_duration_seconds", 0)
        
        optimized_tokens = optimized_results["token_optimization"]["optimized_tokens_actual"]
        baseline_tokens = baseline_results.get("estimated_tokens_used", 0)
        
        time_improvement = 0.0
        if baseline_time > 0:
            time_improvement = ((baseline_time - optimized_time) / baseline_time) * 100
        
        token_improvement = 0.0
        if baseline_tokens > 0:
            token_improvement = ((baseline_tokens - optimized_tokens) / baseline_tokens) * 100
        
        return {
            "execution_time": {
                "baseline_seconds": baseline_time,
                "optimized_seconds": optimized_time,
                "improvement_percent": time_improvement
            },
            "token_usage": {
                "baseline_tokens": baseline_tokens,
                "optimized_tokens": optimized_tokens,
                "reduction_percent": token_improvement
            },
            "overall_efficiency": {
                "performance_improvement": time_improvement,
                "resource_optimization": token_improvement,
                "efficiency_score": (time_improvement + token_improvement) / 2
            }
        }
    
    def get_workflow_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive workflow performance statistics.
        
        Returns:
            Complete performance statistics
        """
        return {
            "workflow_stats": dict(self.workflow_stats),
            "template_generator_stats": self.template_generator.get_performance_stats(),
            "data_mapper_stats": self.data_mapper.get_performance_stats()
        }