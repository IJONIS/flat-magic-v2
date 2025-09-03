"""Performance testing script for optimized Gemini API implementation.

This script validates the performance improvements in:
- API response times (target: <5s per request)
- Safety filter compliance (target: <5% rejection rate)
- Batch processing efficiency
- Memory usage and prompt compression
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from sku_analyzer.shared.gemini_client import GeminiClient, AIProcessingConfig
from sku_analyzer.step5_mapping.processor import MappingProcessor
from sku_analyzer.step5_mapping.models import ProcessingConfig


# Configure logging for performance testing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceTestSuite:
    """Comprehensive performance testing for optimized Gemini API implementation."""
    
    def __init__(self):
        """Initialize performance test suite with optimized configurations."""
        # Performance-optimized AI configuration
        self.ai_config = AIProcessingConfig(
            max_tokens=2048,
            timeout_seconds=15,
            max_concurrent=3,
            max_retries=1,
            batch_size=3,
            max_prompt_size=8000,
            max_variants_per_request=5,
            enable_prompt_compression=True
        )
        
        # Processing configuration with fast settings
        self.processing_config = ProcessingConfig(
            max_retries=1,
            timeout_seconds=15,
            batch_size=3,
            confidence_threshold=0.5
        )
        
        self.results = {
            'api_performance': {},
            'batch_performance': {},
            'compression_performance': {},
            'safety_compliance': {},
            'overall_assessment': {}
        }
    
    async def run_comprehensive_performance_tests(self, output_dir: Path) -> dict:
        """Run comprehensive performance tests and return results.
        
        Args:
            output_dir: Directory containing test data
            
        Returns:
            Performance test results
        """
        logger.info("Starting comprehensive performance test suite")
        start_time = time.perf_counter()
        
        try:
            # Test 1: API Response Time Performance
            logger.info("Test 1: API response time performance")
            await self._test_api_response_times(output_dir)
            
            # Test 2: Batch Processing Performance
            logger.info("Test 2: Batch processing performance")
            await self._test_batch_processing_efficiency(output_dir)
            
            # Test 3: Prompt Compression Effectiveness
            logger.info("Test 3: Prompt compression effectiveness")
            await self._test_prompt_compression(output_dir)
            
            # Test 4: Safety Filter Compliance
            logger.info("Test 4: Safety filter compliance")
            await self._test_safety_filter_compliance(output_dir)
            
            # Test 5: End-to-End Performance
            logger.info("Test 5: End-to-end performance validation")
            await self._test_end_to_end_performance(output_dir)
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Generate overall assessment
            self._generate_overall_assessment(total_time)
            
            logger.info(f"Performance test suite completed in {total_time:.1f}ms")
            return self.results
            
        except Exception as e:
            logger.error(f"Performance test suite failed: {e}")
            self.results['overall_assessment'] = {
                'status': 'FAILED',
                'error': str(e),
                'total_time_ms': (time.perf_counter() - start_time) * 1000
            }
            return self.results
    
    async def _test_api_response_times(self, output_dir: Path) -> None:
        """Test API response time performance with different prompt sizes."""
        # Find a sample parent directory
        parent_dirs = list(output_dir.glob("parent_*"))
        if not parent_dirs:
            logger.warning("No parent directories found for API testing")
            return
        
        sample_parent = parent_dirs[0]
        parent_sku = sample_parent.name.replace("parent_", "")
        
        # Load sample data
        step2_file = sample_parent / "step2_compressed.json"
        if not step2_file.exists():
            logger.warning("No step2 file found for API testing")
            return
        
        with open(step2_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        # Initialize optimized client
        client = GeminiClient(config=self.ai_config)
        
        response_times = []
        success_count = 0
        
        # Test multiple requests to get average performance
        for i in range(5):
            try:
                # Create test prompt (optimized size)
                test_prompt = f"""Product mapping test {i+1}:
PARENT: {parent_sku}
DATA: {json.dumps(sample_data.get('parent_data', {}), separators=(',', ':'))}
OUTPUT: {{"parent_sku":"{parent_sku}","parent_data":{{"brand_name":"test"}},"metadata":{{"confidence":0.8}}}}"""
                
                start_time = time.perf_counter()
                response = await client.generate_mapping(
                    prompt=test_prompt,
                    operation_name=f"perf_test_{i}"
                )
                response_time = (time.perf_counter() - start_time) * 1000
                
                response_times.append(response_time)
                success_count += 1
                
                logger.info(f"API test {i+1}: {response_time:.1f}ms")
                
            except Exception as e:
                logger.error(f"API test {i+1} failed: {e}")
        
        # Calculate performance metrics
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        max_response_time = max(response_times) if response_times else 0.0
        min_response_time = min(response_times) if response_times else 0.0
        success_rate = success_count / 5
        
        self.results['api_performance'] = {
            'average_response_time_ms': avg_response_time,
            'max_response_time_ms': max_response_time,
            'min_response_time_ms': min_response_time,
            'success_rate': success_rate,
            'meets_target': avg_response_time <= 5000,  # 5s target
            'performance_rating': 'EXCELLENT' if avg_response_time <= 3000 else 
                                 'GOOD' if avg_response_time <= 5000 else 'NEEDS_IMPROVEMENT'
        }
    
    async def _test_batch_processing_efficiency(self, output_dir: Path) -> None:
        """Test batch processing performance with multiple parents."""
        # Initialize mapping processor
        processor = MappingProcessor(
            config=self.processing_config,
            ai_config=self.ai_config
        )
        
        # Find parent directories for batch testing
        parent_dirs = list(output_dir.glob("parent_*"))[:6]  # Test with up to 6 parents
        parent_skus = [d.name.replace("parent_", "") for d in parent_dirs]
        
        if len(parent_skus) < 2:
            logger.warning("Insufficient parent directories for batch testing")
            return
        
        logger.info(f"Testing batch processing with {len(parent_skus)} parents")
        
        # Test batch processing performance
        start_time = time.perf_counter()
        
        try:
            batch_results = await processor.batch_processor.process_parents_batch(
                parent_skus, output_dir, processor.process_parent_directory
            )
            
            batch_time = (time.perf_counter() - start_time) * 1000
            successful_parents = sum(1 for r in batch_results if r.success)
            
            # Calculate batch efficiency metrics
            avg_time_per_parent = batch_time / len(parent_skus)
            throughput_per_minute = (len(parent_skus) / batch_time) * 60000
            
            self.results['batch_performance'] = {
                'total_parents': len(parent_skus),
                'successful_parents': successful_parents,
                'success_rate': successful_parents / len(parent_skus),
                'total_batch_time_ms': batch_time,
                'avg_time_per_parent_ms': avg_time_per_parent,
                'throughput_per_minute': throughput_per_minute,
                'batch_efficiency': 'OPTIMAL' if avg_time_per_parent <= 8000 else 'NEEDS_OPTIMIZATION'
            }
            
        except Exception as e:
            logger.error(f"Batch processing test failed: {e}")
            self.results['batch_performance'] = {'status': 'FAILED', 'error': str(e)}
    
    async def _test_prompt_compression(self, output_dir: Path) -> None:
        """Test prompt compression effectiveness."""
        from sku_analyzer.shared.gemini_client import PromptOptimizer
        
        # Find sample data for compression testing
        parent_dirs = list(output_dir.glob("parent_*"))
        if not parent_dirs:
            logger.warning("No data found for compression testing")
            return
        
        sample_parent = parent_dirs[0]
        step2_file = sample_parent / "step2_compressed.json"
        
        if not step2_file.exists():
            return
        
        with open(step2_file, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        optimizer = PromptOptimizer()
        
        # Test data compression
        original_size = len(json.dumps(sample_data))
        compressed_data = optimizer.compress_product_data(sample_data, max_fields=5)
        compressed_size = len(json.dumps(compressed_data))
        
        compression_ratio = compressed_size / original_size if original_size > 0 else 0
        size_reduction = original_size - compressed_size
        
        # Test template compression
        template_file = output_dir / "flat_file_analysis" / "step4_template.json"
        template_compression_ratio = 0
        
        if template_file.exists():
            with open(template_file, 'r', encoding='utf-8') as f:
                template_data = json.load(f)
            
            original_template_size = len(json.dumps(template_data))
            essential_fields = optimizer.extract_essential_template_fields(
                template_data.get('template_structure', {}), max_fields=8
            )
            compressed_template_size = len(json.dumps(essential_fields))
            template_compression_ratio = compressed_template_size / original_template_size if original_template_size > 0 else 0
        
        self.results['compression_performance'] = {
            'original_data_size_bytes': original_size,
            'compressed_data_size_bytes': compressed_size,
            'data_compression_ratio': compression_ratio,
            'size_reduction_bytes': size_reduction,
            'template_compression_ratio': template_compression_ratio,
            'compression_effectiveness': 'EXCELLENT' if compression_ratio <= 0.3 else
                                      'GOOD' if compression_ratio <= 0.5 else 'MODERATE'
        }
    
    async def _test_safety_filter_compliance(self, output_dir: Path) -> None:
        """Test safety filter compliance with various prompt types."""
        client = GeminiClient(config=self.ai_config)
        
        # Test prompts with varying complexity
        test_prompts = [
            # Simple safe prompt
            '{"parent_sku":"test1","parent_data":{"brand_name":"TestBrand"},"metadata":{"confidence":0.8}}',
            
            # Medium complexity prompt
            'Map product: MANUFACTURER_NAME="TestCorp" to brand_name. Output: {"parent_data":{"brand_name":"TestCorp"}}',
            
            # More complex prompt (but still safe)
            'German Amazon product mapping: Parent 4301, Brand: EIKO, Category: Clothing. Map to Amazon format.',
        ]
        
        safety_results = []
        blocked_count = 0
        
        for i, prompt in enumerate(test_prompts):
            try:
                response = await client.generate_mapping(
                    prompt=prompt,
                    operation_name=f"safety_test_{i}"
                )
                safety_results.append({'prompt_index': i, 'status': 'SUCCESS'})
                
            except Exception as e:
                if 'safety filter' in str(e).lower():
                    blocked_count += 1
                    safety_results.append({'prompt_index': i, 'status': 'BLOCKED', 'reason': str(e)})
                else:
                    safety_results.append({'prompt_index': i, 'status': 'ERROR', 'reason': str(e)})
        
        safety_block_rate = blocked_count / len(test_prompts)
        
        self.results['safety_compliance'] = {
            'total_prompts_tested': len(test_prompts),
            'blocked_prompts': blocked_count,
            'safety_block_rate': safety_block_rate,
            'meets_target': safety_block_rate <= 0.05,  # 5% target
            'compliance_rating': 'EXCELLENT' if safety_block_rate == 0 else
                               'GOOD' if safety_block_rate <= 0.05 else 'NEEDS_IMPROVEMENT',
            'test_results': safety_results
        }
    
    async def _test_end_to_end_performance(self, output_dir: Path) -> None:
        """Test complete end-to-end performance with a single parent."""
        processor = MappingProcessor(
            config=self.processing_config,
            ai_config=self.ai_config
        )
        
        # Find a sample parent for end-to-end testing
        parent_dirs = list(output_dir.glob("parent_*"))
        if not parent_dirs:
            logger.warning("No parent directories found for e2e testing")
            return
        
        sample_parent = parent_dirs[0]
        parent_sku = sample_parent.name.replace("parent_", "")
        
        # Test complete processing pipeline
        start_time = time.perf_counter()
        
        try:
            result = await processor.process_parent_directory(
                parent_sku,
                output_dir / "flat_file_analysis" / "step4_template.json",
                sample_parent / "step2_compressed.json",
                sample_parent
            )
            
            e2e_time = (time.perf_counter() - start_time) * 1000
            
            self.results['end_to_end_performance'] = {
                'parent_sku': parent_sku,
                'success': result.success,
                'processing_time_ms': e2e_time,
                'confidence': result.confidence,
                'mapped_fields': result.mapped_fields_count,
                'performance_rating': 'EXCELLENT' if e2e_time <= 8000 else
                                    'GOOD' if e2e_time <= 15000 else 'NEEDS_IMPROVEMENT',
                'error': result.error if not result.success else None
            }
            
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            self.results['end_to_end_performance'] = {
                'status': 'FAILED',
                'error': str(e),
                'processing_time_ms': (time.perf_counter() - start_time) * 1000
            }
    
    def _generate_overall_assessment(self, total_test_time: float) -> None:
        """Generate overall performance assessment."""
        api_perf = self.results.get('api_performance', {})
        batch_perf = self.results.get('batch_performance', {})
        compression_perf = self.results.get('compression_performance', {})
        safety_perf = self.results.get('safety_compliance', {})
        e2e_perf = self.results.get('end_to_end_performance', {})
        
        # Calculate overall score
        scores = []
        
        # API performance score (0-100)
        if api_perf.get('meets_target'):
            scores.append(90 if api_perf['average_response_time_ms'] <= 3000 else 75)
        else:
            scores.append(40)
        
        # Safety compliance score (0-100)
        if safety_perf.get('meets_target'):
            scores.append(95 if safety_perf['safety_block_rate'] == 0 else 80)
        else:
            scores.append(50)
        
        # Batch efficiency score (0-100)
        if batch_perf.get('success_rate', 0) >= 0.9:
            scores.append(85)
        elif batch_perf.get('success_rate', 0) >= 0.8:
            scores.append(70)
        else:
            scores.append(50)
        
        # Compression effectiveness score (0-100)
        compression_ratio = compression_perf.get('data_compression_ratio', 1.0)
        if compression_ratio <= 0.3:
            scores.append(90)
        elif compression_ratio <= 0.5:
            scores.append(75)
        else:
            scores.append(60)
        
        overall_score = sum(scores) / len(scores) if scores else 0
        
        # Performance targets met
        targets_met = {
            'api_response_time': api_perf.get('meets_target', False),
            'safety_compliance': safety_perf.get('meets_target', False),
            'batch_efficiency': batch_perf.get('success_rate', 0) >= 0.9,
            'compression_effective': compression_perf.get('data_compression_ratio', 1.0) <= 0.5
        }
        
        all_targets_met = all(targets_met.values())
        
        self.results['overall_assessment'] = {
            'overall_score': overall_score,
            'performance_grade': 'A' if overall_score >= 85 else 
                               'B' if overall_score >= 75 else
                               'C' if overall_score >= 65 else 'D',
            'all_targets_met': all_targets_met,
            'targets_met': targets_met,
            'total_test_time_ms': total_test_time,
            'optimization_status': 'OPTIMAL' if all_targets_met and overall_score >= 85 else
                                 'GOOD' if overall_score >= 75 else 'NEEDS_IMPROVEMENT',
            'key_improvements': self._identify_key_improvements()
        }
    
    def _identify_key_improvements(self) -> list:
        """Identify key areas for improvement based on test results."""
        improvements = []
        
        api_perf = self.results.get('api_performance', {})
        safety_perf = self.results.get('safety_compliance', {})
        batch_perf = self.results.get('batch_performance', {})
        compression_perf = self.results.get('compression_performance', {})
        
        if not api_perf.get('meets_target'):
            improvements.append('Reduce API response times with further prompt optimization')
        
        if not safety_perf.get('meets_target'):
            improvements.append('Improve safety filter compliance with content sanitization')
        
        if batch_perf.get('success_rate', 0) < 0.9:
            improvements.append('Enhance batch processing reliability')
        
        if compression_perf.get('data_compression_ratio', 1.0) > 0.5:
            improvements.append('Implement more aggressive prompt compression')
        
        return improvements or ['Performance is within optimal parameters']


async def main():
    """Run performance tests on available output directory."""
    # Find the most recent output directory
    output_dirs = list(Path("/Users/jaminmahmood/Desktop/Flat Magic v6/production_output").glob("*"))
    output_dirs = [d for d in output_dirs if d.is_dir() and d.name.isdigit()]
    
    if not output_dirs:
        logger.error("No production output directories found")
        return
    
    # Use the most recent output directory
    latest_output = max(output_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"Using output directory: {latest_output}")
    
    # Run performance tests
    test_suite = PerformanceTestSuite()
    results = await test_suite.run_comprehensive_performance_tests(latest_output)
    
    # Save results
    results_file = Path("/Users/jaminmahmood/Desktop/Flat Magic v6/performance_test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    overall = results.get('overall_assessment', {})
    logger.info(f"\n{'='*60}")
    logger.info("PERFORMANCE TEST RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Score: {overall.get('overall_score', 0):.1f}/100")
    logger.info(f"Performance Grade: {overall.get('performance_grade', 'N/A')}")
    logger.info(f"Optimization Status: {overall.get('optimization_status', 'Unknown')}")
    logger.info(f"All Targets Met: {overall.get('all_targets_met', False)}")
    
    if overall.get('key_improvements'):
        logger.info(f"\nKey Improvements:")
        for improvement in overall['key_improvements']:
            logger.info(f"  - {improvement}")
    
    logger.info(f"\nDetailed results saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())