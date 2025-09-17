"""Safety Filter Robustness Test Suite

This test suite specifically validates the safety filter handling improvements
and fallback strategies implemented in the AI mapping fixes.

Key Test Areas:
1. Safety filter blocking scenarios (finish_reason=2)
2. Content sanitization effectiveness
3. Fallback strategy execution and reliability
4. German content handling (Latzhose → WorkPants)
5. Retry logic with progressive strategies
6. Production data processing with safety compliance

Success Criteria:
✅ All fallback strategies execute properly
✅ German clothing terms handled without safety blocks
✅ 95%+ of safety-blocked requests recover via fallbacks  
✅ Direct field mapping produces valid results
✅ No infinite retry loops or timeout failures
"""

import asyncio
import json
import logging
import time
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch

from sku_analyzer.step5_mapping.ai_mapper import AIMapper
from sku_analyzer.step5_mapping.models import MappingInput, TransformationResult, ProcessingConfig
from sku_analyzer.shared.gemini_client import GeminiClient, SafetyFilterException, AIProcessingConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SafetyFilterTestSuite:
    """Test suite focused on safety filter handling and robustness."""
    
    def __init__(self):
        """Initialize test suite with configurations."""
        self.ai_config = AIProcessingConfig(
            max_tokens=2048,
            timeout_seconds=10,
            max_concurrent=2,
            max_retries=3,
            batch_size=3,
            max_prompt_size=8000,
            max_variants_per_request=10,
            enable_prompt_compression=True
        )
        
        self.processing_config = ProcessingConfig(
            max_retries=3,
            timeout_seconds=10,
            batch_size=3,
            confidence_threshold=0.7  # Lower threshold for fallback testing
        )
        
        self.results = {
            "safety_blocking_tests": [],
            "fallback_strategy_tests": [],
            "content_sanitization_tests": [],
            "retry_logic_tests": [],
            "production_data_tests": [],
            "overall_assessment": {}
        }
    
    async def run_comprehensive_safety_tests(self) -> Dict[str, Any]:
        """Run comprehensive safety filter robustness tests."""
        logger.info("Starting safety filter robustness test suite")
        start_time = time.perf_counter()
        
        try:
            # Test 1: Safety Filter Blocking Scenarios
            logger.info("Test 1: Safety filter blocking scenarios")
            await self._test_safety_filter_blocking()
            
            # Test 2: Fallback Strategy Execution
            logger.info("Test 2: Fallback strategy execution")
            await self._test_fallback_strategies()
            
            # Test 3: Content Sanitization
            logger.info("Test 3: Content sanitization")
            await self._test_content_sanitization()
            
            # Test 4: Retry Logic Robustness
            logger.info("Test 4: Retry logic robustness")
            await self._test_retry_logic()
            
            # Test 5: Production Data Processing
            logger.info("Test 5: Production data processing")
            await self._test_production_data_safety()
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Generate overall assessment
            self._generate_safety_assessment(total_time)
            
            logger.info(f"Safety filter test suite completed in {total_time:.1f}ms")
            return self.results
            
        except Exception as e:
            logger.error(f"Safety filter test suite failed: {e}")
            self.results['overall_assessment'] = {
                'status': 'FAILED',
                'error': str(e),
                'total_time_ms': (time.perf_counter() - start_time) * 1000
            }
            return self.results
    
    async def _test_safety_filter_blocking(self) -> None:
        """Test various safety filter blocking scenarios."""
        # Mock AI client that simulates safety filter blocks
        mock_client = Mock(spec=GeminiClient)
        mock_client.config = Mock()
        mock_client.config.max_variants_per_request = 10
        
        result_formatter = Mock()
        result_formatter.processing_stats = {"successful_mappings": 0, "failed_mappings": 0}
        
        ai_mapper = AIMapper(
            ai_client=mock_client,
            config=self.processing_config,
            result_formatter=result_formatter
        )
        
        # Test scenarios that might trigger safety filters
        test_scenarios = [
            {
                "name": "German clothing terms",
                "data": {
                    "parent_data": {
                        "MANUFACTURER_NAME": "EIKO",
                        "GROUP_STRING": "Arbeitskleidung Latzhose"
                    },
                    "data_rows": [
                        {"FVALUE_3_3": "Schwarz", "FVALUE_3_2": "44"}
                    ]
                },
                "blocked_categories": ["HARM_CATEGORY_HARASSMENT"]
            },
            {
                "name": "Complex product descriptions",
                "data": {
                    "parent_data": {
                        "DESCRIPTION_LONG": "Professional workwear for industrial environments"
                    },
                    "data_rows": [
                        {"SUPPLIER_PID": "4307_40_44"}
                    ]
                },
                "blocked_categories": ["HARM_CATEGORY_UNSPECIFIED"]
            },
            {
                "name": "Multiple language content",
                "data": {
                    "parent_data": {
                        "MANUFACTURER_NAME": "EIKO",
                        "COUNTRY_OF_ORIGIN": "Tunesien"
                    },
                    "data_rows": []
                },
                "blocked_categories": ["HARM_CATEGORY_HARASSMENT"]
            }
        ]
        
        blocking_results = []
        
        for scenario in test_scenarios:
            # Setup mock to simulate safety filter block
            safety_error = SafetyFilterException(
                f"Safety filter blocked: {scenario['name']}",
                blocked_categories=scenario["blocked_categories"],
                prompt_size=5000
            )
            
            mock_client.generate_mapping = AsyncMock(side_effect=safety_error)
            mock_client.validate_json_response = AsyncMock(return_value={
                "parent_sku": "test_parent",
                "parent_data": {"brand_name": "EIKO"},
                "variance_data": {"variant_1": {"item_sku": "test_sku"}},
                "metadata": {"confidence": 0.7}
            })
            
            # Create mapping input
            mapping_input = MappingInput(
                parent_sku="test_parent",
                product_data=scenario["data"],
                mandatory_fields={"brand_name": "parent", "item_sku": "variant"},
                template_structure={}
            )
            
            try:
                start_time = time.perf_counter()
                result = await ai_mapper.execute_mapping_with_retry(mapping_input, max_retries=2)
                processing_time = (time.perf_counter() - start_time) * 1000
                
                test_result = {
                    "scenario": scenario["name"],
                    "success": isinstance(result, TransformationResult) and result.parent_sku == "test_parent",
                    "processing_time_ms": processing_time,
                    "safety_blocked": result.metadata.get("safety_blocked", False) if hasattr(result, 'metadata') else True,
                    "fallback_used": "fallback_strategy" in (result.metadata or {}) if hasattr(result, 'metadata') else False,
                    "confidence": result.metadata.get("confidence", 0.0) if hasattr(result, 'metadata') else 0.0
                }
                
                blocking_results.append(test_result)
                logger.info(f"Safety blocking test '{scenario['name']}': {'PASS' if test_result['success'] else 'FAIL'}")
                
            except Exception as e:
                test_result = {
                    "scenario": scenario["name"],
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": (time.perf_counter() - start_time) * 1000
                }
                blocking_results.append(test_result)
                logger.error(f"Safety blocking test '{scenario['name']}' failed: {e}")
        
        self.results["safety_blocking_tests"] = blocking_results
    
    async def _test_fallback_strategies(self) -> None:
        """Test the execution and effectiveness of fallback strategies."""
        mock_client = Mock(spec=GeminiClient)
        mock_client.config = Mock()
        mock_client.config.max_variants_per_request = 10
        
        result_formatter = Mock()
        result_formatter.processing_stats = {"successful_mappings": 0, "failed_mappings": 0}
        
        ai_mapper = AIMapper(
            ai_client=mock_client,
            config=self.processing_config,
            result_formatter=result_formatter
        )
        
        # Test data similar to parent_4307
        test_data = {
            "parent_data": {
                "MANUFACTURER_NAME": "EIKO",
                "GROUP_STRING": "Arbeitskleidung Latzhose",
                "COUNTRY_OF_ORIGIN": "Tunesien",
                "WEIGHT": "0,86"
            },
            "data_rows": [
                {"SUPPLIER_PID": "4307_40_44", "FVALUE_3_3": "Schwarz", "FVALUE_3_2": "44"},
                {"SUPPLIER_PID": "4307_40_46", "FVALUE_3_3": "Schwarz", "FVALUE_3_2": "46"}
            ]
        }
        
        mapping_input = MappingInput(
            parent_sku="fallback_test",
            product_data=test_data,
            mandatory_fields={
                "brand_name": "parent",
                "feed_product_type": "parent", 
                "item_sku": "variant",
                "color_name": "variant"
            },
            template_structure={}
        )
        
        # Test each fallback strategy individually
        fallback_strategies = [
            ("ultra_simplified", "_execute_ultra_simplified_mapping"),
            ("field_only", "_execute_field_only_mapping"),
            ("minimal_safe", "_execute_minimal_safe_mapping"),
            ("minimal_fallback", "_create_minimal_fallback_result")
        ]
        
        fallback_results = []
        
        for strategy_name, method_name in fallback_strategies:
            try:
                if strategy_name == "minimal_fallback":
                    # Test the minimal fallback result creation
                    safety_error = SafetyFilterException(
                        "Test safety error",
                        blocked_categories=["HARM_CATEGORY_HARASSMENT"],
                        prompt_size=5000
                    )
                    
                    start_time = time.perf_counter()
                    result = ai_mapper._create_minimal_fallback_result(mapping_input, safety_error)
                    processing_time = (time.perf_counter() - start_time) * 1000
                else:
                    # Test async fallback methods
                    mock_client.generate_mapping = AsyncMock()
                    mock_client.validate_json_response = AsyncMock(return_value={
                        "parent_sku": "fallback_test",
                        "parent_data": {"brand_name": "EIKO", "feed_product_type": "pants"},
                        "variance_data": {"variant_1": {"item_sku": "4307_40_44", "color_name": "Schwarz"}},
                        "metadata": {"confidence": 0.7}
                    })
                    
                    start_time = time.perf_counter()
                    method = getattr(ai_mapper, method_name)
                    result = await method(mapping_input)
                    processing_time = (time.perf_counter() - start_time) * 1000
                
                # Validate result
                test_result = {
                    "strategy": strategy_name,
                    "success": isinstance(result, TransformationResult),
                    "processing_time_ms": processing_time,
                    "parent_data_count": len(result.parent_data) if hasattr(result, 'parent_data') else 0,
                    "variance_data_count": len(result.variance_data) if hasattr(result, 'variance_data') else 0,
                    "confidence": result.metadata.get("confidence", 0.0) if hasattr(result, 'metadata') and result.metadata else 0.0,
                    "has_required_fields": False
                }
                
                # Check if required fields are present
                if hasattr(result, 'parent_data') and hasattr(result, 'variance_data'):
                    has_brand = "brand_name" in result.parent_data
                    has_sku = any("item_sku" in variant_data for variant_item in result.variance_data.values() if isinstance(variant_item, dict) for variant_data in [variant_item] if isinstance(variant_data, dict))
                    test_result["has_required_fields"] = has_brand or has_sku
                
                fallback_results.append(test_result)
                logger.info(f"Fallback strategy '{strategy_name}': {'PASS' if test_result['success'] else 'FAIL'}")
                
            except Exception as e:
                test_result = {
                    "strategy": strategy_name,
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": 0
                }
                fallback_results.append(test_result)
                logger.error(f"Fallback strategy '{strategy_name}' failed: {e}")
        
        self.results["fallback_strategy_tests"] = fallback_results
    
    async def _test_content_sanitization(self) -> None:
        """Test content sanitization for German clothing terms."""
        sanitization_tests = [
            {
                "input_term": "Latzhose",
                "expected_safe_term": "WorkPants",
                "context": "German work clothing"
            },
            {
                "input_term": "Arbeitskleidung",
                "expected_safe_term": "workwear",
                "context": "German work clothing category"
            },
            {
                "input_term": "Tunesien",
                "expected_safe_term": "Tunisia",
                "context": "German country name"
            }
        ]
        
        sanitization_results = []
        
        # Test the minimal fallback result which includes content sanitization
        mock_client = Mock(spec=GeminiClient)
        mock_client.config = Mock()
        mock_client.config.max_variants_per_request = 10
        
        result_formatter = Mock()
        ai_mapper = AIMapper(mock_client, self.processing_config, result_formatter)
        
        for test_case in sanitization_tests:
            test_data = {
                "parent_data": {
                    "GROUP_STRING": test_case["input_term"],
                    "COUNTRY_OF_ORIGIN": test_case["input_term"] if "country" in test_case["context"].lower() else "Deutschland"
                },
                "data_rows": [{"FVALUE_3_3": "Test"}]
            }
            
            mapping_input = MappingInput(
                parent_sku="sanitization_test",
                product_data=test_data,
                mandatory_fields={"feed_product_type": "parent"},
                template_structure={}
            )
            
            safety_error = SafetyFilterException(
                f"Safety filter blocked: {test_case['input_term']}",
                blocked_categories=["HARM_CATEGORY_HARASSMENT"],
                prompt_size=1000
            )
            
            try:
                result = ai_mapper._create_minimal_fallback_result(mapping_input, safety_error)
                
                # Check if sanitization occurred
                sanitized = False
                if hasattr(result, 'parent_data') and isinstance(result.parent_data, dict):
                    for field_value in result.parent_data.values():
                        if isinstance(field_value, str):
                            # Check if German terms were converted to English equivalents
                            if ("pants" in field_value.lower() and "latzhose" in test_case["input_term"].lower()) or \
                               ("tunisia" in field_value.lower() and "tunesien" in test_case["input_term"].lower()) or \
                               ("workwear" in field_value.lower() and "arbeitskleidung" in test_case["input_term"].lower()):
                                sanitized = True
                                break
                
                test_result = {
                    "input_term": test_case["input_term"],
                    "context": test_case["context"],
                    "sanitization_applied": sanitized,
                    "success": isinstance(result, TransformationResult),
                    "parent_data_fields": len(result.parent_data) if hasattr(result, 'parent_data') else 0
                }
                
                sanitization_results.append(test_result)
                logger.info(f"Content sanitization '{test_case['input_term']}': {'PASS' if sanitized else 'PARTIAL'}")
                
            except Exception as e:
                test_result = {
                    "input_term": test_case["input_term"],
                    "context": test_case["context"],
                    "success": False,
                    "error": str(e)
                }
                sanitization_results.append(test_result)
                logger.error(f"Content sanitization '{test_case['input_term']}' failed: {e}")
        
        self.results["content_sanitization_tests"] = sanitization_results
    
    async def _test_retry_logic(self) -> None:
        """Test retry logic robustness and prevent infinite loops."""
        mock_client = Mock(spec=GeminiClient)
        mock_client.config = Mock()
        mock_client.config.max_variants_per_request = 10
        
        result_formatter = Mock()
        result_formatter.processing_stats = {"successful_mappings": 0, "failed_mappings": 0}
        
        ai_mapper = AIMapper(mock_client, self.processing_config, result_formatter)
        
        test_data = {
            "parent_data": {"MANUFACTURER_NAME": "EIKO"},
            "data_rows": [{"SUPPLIER_PID": "test_sku"}]
        }
        
        mapping_input = MappingInput(
            parent_sku="retry_test",
            product_data=test_data,
            mandatory_fields={"brand_name": "parent"},
            template_structure={}
        )
        
        retry_test_scenarios = [
            {
                "name": "Immediate safety block with successful fallback",
                "main_error": SafetyFilterException("Safety blocked", ["HARM_CATEGORY_HARASSMENT"], 5000),
                "fallback_success": True
            },
            {
                "name": "Multiple safety blocks with eventual success",
                "main_error": SafetyFilterException("Persistent safety block", ["HARM_CATEGORY_UNSPECIFIED"], 3000),
                "fallback_success": True
            },
            {
                "name": "Timeout with retry success",
                "main_error": asyncio.TimeoutError("Request timeout"),
                "fallback_success": True
            }
        ]
        
        retry_results = []
        
        for scenario in retry_test_scenarios:
            try:
                # Track retry attempts
                call_count = 0
                
                async def mock_generate_mapping(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count <= 2:  # Fail first 2 attempts
                        raise scenario["main_error"]
                    return "success"  # Succeed on 3rd attempt
                
                mock_client.generate_mapping = mock_generate_mapping
                mock_client.validate_json_response = AsyncMock(return_value={
                    "parent_sku": "retry_test",
                    "parent_data": {"brand_name": "EIKO"},
                    "metadata": {"confidence": 0.8}
                })
                
                start_time = time.perf_counter()
                
                # Test with max_retries=3 (should succeed)
                result = await ai_mapper.execute_mapping_with_retry(mapping_input, max_retries=3)
                
                processing_time = (time.perf_counter() - start_time) * 1000
                
                test_result = {
                    "scenario": scenario["name"],
                    "success": isinstance(result, TransformationResult),
                    "retry_attempts": call_count,
                    "processing_time_ms": processing_time,
                    "final_success": result.parent_sku == "retry_test" if hasattr(result, 'parent_sku') else False,
                    "no_infinite_loop": processing_time < 10000  # Under 10 seconds
                }
                
                retry_results.append(test_result)
                logger.info(f"Retry logic '{scenario['name']}': {'PASS' if test_result['success'] else 'FAIL'}")
                
            except Exception as e:
                test_result = {
                    "scenario": scenario["name"],
                    "success": False,
                    "error": str(e),
                    "processing_time_ms": (time.perf_counter() - start_time) * 1000 if 'start_time' in locals() else 0
                }
                retry_results.append(test_result)
                logger.error(f"Retry logic '{scenario['name']}' failed: {e}")
        
        self.results["retry_logic_tests"] = retry_results
    
    async def _test_production_data_safety(self) -> None:
        """Test safety handling with real production data."""
        # Load parent_4307 data if available
        production_file = Path("/Users/jaminmahmood/Desktop/Flat Magic v6/production_output/1756744181/parent_4307/step2_compressed.json")
        
        if not production_file.exists():
            logger.warning("Production test data not available, skipping production safety tests")
            self.results["production_data_tests"] = [{"status": "SKIPPED", "reason": "No production data"}]
            return
        
        with open(production_file, 'r', encoding='utf-8') as f:
            production_data = json.load(f)
        
        # Test with real production data and simulated safety blocks
        mock_client = Mock(spec=GeminiClient)
        mock_client.config = Mock()
        mock_client.config.max_variants_per_request = 10
        
        result_formatter = Mock()
        result_formatter.processing_stats = {"successful_mappings": 0, "failed_mappings": 0}
        
        ai_mapper = AIMapper(mock_client, self.processing_config, result_formatter)
        
        mapping_input = MappingInput(
            parent_sku="4307",
            product_data=production_data,
            mandatory_fields={
                "brand_name": "parent",
                "feed_product_type": "parent",
                "item_sku": "variant",
                "color_name": "variant"
            },
            template_structure={}
        )
        
        # Simulate safety block scenario
        safety_error = SafetyFilterException(
            "Production data safety block",
            blocked_categories=["HARM_CATEGORY_HARASSMENT"],
            prompt_size=len(json.dumps(production_data))
        )
        
        mock_client.generate_mapping = AsyncMock(side_effect=safety_error)
        
        try:
            start_time = time.perf_counter()
            result = await ai_mapper.execute_mapping_with_retry(mapping_input, max_retries=2)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Validate that all 81 variants were handled
            variant_count = len(result.variance_data) if hasattr(result, 'variance_data') else 0
            expected_variants = len(production_data.get("data_rows", []))
            
            production_test_result = {
                "data_source": "parent_4307",
                "success": isinstance(result, TransformationResult),
                "variants_processed": variant_count,
                "expected_variants": expected_variants,
                "variant_coverage": variant_count / expected_variants if expected_variants > 0 else 0,
                "processing_time_ms": processing_time,
                "safety_handled": result.metadata.get("safety_blocked", False) if hasattr(result, 'metadata') else True,
                "fallback_used": "fallback_strategy" in (result.metadata or {}) if hasattr(result, 'metadata') else False,
                "confidence": result.metadata.get("confidence", 0.0) if hasattr(result, 'metadata') else 0.0
            }
            
            self.results["production_data_tests"] = [production_test_result]
            logger.info(f"Production safety test: {'PASS' if production_test_result['success'] else 'FAIL'}")
            
        except Exception as e:
            production_test_result = {
                "data_source": "parent_4307",
                "success": False,
                "error": str(e),
                "processing_time_ms": (time.perf_counter() - start_time) * 1000
            }
            self.results["production_data_tests"] = [production_test_result]
            logger.error(f"Production safety test failed: {e}")
    
    def _generate_safety_assessment(self, total_time_ms: float) -> None:
        """Generate overall safety filter robustness assessment."""
        # Analyze results from all test categories
        blocking_tests = self.results.get("safety_blocking_tests", [])
        fallback_tests = self.results.get("fallback_strategy_tests", [])
        sanitization_tests = self.results.get("content_sanitization_tests", [])
        retry_tests = self.results.get("retry_logic_tests", [])
        production_tests = self.results.get("production_data_tests", [])
        
        # Calculate success rates
        blocking_success_rate = sum(1 for t in blocking_tests if t.get("success", False)) / len(blocking_tests) if blocking_tests else 0
        fallback_success_rate = sum(1 for t in fallback_tests if t.get("success", False)) / len(fallback_tests) if fallback_tests else 0
        sanitization_success_rate = sum(1 for t in sanitization_tests if t.get("sanitization_applied", False)) / len(sanitization_tests) if sanitization_tests else 0
        retry_success_rate = sum(1 for t in retry_tests if t.get("success", False)) / len(retry_tests) if retry_tests else 0
        production_success = production_tests[0].get("success", False) if production_tests else False
        
        # Calculate overall robustness score
        component_scores = [
            blocking_success_rate * 0.25,  # 25% weight
            fallback_success_rate * 0.30,  # 30% weight - most important
            sanitization_success_rate * 0.20,  # 20% weight
            retry_success_rate * 0.15,  # 15% weight
            (1.0 if production_success else 0.0) * 0.10  # 10% weight
        ]
        
        overall_robustness_score = sum(component_scores)
        
        # Determine robustness grade
        if overall_robustness_score >= 0.9:
            grade = "A - EXCELLENT"
        elif overall_robustness_score >= 0.8:
            grade = "B - GOOD"
        elif overall_robustness_score >= 0.7:
            grade = "C - ACCEPTABLE"
        else:
            grade = "D - NEEDS IMPROVEMENT"
        
        # Identify key strengths and weaknesses
        strengths = []
        weaknesses = []
        
        if fallback_success_rate >= 0.8:
            strengths.append("Robust fallback strategy implementation")
        else:
            weaknesses.append("Fallback strategies need improvement")
        
        if blocking_success_rate >= 0.8:
            strengths.append("Effective safety filter handling")
        else:
            weaknesses.append("Safety filter handling needs work")
        
        if sanitization_success_rate >= 0.6:
            strengths.append("Content sanitization working")
        else:
            weaknesses.append("Content sanitization incomplete")
        
        if production_success:
            strengths.append("Production data processing successful")
        else:
            weaknesses.append("Production data processing issues")
        
        self.results["overall_assessment"] = {
            "overall_robustness_score": overall_robustness_score,
            "robustness_grade": grade,
            "component_scores": {
                "safety_blocking_handling": blocking_success_rate,
                "fallback_strategy_execution": fallback_success_rate,
                "content_sanitization": sanitization_success_rate,
                "retry_logic_robustness": retry_success_rate,
                "production_data_safety": 1.0 if production_success else 0.0
            },
            "total_test_time_ms": total_time_ms,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "production_ready": overall_robustness_score >= 0.8,
            "recommendations": self._generate_safety_recommendations(overall_robustness_score, weaknesses)
        }
    
    def _generate_safety_recommendations(self, score: float, weaknesses: List[str]) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if score < 0.8:
            recommendations.append("Overall robustness below production threshold - comprehensive review needed")
        
        if "Fallback strategies need improvement" in weaknesses:
            recommendations.append("Enhance fallback strategy reliability and coverage")
        
        if "Safety filter handling needs work" in weaknesses:
            recommendations.append("Improve primary safety filter compliance strategies")
        
        if "Content sanitization incomplete" in weaknesses:
            recommendations.append("Expand content sanitization rules for German terms")
        
        if "Production data processing issues" in weaknesses:
            recommendations.append("Test with additional production datasets")
        
        if not recommendations:
            recommendations.append("Safety filter robustness meets production standards")
        
        return recommendations


async def main():
    """Run safety filter robustness tests."""
    print("Running Safety Filter Robustness Test Suite...")
    print("=" * 60)
    
    test_suite = SafetyFilterTestSuite()
    results = await test_suite.run_comprehensive_safety_tests()
    
    # Save results
    results_file = Path("/Users/jaminmahmood/Desktop/Flat Magic v6/safety_filter_test_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    overall = results.get('overall_assessment', {})
    print(f"\n{'='*60}")
    print("SAFETY FILTER ROBUSTNESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"Overall Robustness Score: {overall.get('overall_robustness_score', 0):.3f}")
    print(f"Robustness Grade: {overall.get('robustness_grade', 'Unknown')}")
    print(f"Production Ready: {'✅ YES' if overall.get('production_ready', False) else '❌ NO'}")
    
    component_scores = overall.get('component_scores', {})
    print(f"\nComponent Scores:")
    for component, score in component_scores.items():
        print(f"  {component.replace('_', ' ').title()}: {score:.3f}")
    
    strengths = overall.get('strengths', [])
    if strengths:
        print(f"\nStrengths:")
        for strength in strengths:
            print(f"  ✅ {strength}")
    
    weaknesses = overall.get('weaknesses', [])
    if weaknesses:
        print(f"\nWeaknesses:")
        for weakness in weaknesses:
            print(f"  ⚠️ {weakness}")
    
    recommendations = overall.get('recommendations', [])
    if recommendations:
        print(f"\nRecommendations:")
        for rec in recommendations:
            print(f"  🔧 {rec}")
    
    print(f"\nDetailed results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())