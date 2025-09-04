"""Optimized Gemini client with performance enhancements and safety compliance.

This module provides a highly optimized Gemini API client focused on:
- Minimal prompt payloads to avoid safety filters
- Efficient request batching and rate limiting
- Comprehensive performance monitoring
- Fast response times with reduced retry overhead
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from pydantic import BaseModel

from .performance import PerformanceMonitor


class GeminiResponse(BaseModel):
    """Structured response from Gemini API."""
    
    content: str = ""
    usage_metadata: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    safety_ratings: Optional[List[Dict[str, Any]]] = None


class AIProcessingConfig(BaseModel):
    """Optimized configuration for AI processing operations."""
    
    model_name: str = "gemini-2.0-flash-exp"
    temperature: float = 0.0
    max_tokens: int = 2048  # Reduced from 4096 for faster responses
    timeout_seconds: int = 15  # Reduced from 30 for faster failure detection
    max_concurrent: int = 3  # Increased from 1 for better throughput
    max_retries: int = 1  # Reduced from 2 for faster failure handling
    batch_size: int = 3  # Increased from 1 for batch processing
    confidence_threshold: float = 0.5
    api_key: str = ""
    
    # New optimization parameters
    max_prompt_size: int = 8000  # Character limit for prompts
    max_variants_per_request: int = 5  # Limit variants to reduce payload
    enable_prompt_compression: bool = True
    safety_filter_retry_delay: float = 0.5  # Reduced retry delay


class SafetyFilterException(Exception):
    """Exception raised when Gemini safety filters block content generation."""
    
    def __init__(self, message: str, finish_reason: str = None, safety_ratings: List[Dict[str, Any]] = None, prompt_size: int = 0):
        super().__init__(message)
        self.finish_reason = finish_reason
        self.safety_ratings = safety_ratings or []
        self.prompt_size = prompt_size
        self.blocked_categories = self._extract_blocked_categories()
    
    def _extract_blocked_categories(self) -> List[str]:
        """Extract categories that were blocked by safety filters."""
        if not self.safety_ratings:
            return []
        
        blocked = []
        for rating in self.safety_ratings:
            if rating.get("probability") in ["HIGH", "MEDIUM"]:
                blocked.append(rating.get("category", "UNKNOWN"))
        return blocked


class PromptOptimizer:
    """Optimizes prompts for reduced size and safety compliance."""
    
    @staticmethod
    def compress_product_data(product_data: Dict[str, Any], max_fields: int = 10, ultra_safe_mode: bool = False) -> Dict[str, Any]:
        """Compress product data to essential fields only with type safety."""
        if not isinstance(product_data, dict):
            logging.getLogger(__name__).warning(
                f"Expected dict for product_data, got {type(product_data).__name__}: {product_data}"
            )
            return {'parent_data': {}, 'data_rows': []}
        
        parent_data = product_data.get('parent_data', {})
        data_rows = product_data.get('data_rows', [])
        
        # Type safety checks
        if not isinstance(parent_data, dict):
            logging.getLogger(__name__).warning(
                f"Expected dict for parent_data, got {type(parent_data).__name__}: {parent_data}"
            )
            parent_data = {}
        
        if not isinstance(data_rows, list):
            logging.getLogger(__name__).warning(
                f"Expected list for data_rows, got {type(data_rows).__name__}: {data_rows}"
            )
            data_rows = []
        
        # Essential fields that are safe and commonly mapped
        if ultra_safe_mode:
            # Ultra-conservative field list for maximum safety compliance
            safe_essential_fields = [
                'MANUFACTURER_PID', 'FVALUE_3_1', 'FVALUE_3_2', 'SUPPLIER_PID'
            ]
        else:
            safe_essential_fields = [
                'MANUFACTURER_NAME', 'MANUFACTURER_PID', 'GROUP_STRING', 'WEIGHT',
                'FVALUE_3_1', 'FVALUE_3_2', 'FVALUE_3_3', 'SUPPLIER_PID'
            ]
        
        # Extract only safe parent fields
        compressed_parent = {}
        for field in safe_essential_fields:
            if field in parent_data:
                value = str(parent_data[field])
                # Remove potentially problematic content
                if len(value) > 100 or any(trigger in value.lower() for trigger in 
                    ['http', 'www', 'description', 'long', 'detail']):
                    continue
                compressed_parent[field] = value
        
        # Compress variants to essential fields only with type safety
        compressed_variants = []
        for i, variant in enumerate(data_rows[:max_fields]):  # Limit variant count
            if not isinstance(variant, dict):
                logging.getLogger(__name__).warning(
                    f"Expected dict for variant {i}, got {type(variant).__name__}: {variant}"
                )
                continue
            
            compressed_variant = {}
            for field in safe_essential_fields:
                if field in variant:
                    value = str(variant[field])
                    if len(value) <= 50:  # Keep only short values
                        compressed_variant[field] = value
            
            if compressed_variant:
                compressed_variants.append(compressed_variant)
        
        return {
            'parent_data': compressed_parent,
            'data_rows': compressed_variants
        }
    
    @staticmethod
    def extract_essential_template_fields(template_structure: Dict[str, Any], max_fields: int = 8) -> Dict[str, Any]:
        """Extract only the most essential template fields with type safety."""
        if not isinstance(template_structure, dict):
            logging.getLogger(__name__).warning(
                f"Expected dict for template_structure, got {type(template_structure).__name__}: {template_structure}"
            )
            return {}
        
        essential_fields = {}
        
        # Parent fields - most critical ones with type safety
        parent_product = template_structure.get('parent_product', {})
        if not isinstance(parent_product, dict):
            parent_product = {}
        
        parent_fields = parent_product.get('fields', {})
        if not isinstance(parent_fields, dict):
            parent_fields = {}
        
        priority_parent_fields = ['brand_name', 'feed_product_type', 'item_name', 'manufacturer']
        
        for field in priority_parent_fields:
            if field in parent_fields:
                field_info = parent_fields[field]
                if isinstance(field_info, dict):
                    validation_rules = field_info.get('validation_rules', {})
                    if not isinstance(validation_rules, dict):
                        validation_rules = {}
                    
                    # Simplify field structure to reduce payload
                    essential_fields[field] = {
                        'data_type': field_info.get('data_type', 'string'),
                        'required': validation_rules.get('required', True)
                    }
        
        # Variant fields - most critical ones with type safety
        child_variants = template_structure.get('child_variants', {})
        if not isinstance(child_variants, dict):
            child_variants = {}
        
        child_fields = child_variants.get('fields', {})
        if not isinstance(child_fields, dict):
            child_fields = {}
        
        priority_variant_fields = ['item_sku', 'color_name', 'size_name', 'external_product_id']
        
        for field in priority_variant_fields:
            if field in child_fields:
                field_info = child_fields[field]
                if isinstance(field_info, dict):
                    validation_rules = field_info.get('validation_rules', {})
                    if not isinstance(validation_rules, dict):
                        validation_rules = {}
                    
                    essential_fields[field] = {
                        'data_type': field_info.get('data_type', 'string'),
                        'required': validation_rules.get('required', True)
                    }
        
        return essential_fields


class GeminiClient:
    """Optimized async Gemini client with performance focus.
    
    Key Optimizations:
    - Compressed prompts with essential data only
    - Enhanced rate limiting and concurrency
    - Fast failure detection and minimal retries
    - Safety filter avoidance through prompt optimization
    - Efficient batch processing support
    """
    
    def __init__(
        self, 
        config: Optional[AIProcessingConfig] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """Initialize optimized Gemini client.
        
        Args:
            config: AI processing configuration
            performance_monitor: Optional performance monitor
        """
        self.config = config or AIProcessingConfig()
        self.performance_monitor = performance_monitor
        self.logger = logging.getLogger(__name__)
        self.prompt_optimizer = PromptOptimizer()
        
        # Validate API key
        self._api_key = os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")
        
        # Configure the API
        genai.configure(api_key=self._api_key)
        
        # Initialize model with optimized settings
        generation_config = genai.types.GenerationConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            response_mime_type="application/json",
        )
        
        # Optimized safety settings - minimal blocking for product data
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Enhanced rate limiting with burst capability
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._last_request_time = 0.0
        self._min_request_interval = 0.05  # Reduced to 50ms for faster throughput
        
        # Performance tracking
        self._reset_performance_counters()
    
    def _reset_performance_counters(self) -> None:
        """Reset performance tracking counters."""
        self.request_count = 0
        self.total_response_time = 0.0
        self.successful_requests = 0
        self.failed_requests = 0
        self.safety_blocked_requests = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
        self.prompt_compression_ratio = 0.0
        self.average_prompt_size = 0.0
    
    async def generate_mapping(
        self,
        prompt: str,
        timeout_override: Optional[int] = None,
        operation_name: str = "gemini_mapping",
        enable_ultra_safe_fallback: bool = True
    ) -> GeminiResponse:
        """Generate AI mapping with optimized performance.
        
        Args:
            prompt: Formatted prompt for mapping
            timeout_override: Optional timeout override
            operation_name: Name for performance tracking
            
        Returns:
            Structured Gemini response
            
        Raises:
            SafetyFilterException: If content blocked by safety filters
            TimeoutError: If request exceeds timeout
            ValueError: If response is invalid
        """
        # Store original prompt for error reporting
        self._last_prompt = prompt
        
        # Optimize prompt before sending
        optimized_prompt = self._optimize_prompt(prompt)
        
        async with self._semaphore:
            # Enhanced rate limiting
            await self._enforce_rate_limit()
            
            # Performance monitoring context
            if self.performance_monitor:
                with self.performance_monitor.measure_performance(operation_name):
                    return await self._execute_optimized_request(optimized_prompt, timeout_override)
            else:
                return await self._execute_optimized_request(optimized_prompt, timeout_override)
    
    def _optimize_prompt(self, prompt: str) -> str:
        """Optimize prompt for size and safety compliance."""
        original_size = len(prompt)
        
        # If prompt is within size limit, return as-is
        if original_size <= self.config.max_prompt_size:
            return prompt
        
        # Apply aggressive compression for oversized prompts
        lines = prompt.split('\n')
        optimized_lines = []
        current_size = 0
        
        for line in lines:
            # Skip overly long lines that might contain descriptions
            if len(line) > 200:
                continue
            
            # Skip lines with potential safety triggers
            if any(trigger in line.lower() for trigger in [
                'description_long', 'detail', 'feature', 'benefit', 'marketing'
            ]):
                continue
            
            if current_size + len(line) > self.config.max_prompt_size:
                break
                
            optimized_lines.append(line)
            current_size += len(line)
        
        optimized_prompt = '\n'.join(optimized_lines)
        compression_ratio = len(optimized_prompt) / original_size
        self.prompt_compression_ratio = compression_ratio
        
        self.logger.debug(f"Prompt compressed from {original_size} to {len(optimized_prompt)} chars "
                         f"(ratio: {compression_ratio:.2f})")
        
        return optimized_prompt
    
    def _create_ultra_safe_prompt(self, original_prompt: str) -> str:
        """Create ultra-safe minimal prompt for safety filter compliance.
        
        Args:
            original_prompt: Original prompt that triggered safety filters
            
        Returns:
            Ultra-minimal safe prompt (target: <500 chars)
        """
        # Extract just the essential mapping request
        lines = original_prompt.split('\n')
        
        # Find parent SKU if possible
        parent_sku = "unknown"
        for line in lines:
            if "PARENT:" in line:
                parent_sku = line.split(":")[-1].strip()
                break
        
        # Ultra-minimal mapping request
        return f"""Map product fields to Amazon format.
Parent: {parent_sku}
Output JSON: {{"parent_sku":"{parent_sku}","parent_data":{{"brand_name":"value"}},"variance_data":{{"variant_1":{{"item_sku":"value"}}}},"metadata":{{"confidence":0.6,"ultra_safe_fallback":true}}}}"""
    
    async def _execute_optimized_request(
        self, 
        prompt: str, 
        timeout_override: Optional[int]
    ) -> GeminiResponse:
        """Execute request with optimized performance monitoring."""
        start_time = time.perf_counter()
        self.average_prompt_size = (self.average_prompt_size * self.request_count + len(prompt)) / (self.request_count + 1)
        
        try:
            # Execute with reduced timeout for faster failure detection
            timeout = timeout_override or self.config.timeout_seconds
            response = await asyncio.wait_for(
                self._make_optimized_request(prompt),
                timeout=timeout
            )
            
            # Parse response with enhanced safety handling
            parsed_response = self._parse_response_with_safety_handling(response)
            
            # Record performance metrics
            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000  # Convert to ms
            
            self.request_count += 1
            self.total_response_time += response_time
            self.successful_requests += 1
            
            # Record token usage if available
            if parsed_response.usage_metadata:
                prompt_tokens = parsed_response.usage_metadata.get('prompt_token_count', 0)
                response_tokens = parsed_response.usage_metadata.get('candidates_token_count', 0)
                self.total_prompt_tokens += prompt_tokens
                self.total_response_tokens += response_tokens
            
            # Log performance for optimization tracking
            self.logger.debug(f"API response in {response_time:.1f}ms, "
                            f"prompt: {len(prompt)} chars, "
                            f"tokens: {parsed_response.usage_metadata.get('total_token_count', 0) if parsed_response.usage_metadata else 0}")
            
            return parsed_response
            
        except SafetyFilterException as safety_error:
            self.failed_requests += 1
            self.safety_blocked_requests += 1
            
            # Log detailed safety filter information
            self.logger.warning(
                f"Safety filter blocked request (size: {len(prompt)} chars, "
                f"categories: {safety_error.blocked_categories})"
            )
            
            # Try ultra-safe fallback if enabled and not already tried
            if (enable_ultra_safe_fallback and 
                not getattr(self, '_ultra_safe_attempted', False) and
                len(prompt) > 1000):  # Only for large prompts
                
                self.logger.info("Attempting ultra-safe fallback prompt")
                self._ultra_safe_attempted = True
                
                try:
                    # Generate ultra-minimal prompt
                    ultra_safe_prompt = self._create_ultra_safe_prompt(prompt)
                    fallback_response = await self._execute_optimized_request(
                        ultra_safe_prompt, timeout_override
                    )
                    
                    # Mark as fallback response
                    fallback_response.metadata = fallback_response.metadata or {}
                    fallback_response.metadata['ultra_safe_fallback'] = True
                    fallback_response.metadata['original_safety_error'] = str(safety_error)
                    
                    self.successful_requests += 1  # Count as success
                    return fallback_response
                    
                except Exception as fallback_error:
                    self.logger.error(f"Ultra-safe fallback also failed: {fallback_error}")
            
            # Reset flag for next request
            self._ultra_safe_attempted = False
            
            # Add small delay before giving up
            await asyncio.sleep(self.config.safety_filter_retry_delay)
            raise safety_error
        except asyncio.TimeoutError:
            self.failed_requests += 1
            self.logger.warning(f"Gemini request timed out after {timeout}s (prompt size: {len(prompt)} chars)")
            raise TimeoutError(f"Gemini request timed out after {timeout}s")
        except Exception as e:
            self.failed_requests += 1
            raise ValueError(f"Gemini request failed: {e}")
    
    async def _make_optimized_request(self, prompt: str) -> GenerateContentResponse:
        """Make optimized async request to Gemini API."""
        return await asyncio.to_thread(
            self.model.generate_content,
            prompt
        )
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce optimized rate limiting for better throughput."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            wait_time = self._min_request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self._last_request_time = time.time()
    
    def _parse_response_with_safety_handling(self, response: GenerateContentResponse) -> GeminiResponse:
        """Parse Gemini response with enhanced safety filter handling."""
        # Extract finish reason first
        finish_reason = None
        if response.candidates and len(response.candidates) > 0:
            finish_reason = str(response.candidates[0].finish_reason)
        
        # Extract safety ratings
        safety_ratings = None
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if hasattr(candidate, 'safety_ratings') and candidate.safety_ratings:
                safety_ratings = [
                    {
                        "category": str(rating.category),
                        "probability": str(rating.probability)
                    }
                    for rating in candidate.safety_ratings
                ]
        
        # Check for safety filter blocking
        if finish_reason == "2" or finish_reason == "SAFETY":
            # Create detailed error message
            blocked_categories = []
            if safety_ratings:
                for rating in safety_ratings:
                    if rating.get("probability") in ["HIGH", "MEDIUM"]:
                        blocked_categories.append(rating.get("category", "UNKNOWN"))
            
            error_msg = f"Content blocked by Gemini safety filters"
            if blocked_categories:
                error_msg += f" (categories: {', '.join(blocked_categories)})"
            
            raise SafetyFilterException(
                message=error_msg,
                finish_reason=finish_reason,
                safety_ratings=safety_ratings,
                prompt_size=len(getattr(self, '_last_prompt', ''))
            )
        
        # Extract text content safely
        content = ""
        try:
            if hasattr(response, 'text') and response.text:
                content = response.text
            elif response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        # Extract text from parts
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)
                        content = ''.join(text_parts)
        except Exception as e:
            self.logger.warning(f"Error extracting response text: {e}")
            content = ""
        
        # Extract usage metadata if available
        usage_metadata = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_metadata = {
                "prompt_token_count": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "candidates_token_count": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_token_count": getattr(response.usage_metadata, 'total_token_count', 0)
            }
        
        return GeminiResponse(
            content=content,
            usage_metadata=usage_metadata,
            finish_reason=finish_reason,
            safety_ratings=safety_ratings
        )
    
    async def validate_json_response(self, response: GeminiResponse) -> Dict[str, Any]:
        """Validate and parse JSON response with enhanced error handling."""
        if not response.content.strip():
            raise ValueError("Empty response content from Gemini")
        
        try:
            # Handle potential markdown code blocks
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            
            # Additional cleanup for common formatting issues
            content = content.strip()
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            # Enhanced error reporting for debugging
            self.logger.error(f"JSON decode error: {e}")
            self.logger.error(f"Response content preview: {response.content[:200]}...")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
    
    async def generate_batch_mappings(
        self,
        prompts: List[str],
        operation_name: str = "batch_mapping"
    ) -> List[GeminiResponse]:
        """Generate multiple mappings with optimized batch processing.
        
        Args:
            prompts: List of prompts to process
            operation_name: Name for performance tracking
            
        Returns:
            List of Gemini responses
        """
        # Process in parallel with concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def process_single_prompt(prompt: str, index: int) -> GeminiResponse:
            async with semaphore:
                return await self.generate_mapping(
                    prompt, 
                    operation_name=f"{operation_name}_{index}"
                )
        
        # Execute all prompts in parallel
        tasks = [
            process_single_prompt(prompt, i) 
            for i, prompt in enumerate(prompts)
        ]
        
        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_time = (time.perf_counter() - start_time) * 1000
        
        self.logger.info(f"Batch processing completed: {len(prompts)} requests in {batch_time:.1f}ms")
        
        # Convert exceptions to failed responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(GeminiResponse(
                    content='{"error": "batch_processing_failed"}',
                    finish_reason="ERROR"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary with optimization metrics."""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        success_rate = (
            self.successful_requests / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        safety_block_rate = (
            self.safety_blocked_requests / self.request_count
            if self.request_count > 0 else 0.0
        )
        
        return {
            # Basic metrics
            'total_requests': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'safety_blocked_requests': self.safety_blocked_requests,
            'success_rate': success_rate,
            'safety_block_rate': safety_block_rate,
            
            # Performance metrics
            'average_response_time_ms': avg_response_time,
            'total_response_time_ms': self.total_response_time,
            'average_prompt_size_chars': self.average_prompt_size,
            'prompt_compression_ratio': self.prompt_compression_ratio,
            
            # Token usage
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_response_tokens': self.total_response_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_response_tokens,
            'avg_tokens_per_request': (
                (self.total_prompt_tokens + self.total_response_tokens) / self.request_count
                if self.request_count > 0 else 0
            ),
            
            # Performance targets
            'meets_response_time_target': avg_response_time <= 5000,  # <5s target
            'meets_safety_target': safety_block_rate <= 0.05,  # <5% safety blocks
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._reset_performance_counters()