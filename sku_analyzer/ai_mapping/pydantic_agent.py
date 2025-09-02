"""Pydantic AI agent implementation for product mapping."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel

from .models import MappingInput, TransformationResult, AIProcessingConfig
from .prompts.templates import PromptTemplateManager


class ProductMappingDependencies:
    """Dependencies for product mapping agent."""
    
    def __init__(self, prompt_manager: PromptTemplateManager):
        """Initialize dependencies.
        
        Args:
            prompt_manager: Template manager for prompts
        """
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)


class ModernPydanticAgent:
    """Modern Pydantic AI agent for product mapping using latest patterns."""
    
    def __init__(self, config: AIProcessingConfig):
        """Initialize Pydantic AI agent.
        
        Args:
            config: AI processing configuration
        """
        self.config = config
        self.prompt_manager = PromptTemplateManager()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Google model for Pydantic AI
        self.model = GoogleModel(
            model_name=config.model_name,
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Create agent with structured output
        self.agent = Agent(
            model=self.model,
            deps_type=ProductMappingDependencies,
            result_type=TransformationResult,
            system_prompt=self._get_system_prompt()
        )
        
        # Performance tracking
        self._request_count = 0
        self._total_processing_time = 0.0
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for agent.
        
        Returns:
            System prompt string
        """
        return self.prompt_manager.get_system_prompt()
    
    async def map_product_data(
        self,
        mapping_input: MappingInput
    ) -> TransformationResult:
        """Map product data using Pydantic AI agent.
        
        Args:
            mapping_input: Input data for mapping
            
        Returns:
            Structured mapping result
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Create dependencies
            deps = ProductMappingDependencies(self.prompt_manager)
            
            # Generate mapping prompt
            prompt_context = {
                "parent_sku": mapping_input.parent_sku,
                "mandatory_fields": mapping_input.mandatory_fields,
                "product_data": mapping_input.product_data,
                "business_context": mapping_input.business_context
            }
            
            user_prompt = self.prompt_manager.render_mapping_prompt(prompt_context)
            
            # Execute agent with timeout
            result = await asyncio.wait_for(
                self.agent.run(user_prompt, deps=deps),
                timeout=self.config.timeout_seconds
            )
            
            # Track performance
            self._track_request_performance(start_time)
            
            return result.data
            
        except asyncio.TimeoutError:
            self.logger.error(f"Agent timeout for SKU {mapping_input.parent_sku}")
            return self._create_timeout_result(mapping_input.parent_sku)
        except Exception as e:
            self.logger.error(f"Agent error for SKU {mapping_input.parent_sku}: {e}")
            return self._create_error_result(mapping_input.parent_sku, str(e))
    
    async def batch_map_products(
        self,
        mapping_inputs: List[MappingInput]
    ) -> List[TransformationResult]:
        """Process multiple products in batches.
        
        Args:
            mapping_inputs: List of mapping inputs
            
        Returns:
            List of mapping results
        """
        # Process in batches with controlled concurrency
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(mapping_inputs), batch_size):
            batch = mapping_inputs[i:i + batch_size]
            
            # Create tasks for batch
            tasks = [self.map_product_data(input_data) for input_data in batch]
            
            # Execute batch with error handling
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results, handling exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch processing error: {result}")
                    # Create error result
                    error_result = self._create_error_result("unknown", str(result))
                    results.append(error_result)
                else:
                    results.append(result)
        
        return results
    
    def _track_request_performance(self, start_time: float) -> None:
        """Track request performance metrics.
        
        Args:
            start_time: Request start time
        """
        duration = asyncio.get_event_loop().time() - start_time
        self._request_count += 1
        self._total_processing_time += duration
        
        avg_time = self._total_processing_time / self._request_count
        self.logger.debug(
            f"Request #{self._request_count} completed in {duration:.2f}s "
            f"(avg: {avg_time:.2f}s)"
        )
    
    def _create_timeout_result(self, parent_sku: str) -> TransformationResult:
        """Create result for timeout scenarios.
        
        Args:
            parent_sku: SKU identifier
            
        Returns:
            Timeout transformation result
        """
        return TransformationResult(
            parent_sku=parent_sku,
            parent_data={},
            variance_data={},
            metadata={
                "total_mapped_fields": 0,
                "confidence": 0.0,
                "unmapped_mandatory": [],
                "processing_notes": "Request timed out - no transformation performed"
            }
        )
    
    def _create_error_result(self, parent_sku: str, error_msg: str) -> TransformationResult:
        """Create result for error scenarios.
        
        Args:
            parent_sku: SKU identifier
            error_msg: Error message
            
        Returns:
            Error transformation result
        """
        return TransformationResult(
            parent_sku=parent_sku,
            parent_data={},
            variance_data={},
            metadata={
                "total_mapped_fields": 0,
                "confidence": 0.0,
                "unmapped_mandatory": [],
                "processing_notes": f"Processing error: {error_msg}"
            }
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Performance statistics dictionary
        """
        avg_time = (
            self._total_processing_time / self._request_count 
            if self._request_count > 0 else 0.0
        )
        
        return {
            "total_requests": self._request_count,
            "total_processing_time": self._total_processing_time,
            "average_processing_time": avg_time,
            "requests_per_second": self._request_count / max(self._total_processing_time, 0.001)
        }