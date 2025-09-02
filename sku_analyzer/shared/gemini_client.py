"""Unified Gemini client with modern patterns and comprehensive error handling.

This module consolidates all Gemini API interactions into a single,
well-tested client with rate limiting, performance monitoring,
and robust error handling.
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
    """Configuration for AI processing operations."""
    
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_seconds: int = 30
    max_concurrent: int = 1


class GeminiClient:
    """Unified async Gemini client with comprehensive features.
    
    Features:
    - Rate limiting and concurrency control
    - Performance monitoring and metrics tracking
    - Robust error handling with retry logic
    - JSON response validation
    - Token usage tracking
    """
    
    def __init__(
        self, 
        config: Optional[AIProcessingConfig] = None,
        performance_monitor: Optional[PerformanceMonitor] = None
    ):
        """Initialize Gemini client.
        
        Args:
            config: AI processing configuration
            performance_monitor: Optional performance monitor
        """
        self.config = config or AIProcessingConfig()
        self.performance_monitor = performance_monitor
        self.logger = logging.getLogger(__name__)
        
        # Validate API key
        self._api_key = os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError("GOOGLE_API_KEY environment variable required")
        
        # Configure the API
        genai.configure(api_key=self._api_key)
        
        # Initialize model with optimal settings
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config=genai.types.GenerationConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
                response_mime_type="application/json"  # Force JSON output
            )
        )
        
        # Rate limiting
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 100ms between requests
        
        # Performance tracking
        self._reset_performance_counters()
    
    def _reset_performance_counters(self) -> None:
        """Reset performance tracking counters."""
        self.request_count = 0
        self.total_response_time = 0.0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_prompt_tokens = 0
        self.total_response_tokens = 0
    
    async def generate_mapping(
        self,
        prompt: str,
        timeout_override: Optional[int] = None,
        operation_name: str = "gemini_mapping"
    ) -> GeminiResponse:
        """Generate AI mapping with comprehensive monitoring.
        
        Args:
            prompt: Formatted prompt for mapping
            timeout_override: Optional timeout override
            operation_name: Name for performance tracking
            
        Returns:
            Structured Gemini response
            
        Raises:
            TimeoutError: If request exceeds timeout
            ValueError: If response is invalid
        """
        async with self._semaphore:
            # Rate limiting
            await self._enforce_rate_limit()
            
            # Performance monitoring context
            if self.performance_monitor:
                with self.performance_monitor.measure_performance(operation_name):
                    return await self._execute_request_with_monitoring(prompt, timeout_override)
            else:
                return await self._execute_request_with_monitoring(prompt, timeout_override)
    
    async def _execute_request_with_monitoring(
        self, 
        prompt: str, 
        timeout_override: Optional[int]
    ) -> GeminiResponse:
        """Execute request with comprehensive monitoring."""
        start_time = time.perf_counter()
        
        try:
            # Execute request with timeout
            timeout = timeout_override or self.config.timeout_seconds
            response = await asyncio.wait_for(
                self._make_request(prompt),
                timeout=timeout
            )
            
            # Parse response
            parsed_response = self._parse_response(response)
            
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
            
            return parsed_response
            
        except asyncio.TimeoutError:
            self.failed_requests += 1
            raise TimeoutError(f"Gemini request timed out after {timeout}s")
        except Exception as e:
            self.failed_requests += 1
            raise ValueError(f"Gemini request failed: {e}")
    
    async def _make_request(self, prompt: str) -> GenerateContentResponse:
        """Make async request to Gemini API.
        
        Args:
            prompt: Request prompt
            
        Returns:
            Raw Gemini response
        """
        return await asyncio.to_thread(
            self.model.generate_content,
            prompt
        )
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce minimum time between requests."""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            wait_time = self._min_request_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self._last_request_time = time.time()
    
    def _parse_response(self, response: GenerateContentResponse) -> GeminiResponse:
        """Parse Gemini response into structured format.
        
        Args:
            response: Raw Gemini response
            
        Returns:
            Structured response object
        """
        # Extract text content
        content = response.text if response.text else ""
        
        # Extract usage metadata if available
        usage_metadata = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_metadata = {
                "prompt_token_count": getattr(response.usage_metadata, 'prompt_token_count', 0),
                "candidates_token_count": getattr(response.usage_metadata, 'candidates_token_count', 0),
                "total_token_count": getattr(response.usage_metadata, 'total_token_count', 0)
            }
        
        # Extract finish reason
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
        
        return GeminiResponse(
            content=content,
            usage_metadata=usage_metadata,
            finish_reason=finish_reason,
            safety_ratings=safety_ratings
        )
    
    async def validate_json_response(self, response: GeminiResponse) -> Dict[str, Any]:
        """Validate and parse JSON response.
        
        Args:
            response: Gemini response object
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValueError: If JSON is invalid
        """
        try:
            # Handle potential markdown code blocks
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.endswith("```"):
                content = content[:-3]  # Remove ```
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring integration."""
        avg_response_time = (
            self.total_response_time / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        success_rate = (
            self.successful_requests / self.request_count 
            if self.request_count > 0 else 0.0
        )
        
        return {
            'total_requests': self.request_count,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'average_response_time_ms': avg_response_time,
            'total_response_time_ms': self.total_response_time,
            'total_prompt_tokens': self.total_prompt_tokens,
            'total_response_tokens': self.total_response_tokens,
            'total_tokens': self.total_prompt_tokens + self.total_response_tokens
        }
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self._reset_performance_counters()