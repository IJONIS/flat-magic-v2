"""Base prompt management functionality.

This module provides the foundation for all prompt managers
with common functionality and templates.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BasePromptManager(ABC):
    """Base class for prompt managers.
    
    Provides common functionality for all prompt managers including
    template rendering, data limiting, and validation.
    """
    
    def __init__(self):
        """Initialize base prompt manager."""
        self.default_data_limits = {
            'mandatory_fields': 15,
            'product_data': 20,
            'examples': 5
        }
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this prompt type.
        
        Returns:
            System prompt string
        """
        pass
    
    def limit_data_size(
        self, 
        data: Dict[str, Any], 
        limit: int, 
        key_priority: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Limit data size for prompt efficiency.
        
        Args:
            data: Data dictionary to limit
            limit: Maximum number of items to include
            key_priority: Optional list of keys to prioritize
            
        Returns:
            Limited data dictionary
        """
        if not data or len(data) <= limit:
            return data
        
        # If priority keys specified, include them first
        if key_priority:
            limited_data = {}
            remaining_limit = limit
            
            # Add priority keys first
            for key in key_priority:
                if key in data and remaining_limit > 0:
                    limited_data[key] = data[key]
                    remaining_limit -= 1
            
            # Add remaining keys up to limit
            for key, value in data.items():
                if key not in limited_data and remaining_limit > 0:
                    limited_data[key] = value
                    remaining_limit -= 1
            
            return limited_data
        
        # No priority - just take first N items
        return dict(list(data.items())[:limit])
    
    def format_json_data(
        self, 
        data: Dict[str, Any], 
        indent: int = 2
    ) -> str:
        """Format data as JSON string for prompts.
        
        Args:
            data: Data to format
            indent: JSON indentation level
            
        Returns:
            Formatted JSON string
        """
        try:
            return json.dumps(data, indent=indent, ensure_ascii=False)
        except (TypeError, ValueError):
            # Fallback to string representation
            return str(data)
    
    def create_example_section(
        self, 
        examples: List[Dict[str, Any]], 
        title: str = "EXAMPLES"
    ) -> str:
        """Create examples section for prompts.
        
        Args:
            examples: List of example dictionaries
            title: Section title
            
        Returns:
            Formatted examples section
        """
        if not examples:
            return ""
        
        section = f"\n{title}:\n"
        for i, example in enumerate(examples[:self.default_data_limits['examples']], 1):
            section += f"\nExample {i}:\n{self.format_json_data(example)}\n"
        
        return section
    
    def validate_prompt_data(
        self, 
        required_fields: List[str], 
        data: Dict[str, Any]
    ) -> List[str]:
        """Validate that required fields are present in data.
        
        Args:
            required_fields: List of required field names
            data: Data dictionary to validate
            
        Returns:
            List of missing field names
        """
        missing_fields = []
        for field in required_fields:
            if field not in data or data[field] is None:
                missing_fields.append(field)
        
        return missing_fields
    
    def create_instruction_section(
        self, 
        instructions: List[str], 
        title: str = "INSTRUCTIONS"
    ) -> str:
        """Create numbered instructions section.
        
        Args:
            instructions: List of instruction strings
            title: Section title
            
        Returns:
            Formatted instructions section
        """
        section = f"\n{title}:\n"
        for i, instruction in enumerate(instructions, 1):
            section += f"{i}. {instruction}\n"
        
        return section
    
    def create_output_format_section(
        self, 
        format_example: Dict[str, Any], 
        title: str = "REQUIRED JSON OUTPUT FORMAT"
    ) -> str:
        """Create output format section with example.
        
        Args:
            format_example: Example of expected output format
            title: Section title
            
        Returns:
            Formatted output format section
        """
        return f"\n{title}:\n{self.format_json_data(format_example)}\n"
    
    def render_template(
        self, 
        template: str, 
        variables: Dict[str, Any]
    ) -> str:
        """Render template with variables.
        
        Args:
            template: Template string with {variable} placeholders
            variables: Dictionary of variables to substitute
            
        Returns:
            Rendered template string
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
        except Exception as e:
            raise ValueError(f"Template rendering failed: {e}")