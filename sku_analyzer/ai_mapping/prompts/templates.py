"""Template management for AI mapping prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
from jinja2 import Environment, FileSystemLoader, Template


class PromptTemplateManager:
    """Manages Jinja2 templates for AI mapping prompts."""
    
    def __init__(self, template_dir: Path | None = None):
        """Initialize template manager.
        
        Args:
            template_dir: Directory containing template files
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "files"
        
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def render_mapping_prompt(self, context: Dict[str, Any]) -> str:
        """Render product transformation prompt with enhanced context.
        
        Args:
            context: Template context data containing mandatory fields and product data
            
        Returns:
            Rendered prompt string requesting data transformation with variance analysis
        """
        # Enhance context with variance analysis if available
        if "product_data" in context:
            from ..data_transformer import DataVarianceAnalyzer
            analyzer = DataVarianceAnalyzer()
            variance_analysis = analyzer.analyze_product_data(context["product_data"])
            context["variance_analysis"] = variance_analysis
        
        # Add example output structure for format validation
        from ..example_loader import ExampleFormatLoader
        example_loader = ExampleFormatLoader()
        try:
            example_structure = example_loader.load_example_structure()
            context["example_output"] = example_structure
            context["required_schema"] = example_loader.get_required_fields_schema()
        except Exception as e:
            # Log warning but continue without example
            import logging
            logging.getLogger(__name__).warning(f"Could not load example structure: {e}")
        
        template = self.env.get_template("product_transformation.jinja2")
        return template.render(**context)
    
    def render_legacy_mapping_prompt(self, context: Dict[str, Any]) -> str:
        """Render legacy product mapping prompt (DEPRECATED).
        
        This method provides the old field-mapping approach for backward compatibility.
        New implementations should use render_mapping_prompt() for data transformation.
        
        Args:
            context: Template context data
            
        Returns:
            Rendered legacy mapping prompt string
        """
        template = self.env.get_template("product_mapping.jinja2") 
        return template.render(**context)
    
    def render_validation_prompt(self, context: Dict[str, Any]) -> str:
        """Render validation prompt with context.
        
        Args:
            context: Template context data
            
        Returns:
            Rendered prompt string
        """
        template = self.env.get_template("validation.jinja2")
        return template.render(**context)
    
    def render_format_validation_prompt(self, context: Dict[str, Any]) -> str:
        """Render format validation prompt for AI output verification.
        
        Args:
            context: Template context data containing ai_output and validation_errors
            
        Returns:
            Rendered format validation prompt string
        """
        template = self.env.get_template("format_validation.jinja2")
        return template.render(**context)
    
    def get_system_prompt(self) -> str:
        """Get system prompt for AI agent.
        
        Returns:
            System prompt string
        """
        template = self.env.get_template("system_prompt.jinja2")
        return template.render()