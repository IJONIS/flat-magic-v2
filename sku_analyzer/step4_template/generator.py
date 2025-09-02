"""Main template generator coordinating field analysis and template creation.

This module provides the main TemplateGenerator class that orchestrates
the entire template generation process from mandatory fields analysis.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sku_analyzer.shared.performance import PerformanceMonitor
from sku_analyzer.shared.validation import ValidationUtils
from .field_analyzer import FieldAnalyzer
from .template_validator import TemplateValidator
from .field_processor import FieldProcessor


class TemplateGenerationError(Exception):
    """Raised when template generation fails."""
    pass


class TemplateGenerator:
    """Main generator for creating reusable parent-child structure templates.
    
    This class coordinates the entire template generation process including:
    - Field analysis and categorization (AI-powered or deterministic)
    - Template structure creation
    - Validation and quality assessment
    - Performance monitoring
    """

    def __init__(
        self, 
        enable_performance_monitoring: bool = True, 
        enable_ai_categorization: bool = True
    ) -> None:
        """Initialize template generator.
        
        Args:
            enable_performance_monitoring: Whether to enable performance tracking
            enable_ai_categorization: Whether to use AI-powered field categorization
        """
        self.logger = self._setup_logging()
        self.enable_monitoring = enable_performance_monitoring
        self.enable_ai = enable_ai_categorization
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor(enable_performance_monitoring)
        self.field_analyzer = FieldAnalyzer(enable_ai_categorization)
        self.template_validator = TemplateValidator()

    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging.
        
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    async def generate_template_from_mandatory_fields(
        self,
        step3_mandatory_path: str | Path,
        output_path: str | Path
    ) -> Dict[str, Any]:
        """Main template generation function.
        
        Args:
            step3_mandatory_path: Path to step3_mandatory_fields.json
            output_path: Path for step4_template.json output
            
        Returns:
            Generated template structure
            
        Raises:
            TemplateGenerationError: When template generation fails
            FileNotFoundError: If input file doesn't exist
        """
        start_time = time.time()
        self.logger.info("Starting template generation from mandatory fields")

        try:
            with self.performance_monitor.measure_performance("template_generation") as perf:
                # Load mandatory fields data
                mandatory_fields = self._load_json_file(Path(step3_mandatory_path))
                
                if not mandatory_fields:
                    raise TemplateGenerationError("No mandatory fields found in input data")

                # Categorize fields into parent and variant levels
                parent_fields, variant_fields = await self.field_analyzer.categorize_field_levels(
                    mandatory_fields
                )
                
                # Create template structure
                template_structure = self._create_template_structure(
                    parent_fields, variant_fields, mandatory_fields
                )
                
                # Validate template quality
                validation_result = self.template_validator.validate_template(template_structure)
                if not validation_result['valid']:
                    self.logger.warning(f"Template validation issues: {validation_result['issues']}")
                
                # Create final template with metadata
                template_output = self._create_template_output(
                    template_structure, validation_result, Path(step3_mandatory_path).name
                )
                
                # Add field analyzer metadata
                template_output['metadata']['categorization_method'] = (
                    self.field_analyzer.get_categorization_method()
                )
                if self.enable_ai:
                    template_output['metadata']['ai_confidence'] = (
                        self.field_analyzer.get_ai_confidence()
                    )
                
                # Save template
                self._save_json_file(Path(output_path), template_output)
                
                duration_ms = (time.time() - start_time) * 1000
                self.logger.info(
                    f"Template generation completed in {duration_ms:.2f}ms - "
                    f"{len(parent_fields)} parent fields, {len(variant_fields)} variant fields"
                )
                
                return template_output

        except Exception as e:
            self.logger.error(f"Template generation failed: {e}")
            raise TemplateGenerationError(f"Template generation failed: {e}") from e

    def _create_template_structure(
        self,
        parent_fields: List[str],
        variant_fields: List[str],
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create structured template for parent-child relationships.
        
        Args:
            parent_fields: List of parent-level field names
            variant_fields: List of variant-level field names  
            mandatory_fields: Complete mandatory fields data
            
        Returns:
            Template structure dictionary
        """
        template_structure = {
            'parent_product': {
                'fields': {},
                'field_count': len(parent_fields),
                'required_fields': []
            },
            'child_variants': {
                'fields': {},
                'field_count': len(variant_fields),
                'variable_fields': [],
                'inherited_fields': []
            },
            'field_relationships': {
                'parent_defines': [],
                'variant_overrides': [],
                'shared_constraints': {}
            }
        }
        
        # Process parent fields
        for field_name in parent_fields:
            field_data = mandatory_fields[field_name]
            template_structure['parent_product']['fields'][field_name] = {
                'display_name': field_data['display_name'],
                'data_type': field_data['data_type'],
                'constraints': field_data['constraints'],
                'applies_to_children': FieldProcessor.determines_child_inheritance(field_data),
                'validation_rules': FieldProcessor.create_validation_rules(field_data)
            }
            
            # Mark as required if has limited valid values
            if FieldProcessor.is_required_field(field_data):
                template_structure['parent_product']['required_fields'].append(field_name)
        
        # Process variant fields
        for field_name in variant_fields:
            field_data = mandatory_fields[field_name]
            template_structure['child_variants']['fields'][field_name] = {
                'display_name': field_data['display_name'],
                'data_type': field_data['data_type'],
                'constraints': field_data['constraints'],
                'variation_type': FieldProcessor.determine_variation_type(field_data),
                'validation_rules': FieldProcessor.create_validation_rules(field_data)
            }
            
            # Categorize variant field behavior
            if FieldProcessor.is_variable_field(field_data):
                template_structure['child_variants']['variable_fields'].append(field_name)
            
            if FieldProcessor.can_inherit_from_parent(field_data):
                template_structure['child_variants']['inherited_fields'].append(field_name)
        
        # Define field relationships
        template_structure['field_relationships'] = FieldProcessor.create_field_relationships(
            parent_fields, variant_fields, mandatory_fields
        )
        
        return template_structure

    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and validate JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")

    def _save_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save data to JSON file with formatting.
        
        Args:
            file_path: Output file path
            data: Data to save
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


    def _create_template_output(
        self,
        template_structure: Dict[str, Any],
        validation_result: Dict[str, Any],
        source_file: str
    ) -> Dict[str, Any]:
        """Create final template output with metadata.
        
        Args:
            template_structure: Generated template structure
            validation_result: Validation results
            source_file: Source file name
            
        Returns:
            Complete template output
        """
        return {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'source_file': source_file,
                'template_version': '1.0',
                'field_distribution': validation_result['field_distribution'],
                'quality_score': validation_result['quality_score'],
                'validation_status': 'valid' if validation_result['valid'] else 'issues_found',
                'warnings': validation_result.get('warnings', [])
            },
            'template_structure': template_structure,
            'usage_instructions': {
                'description': 'Template for structured parent-child product mapping',
                'parent_product_usage': 'Define shared characteristics and product family',
                'child_variants_usage': 'Define variable attributes and specific variants',
                'inheritance_rules': 'Children inherit parent values unless overridden'
            }
        }