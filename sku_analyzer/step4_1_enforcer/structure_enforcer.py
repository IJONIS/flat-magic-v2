"""
Template Structure Enforcer for deterministic field categorization and AI mapping compliance.

This module generates explicit parent/variant structure examples from step4_template.json
to ensure consistent AI mapping output format with 100% field coverage.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import orjson


class TemplateStructureEnforcer:
    """
    Generates deterministic structure examples from step4_template.json for AI mapping compliance.
    
    This class provides explicit parent_data and variants structure templates to ensure
    AI mapping produces consistent output format with all 23 mandatory fields categorized.
    """
    
    # Deterministic field categorization based on Amazon requirements
    PARENT_FIELDS = {
        'feed_product_type',
        'brand_name', 
        'external_product_id_type',
        'item_name',
        'recommended_browse_nodes',
        'outer_material_type',
        'target_gender',
        'age_range_description', 
        'bottoms_size_system',
        'bottoms_size_class',
        'main_image_url',
        'department_name',
        'country_of_origin',
        'fabric_type'
    }
    
    VARIANT_FIELDS = {
        'item_sku',
        'external_product_id',
        'standard_price',
        'quantity',
        'color_map',
        'color_name',
        'size_name',
        'size_map',
        'list_price_with_tax'
    }
    
    def __init__(self):
        """Initialize the Template Structure Enforcer."""
        self.logger = self._setup_logging()
        self._validate_field_categorization()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_field_categorization(self) -> None:
        """Validate that field categorization covers all 23 mandatory fields."""
        total_fields = len(self.PARENT_FIELDS) + len(self.VARIANT_FIELDS)
        expected_fields = 23
        
        if total_fields != expected_fields:
            raise ValueError(
                f"Field categorization error: Expected {expected_fields} fields, "
                f"got {total_fields} ({len(self.PARENT_FIELDS)} parent + "
                f"{len(self.VARIANT_FIELDS)} variant)"
            )
        
        # Check for field overlap
        field_overlap = self.PARENT_FIELDS & self.VARIANT_FIELDS
        if field_overlap:
            raise ValueError(f"Field categorization overlap detected: {field_overlap}")
        
        self.logger.info(
            f"✅ Field categorization validated: {len(self.PARENT_FIELDS)} parent, "
            f"{len(self.VARIANT_FIELDS)} variant fields"
        )
    
    def load_step4_template(self, job_dir: Path) -> Dict[str, Any]:
        """
        Load step4_template.json from job directory.
        
        Args:
            job_dir: Job output directory containing flat_file_analysis
            
        Returns:
            Template data structure
            
        Raises:
            FileNotFoundError: When step4_template.json doesn't exist
            ValueError: When template data is invalid
        """
        template_file = job_dir / "flat_file_analysis" / "step4_template.json"
        
        if not template_file.exists():
            raise FileNotFoundError(f"Step 4 template not found: {template_file}")
        
        try:
            with open(template_file, 'rb') as f:
                template_data = orjson.loads(f.read())
                
            # Validate required template structure
            if 'template_structure' not in template_data:
                raise ValueError("Invalid template: missing 'template_structure' section")
            
            template_structure = template_data['template_structure']
            if 'parent_product' not in template_structure:
                raise ValueError("Invalid template: missing 'parent_product' section")
            if 'child_variants' not in template_structure:
                raise ValueError("Invalid template: missing 'child_variants' section")
            
            self.logger.info(f"✅ Step 4 template loaded: {template_file}")
            return template_data
            
        except (json.JSONDecodeError, orjson.JSONDecodeError) as e:
            raise ValueError(f"Invalid JSON in template file: {e}") from e
    
    def categorize_fields(self, template_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deterministically categorize ALL 23 mandatory fields into parent/variant groups.
        
        Args:
            template_data: Step 4 template structure
            
        Returns:
            Categorized field structure with validation
        """
        template_structure = template_data['template_structure']
        parent_fields = template_structure.get('parent_product', {}).get('fields', {})
        variant_fields = template_structure.get('child_variants', {}).get('fields', {})
        
        # Merge all available fields from template
        all_template_fields = set(parent_fields.keys()) | set(variant_fields.keys())
        
        # Categorize using deterministic rules
        categorized_parent = {}
        categorized_variant = {}
        
        # Process parent fields
        for field_name in self.PARENT_FIELDS:
            if field_name in parent_fields:
                categorized_parent[field_name] = self._extract_field_example_value(
                    field_name, parent_fields[field_name]
                )
            elif field_name in variant_fields:
                categorized_parent[field_name] = self._extract_field_example_value(
                    field_name, variant_fields[field_name]
                )
            else:
                # Provide default example value
                categorized_parent[field_name] = self._get_default_example_value(field_name)
        
        # Process variant fields  
        for field_name in self.VARIANT_FIELDS:
            if field_name in variant_fields:
                categorized_variant[field_name] = self._extract_field_example_value(
                    field_name, variant_fields[field_name]
                )
            elif field_name in parent_fields:
                categorized_variant[field_name] = self._extract_field_example_value(
                    field_name, parent_fields[field_name]
                )
            else:
                # Provide default example value
                categorized_variant[field_name] = self._get_default_example_value(field_name)
        
        # Validate field coverage
        parent_coverage = len(categorized_parent)
        variant_coverage = len(categorized_variant)
        total_coverage = parent_coverage + variant_coverage
        
        self.logger.info(
            f"🎯 Field categorization complete: {parent_coverage} parent, "
            f"{variant_coverage} variant, {total_coverage} total fields"
        )
        
        return {
            'parent_data': categorized_parent,
            'variant_template': categorized_variant,
            'field_categorization': {
                'parent_fields': sorted(list(self.PARENT_FIELDS)),
                'variant_fields': sorted(list(self.VARIANT_FIELDS)),
                'total_fields': total_coverage,
                'coverage_validation': {
                    'parent_coverage': parent_coverage,
                    'variant_coverage': variant_coverage,
                    'expected_total': 23,
                    'coverage_complete': total_coverage == 23
                }
            }
        }
    
    def _extract_field_example_value(self, field_name: str, field_config: Dict[str, Any]) -> str:
        """
        Extract or generate example value for a field based on its configuration.
        
        Args:
            field_name: Name of the field
            field_config: Field configuration from template
            
        Returns:
            Example value for the field
        """
        # Check for allowed values and use the first one
        validation_rules = field_config.get('validation_rules', {})
        allowed_values = validation_rules.get('allowed_values', [])
        
        if allowed_values:
            # Use first allowed value as example
            return str(allowed_values[0])
        
        # Fall back to default example values
        return self._get_default_example_value(field_name)
    
    def _get_default_example_value(self, field_name: str) -> str:
        """
        Get default example value for field when no template data is available.
        
        Args:
            field_name: Name of the field
            
        Returns:
            Default example value
        """
        # Default example values for consistent AI mapping
        default_values = {
            'feed_product_type': 'pants',
            'brand_name': 'EIKO',
            'external_product_id_type': 'EAN',
            'item_name': 'PERCY Zunfthose',
            'recommended_browse_nodes': '1981663031',
            'outer_material_type': 'Cord',
            'target_gender': 'Männlich',
            'age_range_description': 'Erwachsener',
            'bottoms_size_system': 'DE / NL / SE / PL',
            'bottoms_size_class': 'Numerisch',
            'main_image_url': 'https://example.com/image.jpg',
            'department_name': 'Herren',
            'country_of_origin': 'Deutschland',
            'fabric_type': 'Cotton',
            'item_sku': '41282_40_44',
            'external_product_id': '4033976004549',
            'standard_price': '49.99',
            'quantity': '10',
            'color_map': 'Schwarz',
            'color_name': 'Schwarz',
            'size_name': '44',
            'size_map': '44',
            'list_price_with_tax': '59.49'
        }
        
        return default_values.get(field_name, f'example_{field_name}')
    
    def generate_structure_example(self, categorized_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate explicit parent_data and variants structure example for AI mapping.
        
        Args:
            categorized_data: Output from categorize_fields()
            
        Returns:
            Complete structure example with metadata
        """
        parent_data = categorized_data['parent_data']
        variant_template = categorized_data['variant_template']
        field_categorization = categorized_data['field_categorization']
        
        # Create variant examples with different values
        variants = []
        
        # Variant 1
        variant_1 = variant_template.copy()
        variants.append({'variant_1': variant_1})
        
        # Variant 2 with modified values
        variant_2 = variant_template.copy()
        variant_2.update({
            'item_sku': '41282_40_46',
            'external_product_id': '4033976004556',
            'quantity': '8',
            'size_name': '46',
            'size_map': '46'
        })
        variants.append({'variant_2': variant_2})
        
        structure_example = {
            'structure_version': '1.0',
            'generation_timestamp': datetime.now().isoformat() + 'Z',
            'mandatory_field_coverage': f"{field_categorization['total_fields']}/23",
            'parent_data': parent_data,
            'variants': variants,
            'field_categorization': field_categorization
        }
        
        self.logger.info(
            f"✅ Structure example generated: {len(parent_data)} parent fields, "
            f"{len(variants)} variant examples"
        )
        
        return structure_example
    
    def save_structure_example(
        self, 
        job_dir: Path, 
        structure_example: Dict[str, Any]
    ) -> Path:
        """
        Save step4_1_structure_example.json to flat_file_analysis directory.
        
        Args:
            job_dir: Job output directory
            structure_example: Structure example to save
            
        Returns:
            Path to saved file
            
        Raises:
            OSError: When file cannot be written
        """
        flat_file_dir = job_dir / "flat_file_analysis"
        flat_file_dir.mkdir(exist_ok=True)
        
        output_file = flat_file_dir / "step4_1_structure_example.json"
        
        try:
            # Use orjson for fast, deterministic serialization
            json_bytes = orjson.dumps(
                structure_example,
                option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS
            )
            
            with open(output_file, 'wb') as f:
                f.write(json_bytes)
            
            self.logger.info(f"✅ Structure example saved: {output_file}")
            return output_file
            
        except OSError as e:
            self.logger.error(f"Failed to save structure example: {e}")
            raise
    
    def process_template_structure(self, job_dir: Path) -> Dict[str, Any]:
        """
        Complete Step 4.1 processing: load template, categorize fields, generate example.
        
        Args:
            job_dir: Job output directory
            
        Returns:
            Generated structure example
            
        Raises:
            FileNotFoundError: When step4_template.json is missing
            ValueError: When template data is invalid
        """
        self.logger.info(f"🔧 Starting Step 4.1 template structure enforcement for {job_dir.name}")
        
        # Load Step 4 template
        template_data = self.load_step4_template(job_dir)
        
        # Categorize fields deterministically
        categorized_data = self.categorize_fields(template_data)
        
        # Generate structure example
        structure_example = self.generate_structure_example(categorized_data)
        
        # Save structure example
        output_file = self.save_structure_example(job_dir, structure_example)
        
        # Validate results
        coverage = categorized_data['field_categorization']['coverage_validation']
        if not coverage['coverage_complete']:
            self.logger.warning(
                f"⚠️ Incomplete field coverage: {coverage['parent_coverage'] + coverage['variant_coverage']}/23 fields"
            )
        
        self.logger.info(
            f"✅ Step 4.1 template structure enforcement completed: "
            f"{coverage['parent_coverage']} parent + {coverage['variant_coverage']} variant fields"
        )
        
        return structure_example


def process_step4_1_for_job(job_dir: Path) -> Dict[str, Any]:
    """
    Convenience function to process Step 4.1 for a job directory.
    
    Args:
        job_dir: Job output directory containing step4_template.json
        
    Returns:
        Generated structure example
    """
    enforcer = TemplateStructureEnforcer()
    return enforcer.process_template_structure(job_dir)