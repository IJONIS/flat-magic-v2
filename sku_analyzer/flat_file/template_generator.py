"""Step 3: Template generator creating reusable parent-child structure templates.

This module processes mandatory fields analysis to create structured templates
that define optimal parent-child relationships for efficient AI mapping.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .performance_monitor import PerformanceMonitor
from .validation_utils import ValidationUtils
from ..shared.gemini_client import GeminiClient as ModernGeminiClient
from ..step5_mapping.models import AIProcessingConfig


class TemplateGenerationError(Exception):
    """Raised when template generation fails."""
    pass


class AICategorization(Exception):
    """Raised when AI categorization fails."""
    pass


class TemplateGenerator:
    """Generator for creating reusable parent-child structure templates."""

    def __init__(self, enable_performance_monitoring: bool = True, enable_ai_categorization: bool = True) -> None:
        """Initialize template generator.
        
        Args:
            enable_performance_monitoring: Whether to enable performance tracking
            enable_ai_categorization: Whether to use AI-powered field categorization
        """
        self.logger = self._setup_logging()
        self.enable_monitoring = enable_performance_monitoring
        self.enable_ai = enable_ai_categorization
        self.performance_monitor = PerformanceMonitor(enable_performance_monitoring)
        self.validation_utils = ValidationUtils()
        
        # AI configuration for field categorization
        if self.enable_ai:
            self.ai_config = AIProcessingConfig(
                model_name="gemini-2.5-flash",
                temperature=0.1,
                max_tokens=4096,
                timeout_seconds=30,
                max_concurrent=1
            )
            self.ai_client = None  # Lazy initialization
        
        # Fallback configuration for field categorization (deterministic approach)
        self._parent_level_indicators = {
            'brand', 'category', 'manufacturer', 'product_type', 'family',
            'series', 'collection', 'line', 'group', 'classification'
        }
        
        self._variant_level_indicators = {
            'size', 'color', 'color_name', 'material', 'style', 'variant',
            'configuration', 'option', 'specification', 'dimension'
        }

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
            # Load mandatory fields data - it's the fields directly, not nested
            mandatory_fields = self._load_json_file(Path(step3_mandatory_path))
            
            if not mandatory_fields:
                raise TemplateGenerationError("No mandatory fields found in input data")

            # Categorize fields into parent and variant levels
            parent_fields, variant_fields = await self.categorize_field_levels(mandatory_fields)
            
            # Create template structure
            template_structure = self.create_template_structure(
                parent_fields, variant_fields, mandatory_fields
            )
            
            # Validate template quality
            validation_result = self.validate_template(template_structure)
            if not validation_result['valid']:
                self.logger.warning(f"Template validation issues: {validation_result['issues']}")
            
            # Create final template with metadata
            template_output = self._create_template_output(
                template_structure, validation_result, Path(step3_mandatory_path).name
            )
            
            # Add AI categorization metadata if available
            if hasattr(self, '_last_categorization_method'):
                template_output['metadata']['categorization_method'] = self._last_categorization_method
                if hasattr(self, '_last_ai_confidence'):
                    template_output['metadata']['ai_confidence'] = self._last_ai_confidence
            
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

    async def categorize_field_levels(
        self, 
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Categorize fields into parent and variant levels using AI or fallback.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Tuple of (parent_fields, variant_fields) lists
        """
        if self.enable_ai:
            try:
                # Use AI-powered categorization
                result = await self._ai_categorize_fields(mandatory_fields)
                self._last_categorization_method = "ai"
                return result
            except Exception as e:
                self.logger.warning(
                    f"AI categorization failed: {e}. Falling back to deterministic approach."
                )
                # Fall through to deterministic approach
        
        # Deterministic categorization (original logic)
        result = self._deterministic_categorize_fields(mandatory_fields)
        self._last_categorization_method = "deterministic"
        return result

    async def _ai_categorize_fields(
        self, 
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Use AI to categorize fields into parent and variant levels.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Tuple of (parent_fields, variant_fields) lists
            
        Raises:
            AICategorization: When AI categorization fails
        """
        start_time = time.time()
        self.logger.info("Starting AI-powered field categorization")
        
        try:
            # Initialize AI client if needed
            if self.ai_client is None:
                self.ai_client = ModernGeminiClient(
                    config=self.ai_config,
                    performance_tracker=None
                )
            
            # Create AI prompt for field categorization
            prompt = self._create_categorization_prompt(mandatory_fields)
            
            # Execute AI categorization
            response = await self.ai_client.generate_mapping(
                prompt=prompt,
                operation_name="field_categorization"
            )
            
            # Parse and validate AI response
            categorization_result = await self.ai_client.validate_json_response(response)
            validated_result = self._validate_ai_categorization(
                categorization_result, mandatory_fields
            )
            
            parent_fields = validated_result['parent_fields']
            variant_fields = validated_result['variant_fields']
            
            # Apply critical field placement rules
            parent_fields, variant_fields = self._ensure_critical_field_placement(
                parent_fields, variant_fields, mandatory_fields
            )
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"AI categorization completed in {duration_ms:.2f}ms - "
                f"{len(parent_fields)} parent, {len(variant_fields)} variant fields"
            )
            
            # Log AI reasoning for transparency
            if 'reasoning' in validated_result:
                self.logger.debug(f"AI categorization reasoning: {validated_result['reasoning']}")
            
            # Store AI confidence for metadata
            self._last_ai_confidence = validated_result.get('confidence', 0.8)
            
            return parent_fields, variant_fields
            
        except Exception as e:
            self.logger.error(f"AI field categorization failed: {e}")
            raise AICategorization(f"AI categorization failed: {e}") from e
    
    def _deterministic_categorize_fields(
        self, 
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Deterministic field categorization (original logic).
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Tuple of (parent_fields, variant_fields) lists
        """
        parent_fields = []
        variant_fields = []
        
        for field_name, field_data in mandatory_fields.items():
            display_name = field_data.get('display_name', '').lower()
            technical_name = field_name.lower()
            
            # Check for parent-level indicators
            is_parent_field = any(
                indicator in technical_name or indicator in display_name
                for indicator in self._parent_level_indicators
            )
            
            # Check for variant-level indicators
            is_variant_field = any(
                indicator in technical_name or indicator in display_name
                for indicator in self._variant_level_indicators
            )
            
            # Analyze field characteristics
            field_characteristics = self._analyze_field_characteristics(field_data)
            
            # Decision logic based on multiple factors
            if is_parent_field and not is_variant_field:
                parent_fields.append(field_name)
            elif is_variant_field and not is_parent_field:
                variant_fields.append(field_name)
            else:
                # Use field characteristics for ambiguous cases
                if field_characteristics['is_likely_parent']:
                    parent_fields.append(field_name)
                else:
                    variant_fields.append(field_name)
        
        # Ensure critical fields are properly categorized
        parent_fields, variant_fields = self._ensure_critical_field_placement(
            parent_fields, variant_fields, mandatory_fields
        )
        
        self.logger.info(
            f"Deterministic categorization: {len(parent_fields)} parent, {len(variant_fields)} variant"
        )
        
        return parent_fields, variant_fields
    
    def _create_categorization_prompt(self, mandatory_fields: Dict[str, Dict[str, Any]]) -> str:
        """Create AI prompt for field categorization.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            Formatted prompt for AI categorization
        """
        
        prompt = f"""You are an expert e-commerce field categorization AI. Analyze these mandatory fields and categorize them into parent-level (shared across variants) vs variant-level (unique per variant).

MANDATORY FIELDS TO ANALYZE:
{json.dumps(mandatory_fields, indent=2)}

CATEGORIZATION RULES:
- Parent-level: brand, material, category, gender, age group, country of origin, product type, department
- Variant-level: size, color, SKU, price, individual identifiers, specific measurements

Return ONLY valid JSON in this exact format (no comments, no extra text):
{{
  "parent_fields": ["field1", "field2"],
  "variant_fields": ["field3", "field4"],
  "confidence": 0.95,
  "reasoning": "Brief explanation of categorization logic"
}}

Analyze and categorize now:"""
        
        return prompt
    
    def _validate_ai_categorization(
        self, 
        ai_result: Dict[str, Any], 
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate AI categorization response.
        
        Args:
            ai_result: AI categorization response
            mandatory_fields: Original mandatory fields data
            
        Returns:
            Validated categorization result
            
        Raises:
            AICategorization: If validation fails
        """
        try:
            # Check required keys in new format
            if 'parent_level_fields' in ai_result and 'variant_level_fields' in ai_result:
                # New format - extract field names from nested objects
                parent_fields = list(ai_result['parent_level_fields'].keys())
                variant_fields = list(ai_result['variant_level_fields'].keys())
            elif 'parent_fields' in ai_result and 'variant_fields' in ai_result:
                # Fallback format - simple lists
                parent_fields = ai_result['parent_fields']
                variant_fields = ai_result['variant_fields']
            else:
                raise AICategorization("Missing required keys: parent_level_fields/variant_level_fields or parent_fields/variant_fields")
            
            # Validate field lists
            if not isinstance(parent_fields, list) or not isinstance(variant_fields, list):
                raise AICategorization("Field categories must be lists")
            
            # Check all fields are categorized
            all_mandatory_fields = set(mandatory_fields.keys())
            categorized_fields = set(parent_fields + variant_fields)
            
            missing_fields = all_mandatory_fields - categorized_fields
            if missing_fields:
                self.logger.warning(f"AI missed fields: {missing_fields}. Adding to variant level.")
                variant_fields.extend(list(missing_fields))
            
            extra_fields = categorized_fields - all_mandatory_fields
            if extra_fields:
                self.logger.warning(f"AI added unknown fields: {extra_fields}. Removing.")
                parent_fields = [f for f in parent_fields if f in all_mandatory_fields]
                variant_fields = [f for f in variant_fields if f in all_mandatory_fields]
            
            # Check for field duplicates
            duplicates = set(parent_fields) & set(variant_fields)
            if duplicates:
                # Remove from variant, keep in parent
                self.logger.warning(f"Duplicate fields found: {duplicates}. Keeping in parent.")
                variant_fields = [f for f in variant_fields if f not in duplicates]
            
            # Extract confidence from metadata or ai_analysis_summary
            confidence = 0.8  # default
            if 'metadata' in ai_result:
                confidence = ai_result['metadata'].get('categorization_confidence', 0.8)
            elif 'ai_analysis_summary' in ai_result:
                confidence = 0.9  # high confidence for detailed analysis
            
            if not isinstance(confidence, (int, float)) or not (0.0 <= confidence <= 1.0):
                confidence = 0.8
            
            # Extract reasoning
            reasoning = "AI-powered categorization"
            if 'ai_analysis_summary' in ai_result:
                summary = ai_result['ai_analysis_summary']
                reasoning = f"{summary.get('categorization_approach', 'AI analysis')}: {summary.get('confidence_notes', 'Intelligent field categorization')}"
            
            return {
                'parent_fields': parent_fields,
                'variant_fields': variant_fields,
                'reasoning': reasoning,
                'confidence': confidence
            }
            
        except Exception as e:
            raise AICategorization(f"AI response validation failed: {e}") from e

    def create_template_structure(
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
                'applies_to_children': self._determines_child_inheritance(field_data),
                'validation_rules': self._create_validation_rules(field_data)
            }
            
            # Mark as required if has limited valid values
            if self._is_required_field(field_data):
                template_structure['parent_product']['required_fields'].append(field_name)
        
        # Process variant fields
        for field_name in variant_fields:
            field_data = mandatory_fields[field_name]
            template_structure['child_variants']['fields'][field_name] = {
                'display_name': field_data['display_name'],
                'data_type': field_data['data_type'],
                'constraints': field_data['constraints'],
                'variation_type': self._determine_variation_type(field_data),
                'validation_rules': self._create_validation_rules(field_data)
            }
            
            # Categorize variant field behavior
            if self._is_variable_field(field_data):
                template_structure['child_variants']['variable_fields'].append(field_name)
            
            if self._can_inherit_from_parent(field_data):
                template_structure['child_variants']['inherited_fields'].append(field_name)
        
        # Define field relationships
        template_structure['field_relationships'] = self._create_field_relationships(
            parent_fields, variant_fields, mandatory_fields
        )
        
        return template_structure

    def validate_template(self, template_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate template structure and quality.
        
        Args:
            template_structure: Generated template structure
            
        Returns:
            Validation result with status and issues
        """
        issues = []
        warnings = []
        
        # Check parent fields
        parent_fields = template_structure.get('parent_product', {}).get('fields', {})
        if not parent_fields:
            issues.append("No parent fields defined")
        elif len(parent_fields) < 2:
            warnings.append("Very few parent fields - may not provide sufficient structure")
        
        # Check variant fields
        variant_fields = template_structure.get('child_variants', {}).get('fields', {})
        if not variant_fields:
            warnings.append("No variant fields defined - all products will be identical")
        
        # Check field balance
        parent_count = len(parent_fields)
        variant_count = len(variant_fields)
        total_fields = parent_count + variant_count
        
        if total_fields > 0:
            parent_ratio = parent_count / total_fields
            if parent_ratio > 0.8:
                warnings.append("Too many parent fields - may limit variant flexibility")
            elif parent_ratio < 0.2:
                warnings.append("Too few parent fields - may lack product structure")
        
        # Validate field relationships
        relationships = template_structure.get('field_relationships', {})
        if not relationships.get('parent_defines'):
            warnings.append("No parent-defined relationships - limited inheritance")
        
        # Check data type distribution
        data_types = set()
        for field_data in list(parent_fields.values()) + list(variant_fields.values()):
            data_types.add(field_data.get('data_type', 'unknown'))
        
        if len(data_types) < 2:
            warnings.append("Limited data type diversity in template")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'quality_score': self._calculate_quality_score(template_structure),
            'field_distribution': {
                'parent_fields': parent_count,
                'variant_fields': variant_count,
                'parent_ratio': parent_count / total_fields if total_fields > 0 else 0
            }
        }

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

    def _analyze_field_characteristics(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze field characteristics to determine parent/variant placement.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Analysis results
        """
        valid_values = field_data.get('valid_values', [])
        constraints = field_data.get('constraints', {})
        data_type = field_data.get('data_type', 'string')
        
        # Calculate uniqueness ratio
        value_count = constraints.get('value_count', len(valid_values))
        max_length = constraints.get('max_length', 0)
        
        # Characteristics that suggest parent-level field
        is_likely_parent = (
            value_count <= 10 or  # Limited options suggest categorization
            data_type in ['boolean'] or  # Boolean fields often define product categories
            max_length > 100 or  # Long text fields often describe product families
            any(keyword in str(field_data.get('display_name', '')).lower() 
                for keyword in ['typ', 'kategorie', 'klasse', 'gruppe'])
        )
        
        return {
            'is_likely_parent': is_likely_parent,
            'value_count': value_count,
            'max_length': max_length,
            'data_type_category': data_type
        }

    def _ensure_critical_field_placement(
        self,
        parent_fields: List[str],
        variant_fields: List[str],
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """Ensure critical fields are placed in appropriate levels.
        
        Args:
            parent_fields: Current parent field assignments
            variant_fields: Current variant field assignments
            mandatory_fields: All mandatory field data
            
        Returns:
            Adjusted field assignments
        """
        # Critical parent fields that should never be variants
        critical_parent_keywords = ['feed_product_type', 'brand_name', 'manufacturer']
        
        # Critical variant fields that should never be parent
        critical_variant_keywords = ['item_sku', 'color_name', 'size_name']
        
        # Move critical parent fields
        for field_name in list(variant_fields):
            if any(keyword in field_name.lower() for keyword in critical_parent_keywords):
                variant_fields.remove(field_name)
                if field_name not in parent_fields:
                    parent_fields.append(field_name)
                    self.logger.debug(f"Moved {field_name} to parent fields (critical)")
        
        # Move critical variant fields  
        for field_name in list(parent_fields):
            if any(keyword in field_name.lower() for keyword in critical_variant_keywords):
                parent_fields.remove(field_name)
                if field_name not in variant_fields:
                    variant_fields.append(field_name)
                    self.logger.debug(f"Moved {field_name} to variant fields (critical)")
        
        return parent_fields, variant_fields

    def _determines_child_inheritance(self, field_data: Dict[str, Any]) -> bool:
        """Determine if parent field value should be inherited by children.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Whether field determines child inheritance
        """
        # Fields with limited values typically define inheritance
        value_count = field_data.get('constraints', {}).get('value_count', 0)
        return value_count <= 5 and value_count > 0

    def _create_validation_rules(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation rules for field.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Validation rules
        """
        constraints = field_data.get('constraints', {})
        data_type = field_data.get('data_type', 'string')
        
        rules = {
            'required': self._is_required_field(field_data),
            'data_type': data_type
        }
        
        if constraints.get('max_length'):
            rules['max_length'] = constraints['max_length']
        
        if field_data.get('valid_values'):
            rules['allowed_values'] = field_data['valid_values']
        
        return rules

    def _is_required_field(self, field_data: Dict[str, Any]) -> bool:
        """Determine if field is required.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Whether field is required
        """
        # Fields with specific valid values are typically required
        valid_values = field_data.get('valid_values', [])
        return len(valid_values) > 0 and len(valid_values) < 20

    def _determine_variation_type(self, field_data: Dict[str, Any]) -> str:
        """Determine type of variation for variant field.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Variation type string
        """
        field_name = field_data.get('display_name', '').lower()
        
        if 'color' in field_name or 'farbe' in field_name:
            return 'color'
        elif 'size' in field_name or 'größe' in field_name:
            return 'size' 
        elif 'material' in field_name:
            return 'material'
        else:
            return 'attribute'

    def _is_variable_field(self, field_data: Dict[str, Any]) -> bool:
        """Check if field varies between product variants.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Whether field is variable
        """
        # Fields with many possible values typically vary
        value_count = field_data.get('constraints', {}).get('value_count', 0)
        return value_count > 5 or value_count == 0

    def _can_inherit_from_parent(self, field_data: Dict[str, Any]) -> bool:
        """Check if field can inherit value from parent.
        
        Args:
            field_data: Field data dictionary
            
        Returns:
            Whether field can inherit from parent
        """
        # Most fields can potentially inherit, except unique identifiers
        field_name = field_data.get('display_name', '').lower()
        unique_indicators = ['sku', 'id', 'identifier', 'number']
        
        return not any(indicator in field_name for indicator in unique_indicators)

    def _create_field_relationships(
        self,
        parent_fields: List[str],
        variant_fields: List[str],
        mandatory_fields: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create field relationship definitions.
        
        Args:
            parent_fields: Parent field names
            variant_fields: Variant field names
            mandatory_fields: All mandatory field data
            
        Returns:
            Field relationships structure
        """
        parent_defines = []
        variant_overrides = []
        shared_constraints = {}
        
        # Parent fields that define product family
        for field_name in parent_fields:
            field_data = mandatory_fields[field_name]
            if self._determines_child_inheritance(field_data):
                parent_defines.append({
                    'field': field_name,
                    'inheritance_type': 'mandatory',
                    'override_allowed': False
                })
        
        # Variant fields that can override parent values
        for field_name in variant_fields:
            field_data = mandatory_fields[field_name]
            if self._can_inherit_from_parent(field_data):
                variant_overrides.append({
                    'field': field_name,
                    'default_source': 'parent',
                    'variation_required': self._is_variable_field(field_data)
                })
        
        # Shared constraints across field types
        for field_name, field_data in mandatory_fields.items():
            constraints = field_data.get('constraints', {})
            if constraints.get('max_length') and constraints['max_length'] > 50:
                shared_constraints[field_name] = {
                    'max_length': constraints['max_length'],
                    'applies_to': 'all_levels'
                }
        
        return {
            'parent_defines': parent_defines,
            'variant_overrides': variant_overrides,
            'shared_constraints': shared_constraints
        }

    def _calculate_quality_score(self, template_structure: Dict[str, Any]) -> float:
        """Calculate template quality score.
        
        Args:
            template_structure: Template structure
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Field distribution score
        parent_count = len(template_structure.get('parent_product', {}).get('fields', {}))
        variant_count = len(template_structure.get('child_variants', {}).get('fields', {}))
        total_fields = parent_count + variant_count
        
        if total_fields > 0:
            # Ideal ratio is around 30-70% parent fields
            parent_ratio = parent_count / total_fields
            if 0.3 <= parent_ratio <= 0.7:
                score += 0.4
            else:
                score += 0.2
        
        # Relationship complexity score
        relationships = template_structure.get('field_relationships', {})
        if relationships.get('parent_defines'):
            score += 0.3
        if relationships.get('variant_overrides'):
            score += 0.2
        if relationships.get('shared_constraints'):
            score += 0.1
        
        return min(score, 1.0)

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