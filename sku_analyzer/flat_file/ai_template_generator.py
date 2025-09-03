"""AI-powered template generator with intelligent field categorization.

This module uses Gemini AI to intelligently categorize fields as parent vs variant
based on business logic and e-commerce patterns, replacing deterministic keyword matching.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..shared.gemini_client import GeminiClient as ModernGeminiClient
from ..step5_mapping.models import AIProcessingConfig
from .performance_monitor import PerformanceMonitor
from .validation_utils import ValidationUtils


class AITemplateGenerationError(Exception):
    """Raised when AI template generation fails."""
    pass


class AITemplateGenerator:
    """AI-powered generator for creating reusable parent-child structure templates."""

    def __init__(self, enable_performance_monitoring: bool = True) -> None:
        """Initialize AI template generator.
        
        Args:
            enable_performance_monitoring: Whether to enable performance tracking
        """
        self.logger = self._setup_logging()
        self.enable_monitoring = enable_performance_monitoring
        self.performance_monitor = PerformanceMonitor(enable_performance_monitoring)
        self.validation_utils = ValidationUtils()
        
        # Initialize AI client
        self.config = AIProcessingConfig()
        self.gemini_client = ModernGeminiClient(self.config)
        
        # Load field categorization prompt template
        self.categorization_prompt = self._load_categorization_prompt()

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

    def _load_categorization_prompt(self) -> str:
        """Load the AI field categorization prompt template.
        
        Returns:
            Prompt template string
        """
        prompt_path = Path(__file__).parent.parent / "ai_mapping" / "prompts" / "files" / "field_categorization_prompt.jinja2"
        
        try:
            with prompt_path.open('r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.warning(f"Categorization prompt not found at {prompt_path}, using fallback")
            return self._get_fallback_categorization_prompt()

    def _get_fallback_categorization_prompt(self) -> str:
        """Provide fallback prompt if template file is not available.
        
        Returns:
            Basic categorization prompt
        """
        return """Analyze the provided field definitions and categorize them as either PARENT fields 
        (shared across product family) or VARIANT fields (individual SKU level). Consider business logic,
        e-commerce patterns, and field semantics. Return structured JSON with categorization analysis."""

    async def generate_ai_template_from_mandatory_fields(
        self,
        step3_mandatory_path: str | Path,
        output_path: str | Path
    ) -> Dict[str, Any]:
        """Generate template using AI-powered field categorization.
        
        Args:
            step3_mandatory_path: Path to step3_mandatory_fields.json
            output_path: Path for step4_template.json output
            
        Returns:
            Generated template structure with AI analysis
            
        Raises:
            AITemplateGenerationError: When template generation fails
            FileNotFoundError: If input file doesn't exist
        """
        start_time = time.time()
        self.logger.info("Starting AI-powered template generation from mandatory fields")

        try:
            # Load mandatory fields data
            mandatory_fields = self._load_json_file(Path(step3_mandatory_path))
            
            if not mandatory_fields:
                raise AITemplateGenerationError("No mandatory fields found in input data")

            # Use AI to intelligently categorize fields
            ai_categorization = await self._ai_categorize_fields(mandatory_fields)
            
            # Extract parent and variant fields from AI analysis
            parent_fields, variant_fields = self._extract_field_lists_from_ai_analysis(ai_categorization)
            
            # Create template structure with AI insights
            template_structure = self.create_ai_enhanced_template_structure(
                parent_fields, variant_fields, mandatory_fields, ai_categorization
            )
            
            # Validate template quality with AI insights
            validation_result = self.validate_ai_template(template_structure, ai_categorization)
            
            # Create final template with AI metadata
            template_output = self._create_ai_template_output(
                template_structure, validation_result, ai_categorization, 
                Path(step3_mandatory_path).name
            )
            
            # Save template
            self._save_json_file(Path(output_path), template_output)
            
            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"AI template generation completed in {duration_ms:.2f}ms - "
                f"{len(parent_fields)} parent fields, {len(variant_fields)} variant fields "
                f"(Overall confidence: {ai_categorization.get('categorization_summary', {}).get('overall_confidence', 0):.2f})"
            )
            
            return template_output

        except Exception as e:
            self.logger.error(f"AI template generation failed: {e}")
            raise AITemplateGenerationError(f"AI template generation failed: {e}") from e

    async def _ai_categorize_fields(self, mandatory_fields: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Use AI to intelligently categorize fields as parent vs variant.
        
        Args:
            mandatory_fields: Dictionary of mandatory field data
            
        Returns:
            AI categorization analysis with confidence scores and reasoning
        """
        try:
            # Create AI prompt with field data
            prompt = f"""{self.categorization_prompt}

**MANDATORY FIELDS DATA TO ANALYZE**:
```json
{json.dumps(mandatory_fields, indent=2, ensure_ascii=False)}
```

Please analyze these fields and provide the structured categorization analysis as specified in the prompt."""

            # Get AI analysis
            self.logger.info("Requesting AI field categorization analysis...")
            response = await self.gemini_client.generate_content(prompt)
            
            # Parse and validate AI response
            ai_analysis = await self.gemini_client.validate_json_response(response)
            
            # Validate AI response structure
            if not self._validate_ai_categorization_response(ai_analysis):
                raise AITemplateGenerationError("AI categorization response missing required structure")
            
            self.logger.info(
                f"AI categorization completed - Overall confidence: {ai_analysis.get('categorization_summary', {}).get('overall_confidence', 0):.2f}"
            )
            
            return ai_analysis
            
        except Exception as e:
            self.logger.warning(f"AI categorization failed, falling back to rule-based: {e}")
            # Fallback to enhanced rule-based categorization
            return self._fallback_categorization(mandatory_fields)

    def _validate_ai_categorization_response(self, ai_response: Dict[str, Any]) -> bool:
        """Validate that AI response contains required structure.
        
        Args:
            ai_response: AI categorization response
            
        Returns:
            True if response structure is valid
        """
        required_keys = ['categorization_analysis', 'categorization_summary']
        
        if not all(key in ai_response for key in required_keys):
            return False
        
        analysis = ai_response['categorization_analysis']
        if not all(key in analysis for key in ['parent_fields', 'variant_fields']):
            return False
        
        # Check that each field has required properties
        for field_list in [analysis['parent_fields'], analysis['variant_fields']]:
            for field_data in field_list:
                if not all(key in field_data for key in ['field_name', 'confidence_score']):
                    return False
        
        return True

    def _extract_field_lists_from_ai_analysis(self, ai_categorization: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Extract parent and variant field lists from AI analysis.
        
        Args:
            ai_categorization: AI categorization analysis
            
        Returns:
            Tuple of (parent_fields, variant_fields) lists
        """
        analysis = ai_categorization['categorization_analysis']
        
        parent_fields = [field['field_name'] for field in analysis['parent_fields']]
        variant_fields = [field['field_name'] for field in analysis['variant_fields']]
        
        # Handle ambiguous fields based on AI recommendation
        if 'ambiguous_fields' in analysis:
            for ambiguous_field in analysis['ambiguous_fields']:
                recommendation = ambiguous_field.get('recommendation', 'variant')
                field_name = ambiguous_field['field_name']
                
                if recommendation == 'parent':
                    parent_fields.append(field_name)
                else:
                    variant_fields.append(field_name)
                
                self.logger.info(
                    f"Ambiguous field '{field_name}' assigned to {recommendation} level "
                    f"(confidence: {ambiguous_field.get('confidence_score', 0):.2f})"
                )
        
        return parent_fields, variant_fields

    def create_ai_enhanced_template_structure(
        self,
        parent_fields: List[str],
        variant_fields: List[str],
        mandatory_fields: Dict[str, Dict[str, Any]],
        ai_categorization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create enhanced template structure with AI insights.
        
        Args:
            parent_fields: List of parent-level field names
            variant_fields: List of variant-level field names  
            mandatory_fields: Complete mandatory fields data
            ai_categorization: AI categorization analysis
            
        Returns:
            Enhanced template structure dictionary
        """
        template_structure = {
            'parent_product': {
                'fields': {},
                'field_count': len(parent_fields),
                'required_fields': [],
                'ai_categorization_confidence': []
            },
            'child_variants': {
                'fields': {},
                'field_count': len(variant_fields),
                'variable_fields': [],
                'inherited_fields': [],
                'ai_categorization_confidence': []
            },
            'field_relationships': {
                'parent_defines': [],
                'variant_overrides': [],
                'shared_constraints': {},
                'ai_inheritance_patterns': []
            },
            'ai_analysis': {
                'categorization_method': 'ai_powered',
                'overall_confidence': ai_categorization.get('categorization_summary', {}).get('overall_confidence', 0),
                'quality_rating': ai_categorization.get('categorization_summary', {}).get('categorization_quality', 'medium'),
                'business_logic_validation': ai_categorization.get('categorization_summary', {}).get('business_logic_validation', {})
            }
        }
        
        # Process parent fields with AI confidence data
        analysis = ai_categorization['categorization_analysis']
        parent_confidence_map = {field['field_name']: field['confidence_score'] 
                               for field in analysis['parent_fields']}
        
        for field_name in parent_fields:
            field_data = mandatory_fields[field_name]
            confidence = parent_confidence_map.get(field_name, 0.5)
            
            template_structure['parent_product']['fields'][field_name] = {
                'display_name': field_data['display_name'],
                'data_type': field_data['data_type'],
                'constraints': field_data['constraints'],
                'applies_to_children': self._determines_child_inheritance(field_data),
                'validation_rules': self._create_validation_rules(field_data),
                'ai_confidence': confidence,
                'ai_reasoning': self._get_field_reasoning(field_name, analysis['parent_fields'])
            }
            
            template_structure['parent_product']['ai_categorization_confidence'].append({
                'field': field_name,
                'confidence': confidence
            })
            
            # Mark as required based on AI analysis and field characteristics
            if self._is_required_field(field_data):
                template_structure['parent_product']['required_fields'].append(field_name)
        
        # Process variant fields with AI confidence data
        variant_confidence_map = {field['field_name']: field['confidence_score'] 
                                for field in analysis['variant_fields']}
        
        for field_name in variant_fields:
            field_data = mandatory_fields[field_name]
            confidence = variant_confidence_map.get(field_name, 0.5)
            
            template_structure['child_variants']['fields'][field_name] = {
                'display_name': field_data['display_name'],
                'data_type': field_data['data_type'],
                'constraints': field_data['constraints'],
                'variation_type': self._determine_variation_type(field_data),
                'validation_rules': self._create_validation_rules(field_data),
                'ai_confidence': confidence,
                'ai_reasoning': self._get_field_reasoning(field_name, analysis['variant_fields'])
            }
            
            template_structure['child_variants']['ai_categorization_confidence'].append({
                'field': field_name,
                'confidence': confidence
            })
            
            # Categorize variant field behavior
            if self._is_variable_field(field_data):
                template_structure['child_variants']['variable_fields'].append(field_name)
            
            if self._can_inherit_from_parent(field_data):
                template_structure['child_variants']['inherited_fields'].append(field_name)
        
        # Create AI-enhanced field relationships
        template_structure['field_relationships'] = self._create_ai_enhanced_field_relationships(
            parent_fields, variant_fields, mandatory_fields, ai_categorization
        )
        
        return template_structure

    def _get_field_reasoning(self, field_name: str, field_list: List[Dict[str, Any]]) -> str:
        """Extract AI reasoning for field categorization.
        
        Args:
            field_name: Name of field to find reasoning for
            field_list: List of field categorization data from AI
            
        Returns:
            AI reasoning text or default message
        """
        for field_data in field_list:
            if field_data['field_name'] == field_name:
                return field_data.get('category_rationale', 'AI-based categorization')
        return 'AI-based categorization'

    def _create_ai_enhanced_field_relationships(
        self,
        parent_fields: List[str],
        variant_fields: List[str],
        mandatory_fields: Dict[str, Dict[str, Any]],
        ai_categorization: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create AI-enhanced field relationship definitions.
        
        Args:
            parent_fields: Parent field names
            variant_fields: Variant field names
            mandatory_fields: All mandatory field data
            ai_categorization: AI analysis with recommendations
            
        Returns:
            Enhanced field relationships structure
        """
        relationships = {
            'parent_defines': [],
            'variant_overrides': [],
            'shared_constraints': {},
            'ai_inheritance_patterns': []
        }
        
        # Use AI template recommendations if available
        template_recs = ai_categorization.get('template_recommendations', {})
        
        if 'inheritance_patterns' in template_recs:
            relationships['ai_inheritance_patterns'] = template_recs['inheritance_patterns']
        
        # Create parent-defined relationships
        for field_name in parent_fields:
            field_data = mandatory_fields[field_name]
            if self._determines_child_inheritance(field_data):
                relationships['parent_defines'].append({
                    'field': field_name,
                    'inheritance_type': 'mandatory',
                    'override_allowed': False,
                    'ai_confidence': self._get_field_confidence(field_name, ai_categorization, 'parent')
                })
        
        # Create variant override patterns
        for field_name in variant_fields:
            field_data = mandatory_fields[field_name]
            if self._can_inherit_from_parent(field_data):
                relationships['variant_overrides'].append({
                    'field': field_name,
                    'default_source': 'parent',
                    'variation_required': self._is_variable_field(field_data),
                    'ai_confidence': self._get_field_confidence(field_name, ai_categorization, 'variant')
                })
        
        # Create shared constraints with AI insights
        for field_name, field_data in mandatory_fields.items():
            constraints = field_data.get('constraints', {})
            if constraints.get('max_length') and constraints['max_length'] > 50:
                relationships['shared_constraints'][field_name] = {
                    'max_length': constraints['max_length'],
                    'applies_to': 'all_levels',
                    'ai_validated': True
                }
        
        return relationships

    def _get_field_confidence(self, field_name: str, ai_categorization: Dict[str, Any], category: str) -> float:
        """Get AI confidence score for specific field categorization.
        
        Args:
            field_name: Field name to look up
            ai_categorization: AI analysis data
            category: 'parent' or 'variant'
            
        Returns:
            Confidence score (0.0-1.0)
        """
        analysis = ai_categorization.get('categorization_analysis', {})
        field_list = analysis.get(f'{category}_fields', [])
        
        for field_data in field_list:
            if field_data['field_name'] == field_name:
                return field_data.get('confidence_score', 0.5)
        
        return 0.5  # Default for fallback cases

    def validate_ai_template(self, template_structure: Dict[str, Any], ai_categorization: Dict[str, Any]) -> Dict[str, Any]:
        """Validate AI-generated template structure and quality.
        
        Args:
            template_structure: Generated template structure
            ai_categorization: AI categorization analysis
            
        Returns:
            Enhanced validation result with AI insights
        """
        issues = []
        warnings = []
        
        # Standard validation checks
        parent_fields = template_structure.get('parent_product', {}).get('fields', {})
        variant_fields = template_structure.get('child_variants', {}).get('fields', {})
        
        if not parent_fields:
            issues.append("No parent fields defined")
        elif len(parent_fields) < 2:
            warnings.append("Very few parent fields - may not provide sufficient structure")
        
        if not variant_fields:
            warnings.append("No variant fields defined - all products will be identical")
        
        # AI-specific validation checks
        ai_summary = ai_categorization.get('categorization_summary', {})
        overall_confidence = ai_summary.get('overall_confidence', 0)
        
        if overall_confidence < 0.7:
            warnings.append(f"Low AI categorization confidence ({overall_confidence:.2f}) - review recommendations")
        
        # Check for low-confidence fields
        low_confidence_fields = []
        for category in ['parent_product', 'child_variants']:
            for field_data in template_structure[category]['fields'].values():
                if field_data.get('ai_confidence', 1.0) < 0.6:
                    low_confidence_fields.append(field_data.get('display_name', 'unknown'))
        
        if low_confidence_fields:
            warnings.append(f"Low confidence fields detected: {', '.join(low_confidence_fields)}")
        
        # Field balance validation with AI insights
        parent_count = len(parent_fields)
        variant_count = len(variant_fields)
        total_fields = parent_count + variant_count
        
        if total_fields > 0:
            parent_ratio = parent_count / total_fields
            business_validation = ai_summary.get('business_logic_validation', {})
            
            if not business_validation.get('ecommerce_best_practices', True):
                warnings.append("AI detected potential violations of e-commerce best practices")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'quality_score': self._calculate_ai_quality_score(template_structure, ai_categorization),
            'ai_confidence': overall_confidence,
            'ai_quality_rating': ai_summary.get('categorization_quality', 'medium'),
            'field_distribution': {
                'parent_fields': parent_count,
                'variant_fields': variant_count,
                'parent_ratio': parent_count / total_fields if total_fields > 0 else 0
            }
        }

    def _calculate_ai_quality_score(
        self, 
        template_structure: Dict[str, Any], 
        ai_categorization: Dict[str, Any]
    ) -> float:
        """Calculate AI-enhanced template quality score.
        
        Args:
            template_structure: Template structure
            ai_categorization: AI analysis data
            
        Returns:
            Quality score between 0 and 1
        """
        base_score = 0.0
        
        # AI confidence contribution (40% of score)
        ai_confidence = ai_categorization.get('categorization_summary', {}).get('overall_confidence', 0)
        base_score += ai_confidence * 0.4
        
        # Field distribution score (30% of score)
        parent_count = len(template_structure.get('parent_product', {}).get('fields', {}))
        variant_count = len(template_structure.get('child_variants', {}).get('fields', {}))
        total_fields = parent_count + variant_count
        
        if total_fields > 0:
            parent_ratio = parent_count / total_fields
            if 0.3 <= parent_ratio <= 0.7:  # Ideal ratio
                base_score += 0.3
            else:
                base_score += 0.15
        
        # Business logic validation (30% of score)
        business_validation = ai_categorization.get('categorization_summary', {}).get('business_logic_validation', {})
        if business_validation.get('ecommerce_best_practices', False):
            base_score += 0.15
        if business_validation.get('critical_fields_placed', False):
            base_score += 0.15
        
        return min(base_score, 1.0)

    def _fallback_categorization(self, mandatory_fields: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Provide fallback rule-based categorization when AI fails.
        
        Args:
            mandatory_fields: Field definitions
            
        Returns:
            Fallback categorization analysis
        """
        self.logger.info("Using enhanced rule-based fallback categorization")
        
        parent_fields = []
        variant_fields = []
        
        # Enhanced rule-based categorization
        for field_name, field_data in mandatory_fields.items():
            display_name = field_data.get('display_name', '').lower()
            technical_name = field_name.lower()
            
            # Enhanced parent indicators
            parent_indicators = {
                'brand', 'manufacturer', 'category', 'product_type', 'family', 'material',
                'country_of_origin', 'department', 'gender', 'age_range'
            }
            
            # Enhanced variant indicators  
            variant_indicators = {
                'size', 'color', 'sku', 'id', 'external', 'item_name', 'variant',
                'dimension', 'weight', 'price'
            }
            
            is_parent = any(indicator in technical_name or indicator in display_name
                          for indicator in parent_indicators)
            is_variant = any(indicator in technical_name or indicator in display_name  
                           for indicator in variant_indicators)
            
            if is_parent and not is_variant:
                parent_fields.append({
                    'field_name': field_name,
                    'display_name': field_data['display_name'],
                    'category_rationale': 'Rule-based: Contains parent-level indicators',
                    'confidence_score': 0.8
                })
            elif is_variant and not is_parent:
                variant_fields.append({
                    'field_name': field_name,
                    'display_name': field_data['display_name'], 
                    'category_rationale': 'Rule-based: Contains variant-level indicators',
                    'confidence_score': 0.8
                })
            else:
                # Default to variant for ambiguous cases
                variant_fields.append({
                    'field_name': field_name,
                    'display_name': field_data['display_name'],
                    'category_rationale': 'Rule-based: Default variant assignment',
                    'confidence_score': 0.5
                })
        
        return {
            'categorization_analysis': {
                'parent_fields': parent_fields,
                'variant_fields': variant_fields,
                'ambiguous_fields': []
            },
            'categorization_summary': {
                'total_fields_analyzed': len(mandatory_fields),
                'parent_fields_count': len(parent_fields),
                'variant_fields_count': len(variant_fields),
                'ambiguous_fields_count': 0,
                'overall_confidence': 0.7,
                'categorization_quality': 'medium',
                'business_logic_validation': {
                    'ecommerce_best_practices': True,
                    'critical_fields_placed': True
                }
            },
            'template_recommendations': {},
            'quality_assurance': {
                'critical_validations': [],
                'potential_issues': ['AI categorization failed - using rule-based fallback'],
                'optimization_suggestions': ['Consider reviewing field categorizations manually']
            }
        }

    def _create_ai_template_output(
        self,
        template_structure: Dict[str, Any],
        validation_result: Dict[str, Any],
        ai_categorization: Dict[str, Any],
        source_file: str
    ) -> Dict[str, Any]:
        """Create final AI-enhanced template output with comprehensive metadata.
        
        Args:
            template_structure: Generated template structure
            validation_result: Validation results
            ai_categorization: AI categorization analysis
            source_file: Source file name
            
        Returns:
            Complete AI-enhanced template output
        """
        return {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'source_file': source_file,
                'template_version': '2.0_ai_powered',
                'generation_method': 'ai_categorization',
                'field_distribution': validation_result['field_distribution'],
                'quality_score': validation_result['quality_score'],
                'ai_confidence': validation_result['ai_confidence'],
                'ai_quality_rating': validation_result['ai_quality_rating'],
                'validation_status': 'valid' if validation_result['valid'] else 'issues_found',
                'warnings': validation_result.get('warnings', [])
            },
            'template_structure': template_structure,
            'ai_analysis': ai_categorization,
            'usage_instructions': {
                'description': 'AI-powered template for structured parent-child product mapping',
                'parent_product_usage': 'Define shared characteristics and product family (AI-validated)',
                'child_variants_usage': 'Define variable attributes and specific variants (AI-categorized)',
                'inheritance_rules': 'Children inherit parent values unless overridden (AI-optimized)',
                'ai_insights': 'Template generated using business logic analysis and e-commerce patterns'
            },
            'quality_assurance': ai_categorization.get('quality_assurance', {})
        }

    # Utility methods (reuse from original template generator)
    def _load_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Load and validate JSON file."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")

    def _save_json_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Save data to JSON file with formatting."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _determines_child_inheritance(self, field_data: Dict[str, Any]) -> bool:
        """Determine if parent field value should be inherited by children."""
        value_count = field_data.get('constraints', {}).get('value_count', 0)
        return value_count <= 5 and value_count > 0

    def _create_validation_rules(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation rules for field."""
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
        """Determine if field is required."""
        valid_values = field_data.get('valid_values', [])
        return len(valid_values) > 0 and len(valid_values) < 20

    def _determine_variation_type(self, field_data: Dict[str, Any]) -> str:
        """Determine type of variation for variant field."""
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
        """Check if field varies between product variants."""
        value_count = field_data.get('constraints', {}).get('value_count', 0)
        return value_count > 5 or value_count == 0

    def _can_inherit_from_parent(self, field_data: Dict[str, Any]) -> bool:
        """Check if field can inherit value from parent."""
        field_name = field_data.get('display_name', '').lower()
        unique_indicators = ['sku', 'id', 'identifier', 'number']
        return not any(indicator in field_name for indicator in unique_indicators)