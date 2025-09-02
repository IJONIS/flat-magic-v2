"""Validation prompts for AI result verification.

This module provides prompts for validating and verifying
AI results throughout the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base_prompt import BasePromptManager


class ValidationPromptManager(BasePromptManager):
    """Manages prompts for AI result validation.
    
    Provides specialized prompts for validating AI outputs
    and ensuring quality and consistency.
    """
    
    def get_system_prompt(self) -> str:
        """Get system prompt for validation operations.
        
        Returns:
            System prompt for AI validation
        """
        return (
            "You are an expert AI result validator. "
            "Your task is to analyze AI outputs for quality, "
            "consistency, and adherence to requirements."
        )
    
    def create_mapping_validation_prompt(
        self,
        mapping_result: Dict[str, Any],
        mandatory_fields: Dict[str, Any],
        parent_sku: str
    ) -> str:
        """Create prompt for validating mapping results.
        
        Args:
            mapping_result: AI mapping result to validate
            mandatory_fields: Original mandatory fields
            parent_sku: Parent SKU identifier
            
        Returns:
            Mapping validation prompt
        """
        prompt = f"""{self.get_system_prompt()}

TASK: Validate this AI mapping result for parent SKU {parent_sku}.

MAPPING RESULT TO VALIDATE:
{self.format_json_data(mapping_result)}

ORIGINAL MANDATORY FIELDS:
{self.format_json_data(self.limit_data_size(mandatory_fields, 10))}
{self._get_mapping_validation_criteria()}
{self._get_validation_instructions()}
{self._get_validation_output_format()}
Validate and respond:"""
        
        return prompt
    
    def create_template_validation_prompt(
        self,
        template_structure: Dict[str, Any],
        validation_issues: List[str]
    ) -> str:
        """Create prompt for validating template structures.
        
        Args:
            template_structure: Template structure to validate
            validation_issues: Known validation issues
            
        Returns:
            Template validation prompt
        """
        issues_text = "\n- ".join(validation_issues) if validation_issues else "None identified"
        
        prompt = f"""{self.get_system_prompt()}

TASK: Validate and potentially improve this template structure.

TEMPLATE STRUCTURE:
{self.format_json_data(template_structure)}

KNOWN ISSUES:
- {issues_text}
{self._get_template_validation_criteria()}
{self._get_template_improvement_instructions()}
Provide validation assessment and any recommended improvements:"""
        
        return prompt
    
    def _get_mapping_validation_criteria(self) -> str:
        """Get mapping validation criteria.
        
        Returns:
            Formatted validation criteria
        """
        return """
VALIDATION CRITERIA:
- Are mapped values semantically correct for their target fields?
- Is the confidence score realistic (not too high or too low)?
- Are parent vs variant categorizations appropriate?
- Are unmapped mandatory fields truly unmappable from source data?
- Is the overall structure valid and complete?
- Are German language considerations properly handled?"""
    
    def _get_template_validation_criteria(self) -> str:
        """Get template validation criteria.
        
        Returns:
            Formatted template validation criteria
        """
        return """
TEMPLATE VALIDATION CRITERIA:
- Are parent fields truly shared across variants?
- Are variant fields appropriately differentiated?
- Is the field distribution balanced (not too many parent or variant fields)?
- Are field relationships logical and consistent?
- Do validation rules make sense for each field type?
- Is the template structure complete and usable?"""
    
    def _get_validation_instructions(self) -> str:
        """Get general validation instructions.
        
        Returns:
            Formatted instructions section
        """
        instructions = [
            "Analyze each component of the result systematically",
            "Check for logical consistency and semantic correctness",
            "Identify any obvious errors or inconsistencies",
            "Consider the German e-commerce context",
            "Provide specific feedback on issues found",
            "Suggest concrete improvements where needed"
        ]
        
        return self.create_instruction_section(instructions)
    
    def _get_template_improvement_instructions(self) -> str:
        """Get template improvement instructions.
        
        Returns:
            Formatted improvement instructions
        """
        instructions = [
            "Identify structural weaknesses or imbalances",
            "Check for missing or incorrectly categorized fields",
            "Validate relationship definitions",
            "Assess template usability for AI mapping",
            "Recommend specific structural improvements",
            "Maintain template integrity while fixing issues"
        ]
        
        return self.create_instruction_section(instructions, "IMPROVEMENT INSTRUCTIONS")
    
    def _get_validation_output_format(self) -> str:
        """Get validation output format.
        
        Returns:
            Formatted output format specification
        """
        format_example = {
            "validation_status": "passed",  # passed, failed, needs_review
            "quality_score": 0.85,
            "issues_found": [
                "Low confidence mapping for field X",
                "Variant field Y should be parent-level"
            ],
            "recommendations": [
                "Review mapping logic for semantic accuracy",
                "Recategorize field Y based on business logic"
            ],
            "overall_assessment": "Good mapping with minor improvements needed"
        }
        
        return self.create_output_format_section(format_example, "VALIDATION RESULT FORMAT")
    
    def create_confidence_validation_prompt(
        self,
        result_with_confidence: Dict[str, Any],
        confidence_threshold: float = 0.7
    ) -> str:
        """Create prompt for validating confidence scores.
        
        Args:
            result_with_confidence: Result containing confidence scores
            confidence_threshold: Minimum acceptable confidence
            
        Returns:
            Confidence validation prompt
        """
        prompt = f"""{self.get_system_prompt()}

TASK: Validate confidence scores in this AI result.

RESULT WITH CONFIDENCE SCORES:
{self.format_json_data(result_with_confidence)}

CONFIDENCE THRESHOLD: {confidence_threshold}

VALIDATION FOCUS:
- Are confidence scores realistic and well-calibrated?
- Do low-confidence items actually represent uncertainty?
- Are high-confidence items genuinely reliable?
- Is the overall confidence consistent with result quality?

Provide confidence validation assessment:
{self._get_validation_output_format()}"""
        
        return prompt