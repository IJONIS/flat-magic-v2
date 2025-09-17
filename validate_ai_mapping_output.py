"""AI Mapping Output Validation Script

This script provides automated validation of step5_ai_mapping.json outputs
against template requirements and quality standards.

Features:
- Validates field mapping completeness
- Checks data structure compliance  
- Measures mapping quality and confidence
- Identifies missing mandatory fields
- Performance and efficiency analysis
- Clear pass/fail criteria for production readiness

Usage:
    python validate_ai_mapping_output.py [parent_sku] [--all] [--report]
    
Examples:
    python validate_ai_mapping_output.py 4307
    python validate_ai_mapping_output.py --all
    python validate_ai_mapping_output.py 4307 --report
"""

import json
import logging
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ValidationResult:
    """Results of AI mapping validation."""
    parent_sku: str
    success: bool
    confidence: float
    total_variants: int
    mapped_parent_fields: int
    mapped_variant_fields: int
    required_parent_fields: int
    required_variant_fields: int
    missing_required_fields: List[str]
    data_quality_score: float
    performance_score: float
    overall_score: float
    issues: List[str]
    recommendations: List[str]
    processing_time_ms: Optional[float] = None
    safety_blocked: bool = False
    fallback_used: bool = False
    ai_mapping_failed: bool = False


class AIMappingValidator:
    """Comprehensive validator for AI mapping outputs."""
    
    def __init__(self, output_base_dir: str = None):
        """Initialize validator with output directory."""
        if output_base_dir is None:
            output_base_dir = "/Users/jaminmahmood/Desktop/Flat Magic v6/production_output"
        
        self.output_base_dir = Path(output_base_dir)
        self.logger = self._setup_logging()
        
        # Find most recent job directory
        self.current_job_dir = self._find_current_job_dir()
        if not self.current_job_dir:
            raise RuntimeError("No production output directories found")
        
        self.logger.info(f"Using job directory: {self.current_job_dir}")
        
        # Load template for validation
        self.template_data = self._load_template()
        self.mandatory_fields = self._extract_mandatory_fields()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _find_current_job_dir(self) -> Optional[Path]:
        """Find the most recent job directory."""
        if not self.output_base_dir.exists():
            return None
        
        job_dirs = [d for d in self.output_base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not job_dirs:
            return None
        
        # Return most recent (highest timestamp)
        return max(job_dirs, key=lambda x: int(x.name))
    
    def _load_template(self) -> Dict[str, Any]:
        """Load the template structure for validation."""
        template_file = self.current_job_dir / "flat_file_analysis" / "step4_template.json"
        
        if not template_file.exists():
            self.logger.warning(f"Template file not found: {template_file}")
            return {}
        
        with open(template_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _extract_mandatory_fields(self) -> Dict[str, str]:
        """Extract mandatory fields from template structure."""
        mandatory_fields = {}
        template_structure = self.template_data.get("template_structure", {})
        
        # Parent fields
        parent_fields = template_structure.get("parent_product", {}).get("fields", {})
        for field_name, config in parent_fields.items():
            if config.get("validation_rules", {}).get("required", False):
                mandatory_fields[field_name] = "parent"
        
        # Variant fields
        variant_fields = template_structure.get("variant_products", {}).get("fields", {})  
        for field_name, config in variant_fields.items():
            if config.get("validation_rules", {}).get("required", False):
                mandatory_fields[field_name] = "variant"
        
        self.logger.info(f"Found {len(mandatory_fields)} mandatory fields")
        return mandatory_fields
    
    def validate_parent_output(self, parent_sku: str) -> ValidationResult:
        """Validate AI mapping output for a specific parent SKU."""
        parent_dir = self.current_job_dir / f"parent_{parent_sku}"
        output_file = parent_dir / "step5_ai_mapping.json"
        
        if not output_file.exists():
            return ValidationResult(
                parent_sku=parent_sku,
                success=False,
                confidence=0.0,
                total_variants=0,
                mapped_parent_fields=0,
                mapped_variant_fields=0,
                required_parent_fields=len([f for f, scope in self.mandatory_fields.items() if scope == "parent"]),
                required_variant_fields=len([f for f, scope in self.mandatory_fields.items() if scope == "variant"]),
                missing_required_fields=list(self.mandatory_fields.keys()),
                data_quality_score=0.0,
                performance_score=0.0,
                overall_score=0.0,
                issues=[f"Output file not found: {output_file}"],
                recommendations=["Run step5 mapping for this parent"]
            )
        
        # Load mapping output
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        return self._analyze_mapping_output(parent_sku, output_data)
    
    def _analyze_mapping_output(self, parent_sku: str, output_data: Dict[str, Any]) -> ValidationResult:
        """Analyze the mapping output data for quality and completeness."""
        issues = []
        recommendations = []
        
        # Extract basic information
        parent_data = output_data.get("parent_data", {})
        variants = output_data.get("variants", [])
        metadata = output_data.get("metadata", {})
        
        # Basic structure validation
        if not isinstance(parent_data, dict):
            issues.append("Invalid parent_data structure")
            parent_data = {}
        
        if not isinstance(variants, list):
            issues.append("Invalid variants structure")
            variants = []
        
        if not isinstance(metadata, dict):
            issues.append("Invalid metadata structure") 
            metadata = {}
        
        # Extract metadata information
        confidence = float(metadata.get("mapping_confidence", metadata.get("confidence", 0.0)))
        total_variants = len(variants)
        safety_blocked = metadata.get("safety_blocked", False)
        fallback_used = bool(metadata.get("fallback_strategy"))
        ai_mapping_failed = metadata.get("ai_mapping_failed", False)
        
        # Field mapping analysis
        mapped_parent_fields = list(parent_data.keys())
        mapped_variant_fields = set()
        
        # Extract variant fields from all variants
        for variant_item in variants:
            if isinstance(variant_item, dict):
                for variant_key, variant_data in variant_item.items():
                    if isinstance(variant_data, dict):
                        mapped_variant_fields.update(variant_data.keys())
        
        mapped_variant_fields = list(mapped_variant_fields)
        
        # Required fields analysis
        required_parent_fields = [f for f, scope in self.mandatory_fields.items() if scope == "parent"]
        required_variant_fields = [f for f, scope in self.mandatory_fields.items() if scope == "variant"]
        
        missing_required_fields = []
        for field_name, scope in self.mandatory_fields.items():
            if scope == "parent" and field_name not in mapped_parent_fields:
                missing_required_fields.append(field_name)
            elif scope == "variant" and field_name not in mapped_variant_fields:
                missing_required_fields.append(field_name)
        
        # Data quality scoring
        data_quality_score = self._calculate_data_quality_score(
            parent_data, variants, mapped_parent_fields, mapped_variant_fields,
            required_parent_fields, required_variant_fields
        )
        
        # Performance scoring
        performance_score = self._calculate_performance_score(metadata, confidence, total_variants)
        
        # Overall scoring
        overall_score = (data_quality_score * 0.6 + performance_score * 0.4)
        
        # Generate issues and recommendations
        if confidence < 0.8:
            issues.append(f"Low confidence score: {confidence:.2f}")
            recommendations.append("Review input data quality and template alignment")
        
        if safety_blocked:
            issues.append("Safety filter blocked original request")
            if fallback_used:
                recommendations.append("Fallback strategy used - review content sanitization")
            else:
                recommendations.append("Implement improved safety filter handling")
        
        if len(missing_required_fields) > len(self.mandatory_fields) * 0.5:
            issues.append(f"Many required fields missing: {len(missing_required_fields)}/{len(self.mandatory_fields)}")
            recommendations.append("Improve field mapping logic or template alignment")
        
        if total_variants == 0:
            issues.append("No variants processed")
            recommendations.append("Check input data has variant information")
        elif total_variants < 10:
            recommendations.append("Low variant count - verify data completeness")
        
        # Success determination
        success = (
            confidence >= 0.7 and
            total_variants > 0 and
            len(mapped_parent_fields) > 0 and
            len(missing_required_fields) < len(self.mandatory_fields) * 0.7
        )
        
        return ValidationResult(
            parent_sku=parent_sku,
            success=success,
            confidence=confidence,
            total_variants=total_variants,
            mapped_parent_fields=len(mapped_parent_fields),
            mapped_variant_fields=len(mapped_variant_fields),
            required_parent_fields=len(required_parent_fields),
            required_variant_fields=len(required_variant_fields),
            missing_required_fields=missing_required_fields,
            data_quality_score=data_quality_score,
            performance_score=performance_score,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations,
            safety_blocked=safety_blocked,
            fallback_used=fallback_used,
            ai_mapping_failed=ai_mapping_failed
        )
    
    def _calculate_data_quality_score(
        self, 
        parent_data: Dict[str, Any], 
        variants: List[Any],
        mapped_parent_fields: List[str],
        mapped_variant_fields: List[str],
        required_parent_fields: List[str],
        required_variant_fields: List[str]
    ) -> float:
        """Calculate data quality score based on field mapping completeness."""
        score_components = []
        
        # Parent field completeness (40% of score)
        if required_parent_fields:
            parent_completeness = len([f for f in required_parent_fields if f in mapped_parent_fields]) / len(required_parent_fields)
            score_components.append(parent_completeness * 0.4)
        
        # Variant field completeness (30% of score)
        if required_variant_fields:
            variant_completeness = len([f for f in required_variant_fields if f in mapped_variant_fields]) / len(required_variant_fields)
            score_components.append(variant_completeness * 0.3)
        
        # Data richness (20% of score)
        total_mapped_fields = len(set(mapped_parent_fields + mapped_variant_fields))
        total_possible_fields = len(self.mandatory_fields)
        if total_possible_fields > 0:
            richness_score = min(1.0, total_mapped_fields / total_possible_fields)
            score_components.append(richness_score * 0.2)
        
        # Variant coverage (10% of score)
        if variants:
            # Check how many variants have key fields
            variants_with_data = 0
            for variant_item in variants:
                if isinstance(variant_item, dict):
                    for variant_data in variant_item.values():
                        if isinstance(variant_data, dict) and len(variant_data) > 0:
                            variants_with_data += 1
                            break
            
            variant_coverage = variants_with_data / len(variants) if variants else 0
            score_components.append(variant_coverage * 0.1)
        
        return sum(score_components) if score_components else 0.0
    
    def _calculate_performance_score(self, metadata: Dict[str, Any], confidence: float, total_variants: int) -> float:
        """Calculate performance score based on efficiency metrics."""
        score = 0.0
        
        # Confidence score (50% of performance)
        score += confidence * 0.5
        
        # Processing efficiency (25% of performance)
        if total_variants > 0:
            # Higher scores for processing more variants
            if total_variants >= 50:
                score += 0.25
            elif total_variants >= 20:
                score += 0.2
            elif total_variants >= 10:
                score += 0.15
            else:
                score += 0.1
        
        # Safety and reliability (25% of performance)
        if metadata.get("safety_blocked", False):
            if metadata.get("fallback_strategy"):
                score += 0.2  # Good fallback handling
            else:
                score += 0.1  # Safety blocked but handled
        else:
            score += 0.25  # No safety issues
        
        return min(1.0, score)
    
    def validate_all_parents(self) -> Dict[str, ValidationResult]:
        """Validate all parent directories in the current job."""
        results = {}
        
        parent_dirs = list(self.current_job_dir.glob("parent_*"))
        self.logger.info(f"Found {len(parent_dirs)} parent directories to validate")
        
        for parent_dir in parent_dirs:
            parent_sku = parent_dir.name.replace("parent_", "")
            self.logger.info(f"Validating parent {parent_sku}")
            
            result = self.validate_parent_output(parent_sku)
            results[parent_sku] = result
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult]) -> str:
        """Generate a comprehensive validation report."""
        if not results:
            return "No validation results available."
        
        # Calculate summary statistics
        total_parents = len(results)
        successful_parents = sum(1 for r in results.values() if r.success)
        avg_confidence = sum(r.confidence for r in results.values()) / total_parents
        avg_overall_score = sum(r.overall_score for r in results.values()) / total_parents
        total_variants = sum(r.total_variants for r in results.values())
        safety_blocked_count = sum(1 for r in results.values() if r.safety_blocked)
        fallback_used_count = sum(1 for r in results.values() if r.fallback_used)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
AI MAPPING VALIDATION REPORT
============================
Generated: {timestamp}
Job Directory: {self.current_job_dir}

SUMMARY STATISTICS
------------------
Total Parents Processed: {total_parents}
Successful Mappings: {successful_parents} ({successful_parents/total_parents*100:.1f}%)
Average Confidence: {avg_confidence:.3f}
Average Overall Score: {avg_overall_score:.3f}
Total Variants Processed: {total_variants}
Safety Filter Issues: {safety_blocked_count} parents
Fallback Strategies Used: {fallback_used_count} parents

QUALITY THRESHOLDS
------------------
✅ Success Rate: {successful_parents/total_parents*100:.1f}% {'(PASS)' if successful_parents/total_parents >= 0.95 else '(FAIL - Target: 95%)'}
✅ Average Confidence: {avg_confidence:.3f} {'(PASS)' if avg_confidence >= 0.8 else '(FAIL - Target: 0.8)'}  
✅ Safety Handling: {(total_parents-safety_blocked_count)/total_parents*100:.1f}% {'(PASS)' if safety_blocked_count/total_parents <= 0.1 else '(FAIL - Max: 10%)'}

INDIVIDUAL RESULTS
------------------
"""
        
        # Add individual parent results
        for parent_sku, result in sorted(results.items()):
            status = "✅ PASS" if result.success else "❌ FAIL"
            
            report += f"""
Parent {parent_sku}: {status}
  Confidence: {result.confidence:.3f}
  Variants: {result.total_variants}
  Parent Fields: {result.mapped_parent_fields}/{result.required_parent_fields}
  Variant Fields: {result.mapped_variant_fields}/{result.required_variant_fields}
  Overall Score: {result.overall_score:.3f}
  Safety Blocked: {'Yes' if result.safety_blocked else 'No'}
  Fallback Used: {'Yes' if result.fallback_used else 'No'}
"""
            
            if result.issues:
                report += f"  Issues: {'; '.join(result.issues[:3])}\n"
            
            if result.missing_required_fields:
                missing_display = result.missing_required_fields[:5]
                if len(result.missing_required_fields) > 5:
                    missing_display.append(f"...+{len(result.missing_required_fields)-5} more")
                report += f"  Missing Fields: {', '.join(missing_display)}\n"
        
        # Production readiness assessment
        production_ready = (
            successful_parents / total_parents >= 0.95 and
            avg_confidence >= 0.8 and
            safety_blocked_count / total_parents <= 0.1
        )
        
        report += f"""

PRODUCTION READINESS ASSESSMENT
===============================
Status: {'🟢 PRODUCTION READY' if production_ready else '🟡 NEEDS ATTENTION'}

Key Metrics:
- Success Rate: {successful_parents/total_parents*100:.1f}% (Target: ≥95%)
- Avg Confidence: {avg_confidence:.3f} (Target: ≥0.8)
- Safety Issues: {safety_blocked_count/total_parents*100:.1f}% (Target: ≤10%)
- Total Variants: {total_variants}
- Fallback Success: {fallback_used_count}/{safety_blocked_count if safety_blocked_count > 0 else 1} safety issues handled

RECOMMENDATIONS
===============
"""
        
        # Generate recommendations
        if successful_parents / total_parents < 0.95:
            report += "- Improve overall success rate to reach 95% target\n"
        
        if avg_confidence < 0.8:
            report += "- Enhance confidence scores through better template alignment\n"
        
        if safety_blocked_count > total_parents * 0.1:
            report += "- Implement better content sanitization to reduce safety filter blocks\n"
        
        if fallback_used_count < safety_blocked_count:
            report += "- Improve fallback strategy reliability\n"
        
        # Add specific recommendations from individual results
        all_recommendations = set()
        for result in results.values():
            all_recommendations.update(result.recommendations)
        
        for rec in sorted(all_recommendations)[:5]:  # Top 5 recommendations
            report += f"- {rec}\n"
        
        return report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Validate AI mapping outputs")
    parser.add_argument("parent_sku", nargs='?', help="Parent SKU to validate (optional)")
    parser.add_argument("--all", action='store_true', help="Validate all parents")
    parser.add_argument("--report", action='store_true', help="Generate detailed report")
    parser.add_argument("--output-dir", help="Override output directory path")
    
    args = parser.parse_args()
    
    try:
        # Initialize validator
        validator = AIMappingValidator(args.output_dir)
        
        if args.all or (not args.parent_sku and not args.all):
            # Validate all parents
            print("Validating all parent directories...")
            results = validator.validate_all_parents()
            
            if args.report:
                report = validator.generate_validation_report(results)
                
                # Save report to file
                report_file = validator.current_job_dir / "validation_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                
                print(report)
                print(f"\nReport saved to: {report_file}")
            else:
                # Print summary
                total = len(results)
                successful = sum(1 for r in results.values() if r.success)
                avg_confidence = sum(r.confidence for r in results.values()) / total if total > 0 else 0
                
                print(f"\nVALIDATION SUMMARY:")
                print(f"Success Rate: {successful}/{total} ({successful/total*100:.1f}%)")
                print(f"Average Confidence: {avg_confidence:.3f}")
                print("Use --report flag for detailed analysis")
                
        else:
            # Validate specific parent
            result = validator.validate_parent_output(args.parent_sku)
            
            print(f"\nValidation Results for Parent {args.parent_sku}:")
            print(f"Success: {'✅' if result.success else '❌'}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Total Variants: {result.total_variants}")
            print(f"Overall Score: {result.overall_score:.3f}")
            print(f"Mapped Fields: {result.mapped_parent_fields + result.mapped_variant_fields}")
            
            if result.issues:
                print(f"\nIssues:")
                for issue in result.issues:
                    print(f"  - {issue}")
            
            if result.recommendations:
                print(f"\nRecommendations:")
                for rec in result.recommendations:
                    print(f"  - {rec}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()