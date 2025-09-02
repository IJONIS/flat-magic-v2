"""Integration point for AI mapping in existing workflow."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Union

from .data_transformer import DataVarianceAnalyzer
from .models import TransformationResult
from .example_loader import ExampleFormatLoader, FormatEnforcer


class AIMapingIntegration:
    """Integration point for AI mapping with enhanced transformation capabilities."""
    
    def __init__(self, enable_ai: bool = True):
        """Initialize AI mapping integration.
        
        Args:
            enable_ai: Whether to enable AI mapping capabilities
        """
        self.enable_ai = enable_ai
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.processing_stats = {
            "parents_processed": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0
        }
        
        # Initialize variance analyzer
        self.variance_analyzer = DataVarianceAnalyzer()
        
        # Initialize format validation system
        self.example_loader = ExampleFormatLoader()
        self.format_enforcer = FormatEnforcer(self.example_loader)
    
    def process_ai_mapping_step(
        self,
        output_dir: Union[str, Path],
        starting_parent: str = "4301"
    ) -> Dict[str, Any]:
        """Process AI mapping step for all parents.
        
        Args:
            output_dir: Base output directory
            starting_parent: Starting parent SKU
            
        Returns:
            Processing summary
        """
        start_time = time.time()
        self.logger.info("Starting AI mapping step with enhanced transformation")
        
        if not self.enable_ai:
            return {
                "ai_mapping_enabled": False,
                "message": "AI mapping disabled - placeholder functionality only"
            }
        
        try:
            # Convert to Path object if string
            output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
            
            # Find parent directories
            parent_dirs = self._find_parent_directories(output_dir)
            
            if not parent_dirs:
                return {
                    "success": False,
                    "error": "No parent directories found",
                    "searched_in": str(output_dir)
                }
            
            # Process starting parent first
            results = []
            if starting_parent in parent_dirs:
                result = self._process_single_parent(
                    starting_parent, output_dir, is_validation=True
                )
                results.append(result)
                
                # If successful, process remaining parents
                if result.get("success", False):
                    remaining_parents = [p for p in parent_dirs if p != starting_parent]
                    for parent_sku in remaining_parents:
                        result = self._process_single_parent(parent_sku, output_dir)
                        results.append(result)
            
            # Generate summary
            processing_time = time.time() - start_time
            return self._generate_summary(results, processing_time)
            
        except Exception as e:
            self.logger.error(f"AI mapping step failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _find_parent_directories(self, output_dir: Path) -> List[str]:
        """Find parent directories with required files."""
        parent_skus = []
        
        for parent_dir in output_dir.glob("parent_*"):
            if parent_dir.is_dir():
                # Check if required files exist
                step2_file = parent_dir / "step2_compressed.json"
                if step2_file.exists():
                    parent_sku = parent_dir.name.replace("parent_", "")
                    parent_skus.append(parent_sku)
        
        return parent_skus
    
    def _process_single_parent(
        self,
        parent_sku: str,
        output_dir: Path,
        is_validation: bool = False
    ) -> Dict[str, Any]:
        """Process single parent with enhanced transformation logic."""
        start_time = time.time()
        
        try:
            # Load required files
            step3_file = output_dir / "flat_file_analysis" / "step3_mandatory_fields.json"
            step2_file = output_dir / f"parent_{parent_sku}" / "step2_compressed.json"
            
            with step3_file.open('r') as f:
                mandatory_fields_data = json.load(f)
                mandatory_fields = mandatory_fields_data.get("mandatory_fields", {})
            
            with step2_file.open('r') as f:
                product_data = json.load(f)
            
            # Create enhanced transformation result
            transformation_result = self._create_enhanced_transformation(
                parent_sku, mandatory_fields, product_data
            )
            
            # Enforce format compliance
            compliant_result, format_warnings = self.format_enforcer.enforce_format(
                transformation_result, parent_sku, strict=False
            )
            
            if format_warnings:
                self.logger.warning(
                    f"Format warnings for {parent_sku}: {format_warnings}"
                )
            
            # Save result in compliant format
            output_file = output_dir / f"parent_{parent_sku}" / "step3_ai_mapping.json"
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(compliant_result, f, indent=2, ensure_ascii=False)
            
            # Update stats
            processing_time = time.time() - start_time
            self.processing_stats["parents_processed"] += 1
            self.processing_stats["successful_mappings"] += 1
            self.processing_stats["total_processing_time"] += processing_time
            
            confidence = compliant_result.get("metadata", {}).get("mapping_confidence", 0.0)
            current_avg = self.processing_stats["average_confidence"]
            total_processed = self.processing_stats["parents_processed"]
            self.processing_stats["average_confidence"] = (
                (current_avg * (total_processed - 1) + confidence) / total_processed
            )
            
            return {
                "success": True,
                "parent_sku": parent_sku,
                "mapped_fields_count": len(compliant_result.get("parent_data", {})),
                "unmapped_count": 0,  # Compliant format doesn't track unmapped fields
                "confidence": confidence,
                "total_variants": compliant_result.get("metadata", {}).get("total_variants", 0),
                "processing_time_ms": processing_time * 1000,
                "output_file": str(output_file),
                "is_validation": is_validation,
                "format_warnings": len(format_warnings),
                "format_compliant": True
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process parent {parent_sku}: {e}")
            self.processing_stats["failed_mappings"] += 1
            
            return {
                "success": False,
                "parent_sku": parent_sku,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    def _create_enhanced_transformation(
        self,
        parent_sku: str,
        mandatory_fields: Dict[str, Any],
        product_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create enhanced transformation using variance analysis and intelligent mapping."""
        from datetime import datetime
        
        # Use variance analyzer to understand data structure
        analysis = self.variance_analyzer.analyze_product_data(product_data)
        
        parent_data_source = product_data.get("parent_data", {})
        data_rows = product_data.get("data_rows", [])
        
        # Transform parent data using intelligent mapping rules  
        transformed_parent_data = self._transform_parent_data_enhanced(
            parent_data_source, mandatory_fields
        )
        
        # Validate constraints against mandatory fields
        self._validate_constraints(transformed_parent_data, mandatory_fields)
        
        # Create variance records from data rows
        variance_records = self._create_variance_records(
            data_rows, parent_sku, analysis
        )
        
        # Calculate confidence based on filled fields
        filled_parent_fields = sum(1 for v in transformed_parent_data.values() if v)
        confidence = min(0.95, filled_parent_fields / max(len(transformed_parent_data), 1))
        
        # Create result in target format structure
        return {
            "metadata": {
                "parent_id": parent_sku,
                "job_id": f"job_{parent_sku}_{int(time.time())}",
                "transformation_timestamp": datetime.utcnow().isoformat() + "Z",
                "ai_model": "enhanced-rule-based-v1",
                "mapping_confidence": confidence,
                "total_variants": len(variance_records)
            },
            "parent_data": transformed_parent_data,
            "variance_data": variance_records
        }
    
    def _transform_parent_data_enhanced(
        self,
        parent_data_source: Dict[str, Any],
        mandatory_fields: Dict[str, Any]
    ) -> Dict[str, str]:
        """Transform parent data to match target schema exactly."""
        
        # Target parent data schema from example - all 23 mandatory fields
        target_fields = {
            "feed_product_type": "",
            "brand_name": "", 
            "outer_material_type": "",
            "target_gender": "",
            "age_range_description": "",
            "bottoms_size_system": "",
            "bottoms_size_class": "",
            "country_of_origin": "",
            "department_name": "",
            "recommended_browse_nodes": "",
            "item_sku": "",
            "item_name": "",
            "external_product_id": "",
            "external_product_id_type": "",
            "standard_price": "",
            "quantity": "",
            "main_image_url": "",
            "color_map": "",
            "color_name": "",
            "size_name": "",
            "size_map": "",
            "fabric_type": "",
            "list_price_with_tax": ""
        }
        
        # Enhanced mapping rules for all fields
        mapping_rules = {
            "brand_name": ["MANUFACTURER_NAME", "BRAND", "HERSTELLER"],
            "feed_product_type": ["MANUFACTURER_TYPE_DESCRIPTION", "PRODUCT_TYPE"],
            "outer_material_type": ["MATERIAL", "MATERIAL_TYPE", "OBERMATERIAL", "FVALUE_3_5"],
            "target_gender": ["GENDER", "GESCHLECHT", "TARGET_GENDER"],
            "age_range_description": ["AGE_RANGE", "ALTERSGRUPPE"],
            "bottoms_size_system": ["SIZE_SYSTEM", "GRÖßENSYSTEM"],
            "bottoms_size_class": ["SIZE_CLASS", "GRÖßENKLASSE"],
            "country_of_origin": ["COUNTRY_OF_ORIGIN", "HERKUNFTSLAND"],
            "department_name": ["DEPARTMENT", "ABTEILUNG", "GROUP_STRING"],
            "recommended_browse_nodes": ["BROWSE_NODES", "KATEGORIE"],
            "item_sku": ["_parent_sku", "MASTER", "MANUFACTURER_PID"],
            "item_name": ["DESCRIPTION_LONG", "DESCRIPTION_SHORT"],
            "external_product_id": ["MANUFACTURER_PID", "INTERNATIONAL_PID"],
            "external_product_id_type": ["PRODUCT_ID_TYPE"],
            "fabric_type": ["FABRIC", "MATERIAL", "GEWEBE", "FVALUE_3_5"]
        }
        
        # Apply mappings
        for target_field, source_patterns in mapping_rules.items():
            if target_field not in target_fields:
                continue
                
            # Find matching source field
            for pattern in source_patterns:
                for source_field, source_value in parent_data_source.items():
                    if (pattern.lower() in source_field.upper() or 
                        pattern.upper() == source_field.upper()):
                        
                        if source_value and str(source_value).strip():
                            # Apply transformations
                            transformed_value = self._transform_field_value(
                                source_value, target_field, source_field
                            )
                            target_fields[target_field] = transformed_value
                            break
                if target_fields[target_field]:
                    break
        
        # Set defaults for missing required fields
        self._apply_field_defaults(target_fields, parent_data_source)
        
        return target_fields
    
    def _create_variance_records(
        self,
        data_rows: List[Dict[str, Any]], 
        parent_sku: str,
        analysis: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Create variance records matching target schema."""
        
        records = []
        
        for i, row in enumerate(data_rows):
            # Extract key variance fields from compressed data
            supplier_pid = row.get("SUPPLIER_PID", "")
            size_value = str(row.get("FVALUE_3_2", "")).strip()  # Size field
            color_value = str(row.get("FVALUE_3_3", "")).strip()  # Color field
            color_code = str(row.get("FVALUE_3_1", "")).strip()   # Color code field
            
            # Use SUPPLIER_PID if available, otherwise generate
            if supplier_pid:
                item_sku = supplier_pid
            elif size_value and color_value:
                item_sku = f"{parent_sku}_{color_code}_{size_value}"
            else:
                item_sku = f"{parent_sku}_var_{i+1}"
            
            # Map color name to English
            color_map = self._map_color_name(color_value) if color_value else ""
            
            record = {
                "item_sku": item_sku,
                "size_name": size_value,
                "color_name": color_value,
                "size_map": size_value,  # Direct mapping for now
                "color_map": color_map
            }
            
            records.append(record)
        
        return records
    
    def _map_color_name(self, color_value: str) -> str:
        """Map German color names to English equivalents."""
        
        color_mappings = {
            "schwarz": "Black",
            "weiss": "White", 
            "weiß": "White",
            "braun": "Brown",
            "blau": "Blue",
            "rot": "Red",
            "grün": "Green",
            "grau": "Gray",
            "gelb": "Yellow",
            "oliv": "Olive",
            "beige": "Beige",
            "navy": "Navy"
        }
        
        color_lower = color_value.lower().strip()
        return color_mappings.get(color_lower, color_value.title())
    
    def _validate_constraints(
        self,
        output: Dict[str, str],
        mandatory_fields: Dict[str, Any]
    ) -> None:
        """Validate AI output against mandatory field constraints."""
        validation_errors = []
        
        for field_name, value in output.items():
            if field_name in mandatory_fields:
                field_config = mandatory_fields[field_name]
                valid_values = field_config.get("valid_values", [])
                
                # Filter out description/help text from valid values
                actual_valid_values = []
                for val in valid_values:
                    # Skip long descriptions and help text
                    if len(val) < 50 and not ("\n" in val and len(val) > 100):
                        actual_valid_values.append(val.lower())
                
                # Only validate if we have actual constraints
                if actual_valid_values and value:
                    if value.lower() not in actual_valid_values:
                        # Try to find a close match
                        corrected_value = self._find_closest_valid_value(
                            value, actual_valid_values
                        )
                        if corrected_value:
                            output[field_name] = corrected_value
                        else:
                            validation_errors.append(
                                f"Field '{field_name}': '{value}' not in valid values {actual_valid_values}"
                            )
        
        if validation_errors:
            self.logger.warning(f"Constraint validation warnings: {validation_errors}")
    
    def _find_closest_valid_value(self, value: str, valid_values: List[str]) -> str:
        """Find closest matching valid value."""
        value_lower = value.lower()
        
        # Exact match
        if value_lower in valid_values:
            return value_lower
        
        # Partial match
        for valid_val in valid_values:
            if value_lower in valid_val or valid_val in value_lower:
                return valid_val
        
        return ""
    
    def _apply_field_defaults(
        self,
        target_fields: Dict[str, str],
        parent_data_source: Dict[str, Any]
    ) -> None:
        """Apply defaults for missing required fields."""
        
        # Set common defaults
        defaults = {
            "external_product_id_type": "EAN",
            "target_gender": "mens",
            "age_range_description": "adult",
            "bottoms_size_system": "EU",
            "bottoms_size_class": "regular",
            "department_name": "mens",
            "recommended_browse_nodes": "2945",
            "quantity": "1",
            "standard_price": "0.00",
            "list_price_with_tax": "0.00"
        }
        
        for field, default_value in defaults.items():
            if field in target_fields and not target_fields[field]:
                target_fields[field] = default_value
    
    def _transform_parent_data(
        self, 
        parent_data_source: Dict[str, Any], 
        mandatory_fields: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform parent data using intelligent mapping rules."""
        
        # Enhanced mapping rules with priority order
        mapping_rules = {
            "brand_name": ["MANUFACTURER_NAME"],
            "item_name": ["DESCRIPTION_LONG", "DESCRIPTION_SHORT"],
            "item_sku": ["_parent_sku", "MASTER"],
            "external_product_id": ["MANUFACTURER_PID", "INTERNATIONAL_PID"],
            "country_of_origin": ["COUNTRY_OF_ORIGIN"],
            "feed_product_type": ["MANUFACTURER_TYPE_DESCRIPTION"],
            "manufacturer": ["MANUFACTURER_NAME"],
            "weight": ["WEIGHT"],
            "keywords": ["KEYWORDS"]
        }
        
        transformed_data = {}
        
        for target_field, source_patterns in mapping_rules.items():
            if target_field not in mandatory_fields:
                continue
                
            # Find first matching source field with valid value
            for pattern in source_patterns:
                for source_field, source_value in parent_data_source.items():
                    if (pattern.lower() == source_field.lower() or 
                        pattern in source_field):
                        
                        if source_value and str(source_value).strip():
                            # Apply transformations
                            transformed_value = self._transform_field_value(
                                source_value, target_field, source_field
                            )
                            transformed_data[target_field] = transformed_value
                            break
                if target_field in transformed_data:
                    break
        
        return transformed_data
    
    def _extract_variance_data(
        self, 
        analysis: Dict[str, Any], 
        product_data: Dict[str, Any]
    ) -> Dict[str, List[Any]]:
        """Extract variance data using analysis results."""
        
        variance_data = {}
        field_mappings = analysis.get("field_mappings", {})
        data_rows = product_data.get("data_rows", [])
        
        # Use field mappings to extract variance values
        for source_field, target_field in field_mappings.items():
            unique_values = set()
            for row in data_rows:
                if source_field in row and row[source_field] is not None:
                    value = str(row[source_field]).strip()
                    if value:
                        unique_values.add(value)
            
            if unique_values:
                variance_data[target_field] = sorted(list(unique_values))
        
        return variance_data
    
    def _transform_field_value(
        self, 
        source_value: Any, 
        target_field: str, 
        source_field: str
    ) -> str:
        """Transform individual field values with German-to-English translations."""
        
        value = str(source_value).strip()
        
        # Country translations
        if target_field == "country_of_origin":
            country_translations = {
                "Deutschland": "Germany",
                "Tunesien": "Tunisia", 
                "Italien": "Italy",
                "China": "China",
                "Polen": "Poland"
            }
            return country_translations.get(value, value)
        
        # Product type mapping - use valid Amazon values
        elif target_field == "feed_product_type":
            if "hose" in value.lower() or "latzhose" in value.lower():
                return "pants"
            elif "jacke" in value.lower():
                return "jacket"  
            elif "hemd" in value.lower():
                return "shirt"
            elif "lahn" in value.lower():
                return "pants"  # LAHN is a product line for work pants
            return "pants"  # Default to pants for safety wear
        
        # Brand and manufacturer - use as-is but clean
        elif target_field in ["brand_name", "manufacturer"]:
            return value.strip()
        
        # Department name - extract from GROUP_STRING
        elif target_field == "department_name":
            if "|" in value:
                # Extract last part of hierarchy: "Root|Arbeitskleidung|Latzhosen" -> "mens"
                parts = value.split("|")
                if "arbeitskleidung" in parts[-2].lower():
                    return "mens"
            return "mens"  # Default for work clothing
        
        # Item names - construct from available info
        elif target_field == "item_name":
            # If it's a long description, use it directly
            if len(value) > 50:
                return value
            # Otherwise construct from brand + type if available
            return value
            
        # Default: return cleaned value
        return value
    
    def _generate_summary(
        self,
        results: List[Dict[str, Any]],
        total_processing_time: float
    ) -> Dict[str, Any]:
        """Generate processing summary."""
        
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        
        total_confidence = sum(r.get("confidence", 0.0) for r in successful)
        avg_confidence = total_confidence / len(successful) if successful else 0.0
        
        return {
            "ai_mapping_completed": True,
            "summary": {
                "total_parents": len(results),
                "successful": len(successful),
                "failed": len(failed),
                "success_rate": len(successful) / len(results) if results else 0.0,
                "average_confidence": avg_confidence
            },
            "performance": {
                "total_processing_time_ms": total_processing_time * 1000,
                "average_time_per_parent": (
                    total_processing_time * 1000 / len(results) if results else 0.0
                ),
                "processing_stats": self.processing_stats
            },
            "details": results
        }