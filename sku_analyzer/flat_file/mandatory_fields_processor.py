"""Step 3: Mandatory fields processor with compact value extraction.

This module extracts only mandatory fields from template analysis and their valid values,
creating a compact output focused on essential data requirements.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .performance_monitor import PerformanceMonitor
from .validation_utils import ValidationUtils


class MandatoryFieldsProcessor:
    """Processor for extracting mandatory fields and their values.

    Focuses on mandatory fields only, creating compact output with minimal structure
    while maintaining data integrity and type inference capabilities.
    """

    def __init__(self, enable_performance_monitoring: bool = True) -> None:
        """Initialize mandatory fields processor.
        
        Args:
            enable_performance_monitoring: Whether to enable performance tracking
        """
        self.logger = self._setup_logging()
        self.enable_monitoring = enable_performance_monitoring
        self.performance_monitor = PerformanceMonitor(enable_performance_monitoring)
        self.validation_utils = ValidationUtils()

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

    def process_mandatory_fields(
        self,
        step1_path: str | Path,
        step2_path: str | Path,
        output_path: str | Path
    ) -> dict[str, Any]:
        """Main processor for mandatory fields extraction.
        
        Args:
            step1_path: Path to step1_template_columns.json
            step2_path: Path to step2_valid_values.json  
            output_path: Path for step3_mandatory_fields.json output
            
        Returns:
            Dictionary containing processed mandatory fields data
            
        Raises:
            FileNotFoundError: If input files don't exist
            ValueError: If data format is invalid
        """
        start_time = time.time()
        self.logger.info("Starting mandatory fields processing")

        try:
            # Load input files
            step1_data = self._load_json_file(Path(step1_path))
            step2_data = self._load_json_file(Path(step2_path))

            # Identify mandatory fields from step1
            mandatory_fields = self._identify_mandatory_fields(step1_data)
            self.logger.info(f"Identified {len(mandatory_fields)} mandatory fields")

            # Extract values for mandatory fields from step2
            mandatory_data = self._extract_field_values(step2_data, mandatory_fields)

            # Create compact output structure
            output_data = self._create_compact_output(
                mandatory_data,
                [Path(step1_path).name, Path(step2_path).name]
            )

            # Save output
            self._save_json_file(Path(output_path), output_data)

            duration_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"Mandatory fields processing completed in {duration_ms:.2f}ms"
            )

            return output_data

        except Exception as e:
            self.logger.error(f"Error processing mandatory fields: {e}")
            raise

    def _load_json_file(self, file_path: Path) -> dict[str, Any]:
        """Load and validate JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded JSON data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with file_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {file_path}: {e}")

    def _save_json_file(self, file_path: Path, data: dict[str, Any]) -> None:
        """Save data to JSON file with formatting.
        
        Args:
            file_path: Output file path
            data: Data to save
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _identify_mandatory_fields(self, step1_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Filter mandatory fields from step1 template analysis.
        
        Args:
            step1_data: Step 1 template columns data
            
        Returns:
            List of mandatory field mappings
        """
        column_mappings = step1_data.get('column_mappings', [])
        mandatory_fields = []

        for mapping in column_mappings:
            requirement_status = mapping.get('requirement_status')
            if requirement_status == "mandatory":
                mandatory_fields.append(mapping)

        self.logger.debug(f"Found {len(mandatory_fields)} mandatory fields")
        return mandatory_fields

    def _extract_field_values(
        self,
        step2_data: dict[str, Any],
        mandatory_fields: list[dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Extract values for mandatory fields from step2 data.

        Args:
            step2_data: Step 2 valid values data
            mandatory_fields: List of mandatory field mappings

        Returns:
            Dictionary mapping field names to their data and values
        """
        field_validations = step2_data.get('field_validations', {})
        mandatory_data = {}

        for field_mapping in mandatory_fields:
            technical_name = field_mapping['technical_name']
            display_name = field_mapping['display_name']

            # Try both technical_name and display_name as keys in step2 data
            validation_data = (
                field_validations.get(technical_name) or 
                field_validations.get(display_name) or 
                {}
            )
            valid_values = validation_data.get('valid_values', [])

            # Create field entry
            mandatory_data[technical_name] = {
                'display_name': display_name,
                'data_type': self._infer_data_type(valid_values),
                'unique_values': list(set(valid_values)),  # Ensure uniqueness
                'constraints': self._extract_constraints(valid_values, validation_data),
                'source_mapping': field_mapping
            }

        return mandatory_data

    def _infer_data_type(self, values: list[str]) -> str:
        """Infer data type from valid values using simple heuristics.
        
        Args:
            values: List of valid values
            
        Returns:
            Inferred data type string
        """
        if not values:
            return "string"

        # Check for boolean patterns
        boolean_values = {'true', 'false', 'yes', 'no', '1', '0', 'ja', 'nein'}
        if all(str(v).lower() in boolean_values for v in values if v):
            return "boolean"

        # Check for numeric patterns
        numeric_count = 0
        for value in values:
            if value and str(value).strip():
                try:
                    float(str(value).replace(',', '.'))
                    numeric_count += 1
                except ValueError:
                    pass

        # If majority are numeric, classify as numeric
        if len(values) > 0 and numeric_count / len(values) > 0.8:
            return "numeric"

        # Check for date patterns (basic)
        date_indicators = ['date', 'time', 'jahr', 'month', 'day']
        if any(indicator in str(v).lower() for v in values for indicator in date_indicators):
            return "date"

        return "string"

    def _extract_constraints(
        self,
        values: list[str],
        validation_data: dict[str, Any]
    ) -> dict[str, int | None]:
        """Extract constraints from values and validation data.
        
        Args:
            values: List of valid values
            validation_data: Validation statistics
            
        Returns:
            Dictionary of constraints
        """
        constraints = {
            'value_count': len(set(values)),
            'max_length': None
        }

        # Calculate max length for string values
        if values:
            max_len = max(len(str(v)) for v in values if v)
            constraints['max_length'] = max_len if max_len > 0 else None

        return constraints

    def _create_compact_output(
        self,
        mandatory_data: dict[str, dict[str, Any]],
        source_files: list[str]
    ) -> dict[str, Any]:
        """Create compact output structure for mandatory fields.

        Args:
            mandatory_data: Processed mandatory field data
            source_files: List of source file names

        Returns:
            Compact output dictionary with product data only
        """
        # Create clean output with only product-related data
        clean_mandatory_fields = {}
        for field_name, field_data in mandatory_data.items():
            clean_mandatory_fields[field_name] = {
                'display_name': field_data['display_name'],
                'data_type': field_data['data_type'],
                'valid_values': field_data['unique_values'],
                'constraints': field_data['constraints']
            }

        return clean_mandatory_fields
