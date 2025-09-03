"""Main SKU pattern analyzer with performance optimization."""

import asyncio
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd

from ..models import ParentChildRelationship, ProcessingJob
from ..utils.performance_monitor import PerformanceMonitor
from .hierarchy import HierarchyExtractor


class PipelineValidationError(Exception):
    """Raised when pipeline step validation fails."""
    pass


class SkuPatternAnalyzer:
    """Modern async SKU pattern analyzer with hierarchical delimiter-based approach."""
    
    def __init__(self, min_pattern_length: int = 3, min_children: int = 2):
        self.min_pattern_length = min_pattern_length
        self.min_children = min_children
        self.logger = self._setup_logging()
        self.hierarchy_extractor = HierarchyExtractor(min_pattern_length)
        self.performance_monitor = PerformanceMonitor()
        
        # Performance optimization settings - compression always enabled
        self.enable_compression_optimization = True
        self.compression_benchmark_enabled = False
    
    def _get_next_job_number(self) -> int:
        """Get the next consecutive job number."""
        production_dir = Path("production_output")
        if not production_dir.exists():
            return 1
        
        # Find existing job directories (numeric only)
        job_numbers = []
        for job_dir in production_dir.iterdir():
            if job_dir.is_dir() and job_dir.name.isdigit():
                job_numbers.append(int(job_dir.name))
        
        return max(job_numbers, default=0) + 1
    
    def get_latest_job_number(self) -> Optional[int]:
        """Get the most recent job number."""
        production_dir = Path("production_output")
        if not production_dir.exists():
            return None
        
        job_numbers = []
        for job_dir in production_dir.iterdir():
            if job_dir.is_dir() and job_dir.name.isdigit():
                job_numbers.append(int(job_dir.name))
        
        return max(job_numbers) if job_numbers else None
    
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
    
    def completion_check(self, output_dir: Path, relationships: Dict[str, ParentChildRelationship], export_csv: bool = True, validate_flat_file: bool = False, validate_ai_mapping: bool = False) -> None:
        """Check that all expected pipeline output files exist after each step.
        
        Args:
            output_dir: The job output directory
            relationships: Parent-child relationships for validation
            export_csv: Whether CSV export was enabled
            validate_flat_file: Whether to validate flat file template analysis output
            validate_ai_mapping: Whether to validate AI mapping step 5 output
            
        Raises:
            PipelineValidationError: If any expected files are missing
        """
        missing_files = []
        
        # Step 1: Check analysis results file exists
        analysis_file = output_dir / f"analysis_{output_dir.name}.json"
        if not analysis_file.exists():
            missing_files.append(f"Step 1: {analysis_file}")
        
        # Flat File Steps: Check template analysis results (when enabled)
        if validate_flat_file:
            flat_file_dir = output_dir / "flat_file_analysis"
            
            # Flat File Step 1: Template columns
            step1_file = flat_file_dir / "step1_template_columns.json"
            if not step1_file.exists():
                missing_files.append(f"Flat File Step 1: {step1_file}")
            
            # Flat File Step 2: Valid values (if Step 1 exists)
            step2_file = flat_file_dir / "step2_valid_values.json"
            if step1_file.exists() and not step2_file.exists():
                missing_files.append(f"Flat File Step 2: {step2_file}")
            
            # Flat File Step 3: Mandatory fields (if Steps 1 & 2 exist)
            step3_file = flat_file_dir / "step3_mandatory_fields.json"
            if step1_file.exists() and step2_file.exists() and not step3_file.exists():
                missing_files.append(f"Flat File Step 3: {step3_file}")
            
            # Flat File Step 4: Template structure (if Step 3 exists)
            step4_file = flat_file_dir / "step4_template.json"
            if step3_file.exists() and not step4_file.exists():
                missing_files.append(f"Flat File Step 4: {step4_file}")
        
        # Step 2 & 3: Check per-parent files exist (only when CSV export is enabled)
        if relationships and export_csv:
            for parent_sku in relationships.keys():
                parent_folder = output_dir / f"parent_{parent_sku}"
                
                # Step 2: Check CSV data file exists
                csv_file = parent_folder / "data.csv"
                if not csv_file.exists():
                    missing_files.append(f"Step 2: {csv_file}")
                
                # Step 3: Check compressed JSON exists (compression is always enabled)
                json_file = parent_folder / "step2_compressed.json"
                
                # Always validate compressed JSON files - compression is now integral to workflow
                if not json_file.exists():
                    missing_files.append(f"Step 3: {json_file}")
                
                # Step 5: Check AI mapping files exist (when AI mapping validation enabled)
                if validate_ai_mapping:
                    ai_mapping_file = parent_folder / "step5_ai_mapping.json"
                    if not ai_mapping_file.exists():
                        missing_files.append(f"Step 5: {ai_mapping_file}")
        
        # Fail immediately if any files are missing
        if missing_files:
            self.logger.error(f"Pipeline validation failed - missing files:")
            for missing in missing_files:
                self.logger.error(f"  ❌ {missing}")
            raise PipelineValidationError(
                f"Pipeline step validation failed. Missing {len(missing_files)} expected files: "
                f"{', '.join(str(Path(f).name) for f in missing_files)}"
            )
        
        # Log what was validated
        validation_scope = []
        if analysis_file.exists():
            validation_scope.append("analysis metadata")
        if validate_flat_file:
            flat_file_dir = output_dir / "flat_file_analysis"
            flat_file_steps = []
            if (flat_file_dir / "step1_template_columns.json").exists():
                flat_file_steps.append("step1")
            if (flat_file_dir / "step2_valid_values.json").exists():
                flat_file_steps.append("step2")
            if (flat_file_dir / "step3_mandatory_fields.json").exists():
                flat_file_steps.append("step3")
            if (flat_file_dir / "step4_template.json").exists():
                flat_file_steps.append("step4")
            validation_scope.append(f"flat file analysis ({'+'.join(flat_file_steps)})")
        if relationships and export_csv:
            validation_scope.append(f"{len(relationships)} CSV files")
            validation_scope.append(f"{len(relationships)} compressed JSON files")
            if validate_ai_mapping:
                validation_scope.append(f"{len(relationships)} AI mapping files")
        
        self.logger.info(f"✅ Pipeline validation passed - all expected files exist: {', '.join(validation_scope)}")
    
    async def load_xlsx_data(self, file_path: Path) -> pd.DataFrame:
        """Asynchronously load XLSX data with memory optimization."""
        self.logger.info(f"Loading XLSX data from {file_path}")
        
        async with self.performance_monitor.measure_operation("XLSX_Load"):
            try:
                # Use asyncio to run pandas operations in thread pool
                loop = asyncio.get_event_loop()
                df = await loop.run_in_executor(
                    None, 
                    self._load_xlsx_optimized, 
                    file_path
                )
                
                self.logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
                return df
                
            except Exception as e:
                self.logger.error(f"Failed to load XLSX: {e}")
                raise
    
    def _load_xlsx_optimized(self, file_path: Path) -> pd.DataFrame:
        """Optimized XLSX loading with memory efficiency."""
        # Memory-optimized data types
        dtype_mapping = {
            'SUPPLIER_PID': 'string',  # pandas 2.x string dtype
        }
        
        # Load with chunking for large files (future-proofing)
        try:
            df = pd.read_excel(
                file_path,
                dtype=dtype_mapping,
                engine='openpyxl'
            )
            
            # Memory optimization: convert object columns to category where appropriate
            for col in df.select_dtypes(include=['object']).columns:
                if col != 'SUPPLIER_PID':  # Keep SUPPLIER_PID as string for performance
                    unique_ratio = df[col].nunique() / len(df)
                    if unique_ratio < 0.5:  # If less than 50% unique values
                        df[col] = df[col].astype('category')
            
            return df
            
        except Exception as e:
            self.logger.error(f"XLSX loading optimization failed: {e}")
            # Fallback to basic loading
            return pd.read_excel(file_path, engine='openpyxl')
    
    def build_hierarchical_relationships(
        self, 
        hierarchy: Dict[str, Dict[str, List[str]]]
    ) -> Dict[str, ParentChildRelationship]:
        """Build parent-child relationships from hierarchical structure."""
        self.logger.info("Building hierarchical parent-child relationships...")
        
        relationships = {}
        
        for level1_parent, level2_groups in hierarchy.items():
            # Collect all children under this top-level parent
            all_children = []
            for level2_parent, child_skus in level2_groups.items():
                all_children.extend(child_skus)
            
            if len(all_children) >= self.min_children:
                # Use level1_parent as the true parent (e.g., "4301")
                relationship = ParentChildRelationship(
                    parent_sku=level1_parent,
                    metadata={
                        'level1_category': level1_parent,
                        'subcategories': list(level2_groups.keys()),
                        'hierarchy_depth': 1,
                        'total_child_count': len(all_children)
                    }
                )
                
                # Add all children SKUs
                for child_sku in sorted(all_children):
                    relationship.add_child(child_sku)
                
                relationship.calculate_confidence()
                relationships[level1_parent] = relationship
        
        self.logger.info(f"Built {len(relationships)} hierarchical relationships")
        return relationships
    
    async def analyze_flat_file_template(
        self,
        template_path: Optional[Union[str, Path]],
        job_id: str,
        output_dir: Path
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze flat file template and add results to existing job.
        
        Args:
            template_path: Path to XLSM template file (optional)
            job_id: Current job ID
            output_dir: Job output directory
            
        Returns:
            Template analysis results or None if no template provided
            
        Raises:
            Exception: When template analysis fails
        """
        if template_path is None:
            self.logger.info("No template file provided - skipping flat file analysis")
            return None
        
        template_path = Path(template_path)
        if not template_path.exists():
            self.logger.warning(f"Template file not found: {template_path} - skipping flat file analysis")
            return None
        
        if not template_path.suffix.lower() in ['.xlsm', '.xlsx']:
            self.logger.warning(f"Invalid template file type: {template_path.suffix} - skipping flat file analysis")
            return None
        
        self.logger.info(f"🔍 Adding flat file template analysis to job {job_id}")
        
        try:
            # Import template analyzer
            from ..flat_file.template_analyzer import OptimizedXlsmTemplateAnalyzer
            from ..flat_file.mandatory_fields_processor import MandatoryFieldsProcessor
            
            # Initialize template analyzer with performance monitoring
            template_analyzer = OptimizedXlsmTemplateAnalyzer(enable_performance_monitoring=True)
            
            # Run template analysis for the existing job
            template_result = await template_analyzer.analyze_template_for_existing_job(
                template_path, job_id
            )
            
            self.logger.info(
                f"✅ Template analysis completed: {template_result.get('total_mappings', 0)} mappings"
            )
            
            # Step 3: Process mandatory fields if both steps 1 & 2 exist
            flat_file_dir = output_dir / "flat_file_analysis"
            step1_path = flat_file_dir / "step1_template_columns.json"
            step2_path = flat_file_dir / "step2_valid_values.json"
            step3_mandatory_path = flat_file_dir / "step3_mandatory_fields.json"
            
            if step1_path.exists() and step2_path.exists():
                self.logger.info("🎯 Processing mandatory fields (Step 3)")
                
                # Initialize mandatory fields processor
                mandatory_processor = MandatoryFieldsProcessor(enable_performance_monitoring=True)
                
                # Process mandatory fields
                mandatory_processor.process_mandatory_fields(
                    step1_path, step2_path, step3_mandatory_path
                )
                
                self.logger.info("✅ Step 3: Mandatory fields processing completed")
                
                # Step 4: Generate template structure if step 3 exists
                if step3_mandatory_path.exists():
                    await self._generate_template_structure(step3_mandatory_path, flat_file_dir, job_id)
            else:
                self.logger.warning("⚠️ Skipping Steps 3-4: Step 1 or Step 2 output missing")
            
            return template_result
            
        except Exception as e:
            self.logger.error(f"Template analysis failed for job {job_id}: {e}")
            raise Exception(f"Flat file template analysis failed: {e}") from e
    
    async def _generate_template_structure(
        self,
        step3_mandatory_path: Path,
        flat_file_dir: Path,
        job_id: str
    ) -> None:
        """Generate template structure from mandatory fields analysis.
        
        Args:
            step3_mandatory_path: Path to step3_mandatory_fields.json
            flat_file_dir: Flat file analysis directory
            job_id: Current job ID
        """
        try:
            self.logger.info("🏗️ Generating template structure (Step 4)")
            
            # Import template generator
            from ..step4_template.generator import TemplateGenerator
            
            # Initialize template generator
            template_generator = TemplateGenerator(enable_performance_monitoring=True)
            
            # Generate template structure
            step4_template_path = flat_file_dir / "step4_template.json"
            template_result = await template_generator.generate_template_from_mandatory_fields(
                step3_mandatory_path, step4_template_path
            )
            
            # Log template generation results
            metadata = template_result.get('metadata', {})
            field_distribution = metadata.get('field_distribution', {})
            quality_score = metadata.get('quality_score', 0.0)
            
            self.logger.info(
                f"✅ Step 4: Template generation completed - "
                f"Quality: {quality_score:.2f}, "
                f"Parent fields: {field_distribution.get('parent_fields', 0)}, "
                f"Variant fields: {field_distribution.get('variant_fields', 0)}"
            )
            
            # Log warnings if any
            warnings = metadata.get('warnings', [])
            if warnings:
                self.logger.warning(f"Template generation warnings: {warnings}")
            
        except Exception as e:
            self.logger.error(f"Template structure generation failed for job {job_id}: {e}")
            # Don't raise - template generation is optional, analysis can continue
    
    async def add_ai_mapping_to_job(
        self,
        job_id: str,
        starting_parent: str = "4301"
    ) -> Dict[str, Any]:
        """
        Add AI mapping (step 5) to an existing job with template analysis.
        
        Args:
            job_id: Existing job ID with template analysis completed
            starting_parent: Starting parent SKU for AI processing
            
        Returns:
            AI mapping results
            
        Raises:
            ValueError: When job directory or required files don't exist
            Exception: When AI mapping fails
        """
        job_dir = Path("production_output") / str(job_id)
        if not job_dir.exists():
            raise ValueError(f"Job directory not found: {job_dir}")
        
        # Check if step 4 template exists
        template_file = job_dir / "flat_file_analysis" / "step4_template.json"
        if not template_file.exists():
            raise ValueError(f"Step 4 template not found: {template_file}")
        
        self.logger.info(f"🤖 Adding AI mapping (Step 5) to job {job_id}")
        
        try:
            # Import AI mapping processor
            from ..step5_mapping.processor import MappingProcessor
            from ..shared.gemini_client import AIProcessingConfig
            
            # Initialize AI processor
            ai_config = AIProcessingConfig()
            ai_processor = MappingProcessor(ai_config)
            
            # Process AI mapping for all parents
            ai_results = await ai_processor.process_all_parents(
                job_dir, starting_parent
            )
            
            # Log AI mapping results
            summary = ai_results.summary
            successful = summary.get('successful', 0)
            total = summary.get('total_parents', 0)
            avg_confidence = summary.get('average_confidence', 0.0)
            
            self.logger.info(
                f"✅ Step 5: AI mapping completed - "
                f"{successful}/{total} parents processed, "
                f"Average confidence: {avg_confidence:.2f}"
            )
            
            return ai_results
            
        except Exception as e:
            self.logger.error(f"AI mapping failed for job {job_id}: {e}")
            raise Exception(f"AI mapping (Step 5) failed: {e}") from e
    
    async def process_file_with_template(
        self,
        input_path: str | Path,
        template_path: Optional[str | Path] = None,
        export_csv: bool = True,
        enable_compression_benchmark: bool = False
    ) -> str:
        """
        Enhanced processing entry point with integrated template analysis.
        
        This method provides unified workflow for both SKU analysis and template analysis
        within the same job, creating a complete analysis result structure.
        
        Args:
            input_path: Path to XLSX data file for SKU analysis
            template_path: Optional path to XLSM template file for flat file analysis
            export_csv: Whether to export CSV files for parent groups
            enable_compression_benchmark: Whether to run compression benchmarks
            
        Returns:
            Job ID containing both SKU and template analysis results
            
        Raises:
            FileNotFoundError: When input file doesn't exist
            PipelineValidationError: When pipeline validation fails
            Exception: When any analysis step fails
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create job with consecutive number
        job_number = self._get_next_job_number()
        job_id = str(job_number)
        output_dir = Path("production_output") / job_id
        
        job = ProcessingJob(
            job_id=job_id,
            input_path=input_path,
            output_dir=output_dir
        )
        
        # Track whether template analysis was performed
        has_template_analysis = template_path is not None
        
        # Start comprehensive performance monitoring
        async with self.performance_monitor.measure_operation("Complete_Integrated_Pipeline"):
            try:
                # Phase 1: SKU Analysis
                job.status = "loading"
                df = await self.load_xlsx_data(input_path)
                
                job.status = "analyzing"
                hierarchy = self.hierarchy_extractor.extract_hierarchical_patterns(df)
                relationships = self.build_hierarchical_relationships(hierarchy)
                
                job.status = "saving"
                from ..output.json_writer import JsonWriter
                
                # Initialize JSON writer with compression settings
                writer = JsonWriter(
                    enable_compression=self.enable_compression_optimization,
                    max_workers=4
                )
                
                await writer.save_results(
                    job, 
                    relationships, 
                    df, 
                    export_csv,
                    enable_compression_benchmark=enable_compression_benchmark or self.compression_benchmark_enabled
                )
                
                # Phase 2: Template Analysis (if provided)
                template_result = None
                if has_template_analysis:
                    job.status = "template_analyzing"
                    template_result = await self.analyze_flat_file_template(
                        template_path, job_id, output_dir
                    )
                
                # Validate all expected files exist after pipeline completion
                self.completion_check(output_dir, relationships, export_csv, validate_flat_file=has_template_analysis)
                
                job.status = "completed"
                
                # Log performance summary
                summary = self.performance_monitor.get_performance_summary()
                self.logger.info(f"🎯 Integrated pipeline performance: {summary}")
                
                # Log compression capabilities (always enabled)
                capabilities = writer.get_compression_capabilities()
                self.logger.info(f"🗜️ Compression: {capabilities['json_library']} library, {capabilities['performance_features']}")
                
                # Log integration summary
                integration_summary = [f"SKU analysis ({len(relationships)} parent groups)"]
                if template_result:
                    mapping_count = template_result.get('total_mappings', 0)
                    integration_summary.append(f"Template analysis ({mapping_count} column mappings)")
                
                self.logger.info(f"🏗️ Integrated analysis completed: {', '.join(integration_summary)}")
                
                return job_id
                
            except Exception as e:
                job.status = "failed"
                self.logger.error(f"Integrated job {job_id} failed: {e}")
                raise
    
    async def add_template_analysis_to_job(
        self,
        template_path: str | Path,
        job_id: str
    ) -> Dict[str, Any]:
        """
        Add template analysis to an existing completed job.
        
        This method allows adding template analysis as a separate step to jobs
        that were completed without template analysis.
        
        Args:
            template_path: Path to XLSM template file
            job_id: Existing job ID to add template analysis to
            
        Returns:
            Template analysis results
            
        Raises:
            ValueError: When job directory doesn't exist
            Exception: When template analysis fails
        """
        job_dir = Path("production_output") / str(job_id)
        if not job_dir.exists():
            raise ValueError(f"Job directory not found: {job_dir}")
        
        self.logger.info(f"🔗 Adding template analysis to existing job {job_id}")
        
        # Run template analysis for existing job
        template_result = await self.analyze_flat_file_template(
            template_path, job_id, job_dir
        )
        
        if template_result:
            # Update completion check to validate template analysis
            # Note: We don't re-validate SKU analysis files, just the new template analysis
            flat_file_analysis = job_dir / "flat_file_analysis" / "step1_template_columns.json"
            if not flat_file_analysis.exists():
                raise PipelineValidationError(f"Template analysis validation failed: {flat_file_analysis} not found")
            
            self.logger.info(f"✅ Template analysis successfully added to job {job_id}")
        
        return template_result
    
    async def process_file(
        self, 
        input_path: str | Path, 
        export_csv: bool = True,
        enable_compression_benchmark: bool = False
    ) -> str:
        """Main processing entry point with performance monitoring."""
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Create job with consecutive number
        job_number = self._get_next_job_number()
        job_id = str(job_number)
        output_dir = Path("production_output") / job_id
        
        job = ProcessingJob(
            job_id=job_id,
            input_path=input_path,
            output_dir=output_dir
        )
        
        # Start comprehensive performance monitoring
        async with self.performance_monitor.measure_operation("Complete_Pipeline"):
            try:
                job.status = "loading"
                df = await self.load_xlsx_data(input_path)
                
                job.status = "analyzing"
                # Use hierarchical approach for better parent-child discovery
                hierarchy = self.hierarchy_extractor.extract_hierarchical_patterns(df)
                relationships = self.build_hierarchical_relationships(hierarchy)
                
                job.status = "saving"
                from ..output.json_writer import JsonWriter
                
                # Initialize JSON writer with compression settings
                writer = JsonWriter(
                    enable_compression=self.enable_compression_optimization,
                    max_workers=4
                )
                
                await writer.save_results(
                    job, 
                    relationships, 
                    df, 
                    export_csv,
                    enable_compression_benchmark=enable_compression_benchmark or self.compression_benchmark_enabled
                )
                
                # Validate all expected files exist after pipeline completion
                self.completion_check(output_dir, relationships, export_csv)
                
                job.status = "completed"
                
                # Log performance summary
                summary = self.performance_monitor.get_performance_summary()
                self.logger.info(f"🎯 Pipeline performance: {summary}")
                
                # Log compression capabilities (always enabled)
                capabilities = writer.get_compression_capabilities()
                self.logger.info(f"🗜️ Compression: {capabilities['json_library']} library, {capabilities['performance_features']}")
                
                return job_id
                
            except Exception as e:
                job.status = "failed"
                self.logger.error(f"Job {job_id} failed: {e}")
                raise
    
    async def benchmark_performance(self, input_path: str | Path) -> Dict:
        """Run performance benchmark on CSV export operations."""
        input_path = Path(input_path)
        
        self.logger.info("🚀 Starting performance benchmark...")
        
        # Load test data
        df = await self.load_xlsx_data(input_path)
        hierarchy = self.hierarchy_extractor.extract_hierarchical_patterns(df)
        relationships = self.build_hierarchical_relationships(hierarchy)
        
        # Create temporary benchmark job
        benchmark_dir = Path("benchmark_output")
        benchmark_dir.mkdir(exist_ok=True)
        
        # Run benchmark
        from ..utils.performance_monitor import CsvPerformanceBenchmark
        
        metrics = await CsvPerformanceBenchmark.benchmark_csv_export(
            df, relationships, benchmark_dir
        )
        
        # Validate against targets
        validation = self.performance_monitor.validate_performance_targets(
            metrics,
            max_duration_seconds=5.0,
            max_memory_mb=100.0
        )
        
        benchmark_results = {
            'metrics': {
                'duration_seconds': metrics.duration_seconds,
                'peak_memory_mb': metrics.peak_memory_mb,
                'throughput_rows_per_second': metrics.throughput_rows_per_second,
                'memory_efficiency_mb_per_1k_rows': metrics.memory_efficiency_mb_per_1k_rows,
                'files_created': metrics.files_created
            },
            'validation': validation,
            'dataset_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'parent_groups': len(relationships)
            }
        }
        
        # Clean up benchmark files
        import shutil
        if benchmark_dir.exists():
            shutil.rmtree(benchmark_dir)
        
        return benchmark_results
    
    async def run_compression_benchmark(
        self, 
        input_path: str | Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Run comprehensive compression performance benchmark."""
        input_path = Path(input_path)
        
        if output_dir is None:
            output_dir = Path("compression_benchmarks")
        
        self.logger.info("🎯 Starting comprehensive compression benchmark...")
        
        # Load data
        df = await self.load_xlsx_data(input_path)
        hierarchy = self.hierarchy_extractor.extract_hierarchical_patterns(df)
        relationships = self.build_hierarchical_relationships(hierarchy)
        
        # Run comprehensive benchmark
        from ..optimization import CompressionPerformanceBenchmark
        
        benchmark_tool = CompressionPerformanceBenchmark(output_dir)
        benchmark_results = await benchmark_tool.run_full_benchmark_suite(
            df, relationships, f"analyzer_benchmark_{input_path.stem}"
        )
        
        self.logger.info(
            f"✅ Compression benchmark completed: "
            f"{benchmark_results.compression_performance.get('pipeline_metrics', {}).get('overall_compression_ratio', 0):.1%} compression ratio"
        )
        
        return {
            'comprehensive_benchmark': benchmark_results,
            'performance_summary': {
                'compression_ratio': benchmark_results.compression_performance.get('pipeline_metrics', {}).get('overall_compression_ratio', 0),
                'processing_time_seconds': benchmark_results.compression_performance.get('pipeline_metrics', {}).get('total_processing_time_seconds', 0),
                'memory_efficiency': benchmark_results.memory_analysis.get('memory_target_validation', {}),
                'optimal_json_library': benchmark_results.library_comparison.get('recommendation', {}).get('optimal_library', 'json'),
                'performance_targets_met': benchmark_results.performance_targets_met
            }
        }