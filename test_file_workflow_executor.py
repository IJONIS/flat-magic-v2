#!/usr/bin/env python3
"""
Test-File-Based Workflow Executor

A clean, modern workflow executor that dynamically discovers and processes
test files from the /test-files directory. Integrates with existing 
ProductionWorkflowTester for seamless workflow execution.

Features:
- Dynamic file discovery from test-files folder
- Interactive CLI for file selection
- Integration with existing workflow pipeline
- Clean architecture with single responsibility classes
- Comprehensive error handling and validation
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add current directory to path for importing production_workflow_test
sys.path.insert(0, str(Path(__file__).parent))

from sku_analyzer.core.analyzer import SkuPatternAnalyzer


class TestFileDiscovery:
    """Handles discovery and validation of test files in /test-files folder."""
    
    def __init__(self, test_files_dir: str = "test-files"):
        """Initialize file discovery with specified directory.
        
        Args:
            test_files_dir: Directory path containing test files
        """
        self.test_files_dir = Path(test_files_dir)
        self.logger = logging.getLogger(__name__)
    
    def scan_test_files(self) -> Dict[str, List[Path]]:
        """Scan test-files directory for SKU and flat files.
        
        Returns:
            Dictionary with 'sku_files' and 'flat_files' lists
            
        Raises:
            FileNotFoundError: If test-files directory doesn't exist
        """
        if not self.test_files_dir.exists():
            raise FileNotFoundError(
                f"Test files directory not found: {self.test_files_dir}"
            )
        
        discovered_files = {
            "sku_files": [],
            "flat_files": []
        }
        
        # Scan for Excel files
        for file_path in self.test_files_dir.iterdir():
            if file_path.is_file():
                if file_path.suffix.lower() == '.xlsx':
                    discovered_files["sku_files"].append(file_path)
                elif file_path.suffix.lower() == '.xlsm':
                    discovered_files["flat_files"].append(file_path)
        
        self._log_discovery_results(discovered_files)
        return discovered_files
    
    def validate_file_pairs(self, 
                          sku_files: List[Path], 
                          flat_files: List[Path]) -> List[Tuple[Path, Path]]:
        """Validate and create valid file pairs for processing.
        
        Args:
            sku_files: List of discovered SKU files (.xlsx)
            flat_files: List of discovered flat files (.xlsm)
            
        Returns:
            List of valid (sku_file, flat_file) pairs
        """
        if not sku_files:
            raise ValueError("No SKU files (.xlsx) found in test-files directory")
        
        if not flat_files:
            raise ValueError("No flat files (.xlsm) found in test-files directory")
        
        # For now, create all possible combinations
        # Future enhancement: intelligent pairing based on naming patterns
        valid_pairs = []
        for sku_file in sku_files:
            for flat_file in flat_files:
                valid_pairs.append((sku_file, flat_file))
        
        return valid_pairs
    
    def _log_discovery_results(self, discovered_files: Dict[str, List[Path]]) -> None:
        """Log file discovery results."""
        sku_count = len(discovered_files["sku_files"])
        flat_count = len(discovered_files["flat_files"])
        
        self.logger.info(f"📁 Discovered {sku_count} SKU files, {flat_count} flat files")
        
        for sku_file in discovered_files["sku_files"]:
            size_mb = sku_file.stat().st_size / (1024 * 1024)
            self.logger.info(f"   SKU: {sku_file.name} ({size_mb:.1f}MB)")
        
        for flat_file in discovered_files["flat_files"]:
            size_mb = flat_file.stat().st_size / (1024 * 1024)
            self.logger.info(f"   Flat: {flat_file.name} ({size_mb:.1f}MB)")


class TestFileWorkflowExecutor:
    """Executes workflow with dynamically discovered test files."""
    
    def __init__(self, test_files_dir: str = "test-files"):
        """Initialize workflow executor.
        
        Args:
            test_files_dir: Directory path containing test files
        """
        self.file_discovery = TestFileDiscovery(test_files_dir)
        self.logger = logging.getLogger(__name__)
        self.selected_pairs: List[Tuple[Path, Path]] = []
    
    def discover_test_files(self) -> Dict[str, List[Path]]:
        """Discover all available test files.
        
        Returns:
            Dictionary with discovered file lists
        """
        return self.file_discovery.scan_test_files()
    
    def display_file_selection_menu(self, 
                                  file_pairs: List[Tuple[Path, Path]]) -> None:
        """Display interactive file selection menu.
        
        Args:
            file_pairs: Available file pairs for selection
        """
        print("\n📋 Available Test File Combinations:")
        print("=" * 50)
        
        for idx, (sku_file, flat_file) in enumerate(file_pairs, 1):
            sku_size = sku_file.stat().st_size / (1024 * 1024)
            flat_size = flat_file.stat().st_size / (1024 * 1024)
            
            print(f"{idx}. SKU: {sku_file.name} ({sku_size:.1f}MB)")
            print(f"   Flat: {flat_file.name} ({flat_size:.1f}MB)")
            print()
    
    def get_user_selection(self, 
                          file_pairs: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
        """Get user selection of file pairs to process.
        
        Args:
            file_pairs: Available file pairs
            
        Returns:
            Selected file pairs for processing
        """
        while True:
            try:
                selection = input(
                    "Enter selection (number, 'all', or 'quit'): "
                ).strip().lower()
                
                if selection == 'quit':
                    return []
                
                if selection == 'all':
                    return file_pairs
                
                # Parse specific number selection
                selected_idx = int(selection) - 1
                if 0 <= selected_idx < len(file_pairs):
                    return [file_pairs[selected_idx]]
                else:
                    print(f"Invalid selection. Please choose 1-{len(file_pairs)}")
                    
            except ValueError:
                print("Invalid input. Please enter a number, 'all', or 'quit'")
    
    async def execute_workflow_with_files(self, 
                                        sku_file: Path, 
                                        flat_file: Path) -> Dict[str, any]:
        """Execute workflow with specified test files.
        
        Args:
            sku_file: SKU data file path
            flat_file: Flat template file path
            
        Returns:
            Workflow execution results
        """
        self.logger.info(f"🚀 Starting workflow execution")
        self.logger.info(f"   SKU File: {sku_file.name}")
        self.logger.info(f"   Flat File: {flat_file.name}")
        
        try:
            # Create analyzer and process files
            analyzer = self._create_workflow_analyzer(sku_file, flat_file)
            
            # Step 1-4: Execute complete SKU analysis with template
            job_id = await analyzer.process_file_with_template(
                input_path=str(sku_file),
                template_path=str(flat_file),
                export_csv=True
            )
            
            # Step 5: Add AI mapping to complete the pipeline
            await analyzer.add_ai_mapping_to_job(job_id)
            
            results = {
                "job_id": job_id,
                "sku_file": str(sku_file),
                "flat_file": str(flat_file),
                "status": "completed"
            }
            
            self.logger.info(f"✅ Complete workflow with AI mapping completed with job ID: {job_id}")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Workflow execution failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "sku_file": str(sku_file),
                "flat_file": str(flat_file)
            }
    
    def _create_workflow_analyzer(self, 
                               sku_file: Path, 
                               flat_file: Path) -> SkuPatternAnalyzer:
        """Create workflow analyzer for processing test files.
        
        Args:
            sku_file: SKU data file path
            flat_file: Flat template file path
            
        Returns:
            Configured SkuPatternAnalyzer instance
        """
        # Create analyzer instance
        analyzer = SkuPatternAnalyzer()
        
        # Store file paths for execution
        self._current_sku_file = sku_file
        self._current_flat_file = flat_file
        
        return analyzer
    
    async def run_interactive_workflow(self) -> Dict[str, any]:
        """Run complete interactive workflow execution.
        
        Returns:
            Consolidated results from all executed workflows
        """
        print("🎯 Test File Workflow Executor")
        print("=" * 40)
        
        try:
            # Step 1: Discover test files
            discovered_files = self.discover_test_files()
            
            # Step 2: Validate and create file pairs
            file_pairs = self.file_discovery.validate_file_pairs(
                discovered_files["sku_files"],
                discovered_files["flat_files"]
            )
            
            # Step 3: Display selection menu
            self.display_file_selection_menu(file_pairs)
            
            # Step 4: Get user selection
            selected_pairs = self.get_user_selection(file_pairs)
            
            if not selected_pairs:
                print("👋 Workflow execution cancelled by user")
                return {"status": "cancelled"}
            
            # Step 5: Execute workflows for selected pairs
            all_results = []
            
            for idx, (sku_file, flat_file) in enumerate(selected_pairs, 1):
                print(f"\n🔄 Processing combination {idx}/{len(selected_pairs)}")
                
                try:
                    results = await self.execute_workflow_with_files(
                        sku_file, flat_file
                    )
                    results["file_pair"] = {
                        "sku_file": sku_file.name,
                        "flat_file": flat_file.name
                    }
                    all_results.append(results)
                    
                except Exception as e:
                    self.logger.error(f"❌ Workflow failed for {sku_file.name}: {e}")
                    all_results.append({
                        "status": "failed",
                        "error": str(e),
                        "file_pair": {
                            "sku_file": sku_file.name,
                            "flat_file": flat_file.name
                        }
                    })
            
            return {
                "status": "completed",
                "processed_count": len(selected_pairs),
                "results": all_results
            }
            
        except Exception as e:
            self.logger.error(f"❌ Interactive workflow failed: {e}")
            return {
                "status": "error", 
                "error": str(e)
            }


async def main():
    """Main execution function for test file workflow executor."""
    print("🚀 Test File Workflow Executor")
    print("=" * 50)
    print("Features:")
    print("  • Dynamic test file discovery")
    print("  • Interactive file selection")
    print("  • Integration with existing workflow pipeline")
    print("  • Clean error handling and validation")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize and run workflow executor
        executor = TestFileWorkflowExecutor()
        results = await executor.run_interactive_workflow()
        
        # Display final summary
        print("\n📊 EXECUTION SUMMARY")
        print("=" * 30)
        
        if results["status"] == "completed":
            print(f"Files Processed: {results['processed_count']}")
            successful = len([r for r in results['results'] if 'error' not in r])
            failed = results['processed_count'] - successful
            
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            
            if successful > 0:
                print("✅ Workflow execution completed successfully!")
            if failed > 0:
                print(f"⚠️ {failed} workflows failed - check logs for details")
                
        elif results["status"] == "cancelled":
            print("👋 Execution cancelled by user")
        else:
            print(f"❌ Execution error: {results.get('error', 'Unknown error')}")
            
    except KeyboardInterrupt:
        print("\n🛑 Execution interrupted by user")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())