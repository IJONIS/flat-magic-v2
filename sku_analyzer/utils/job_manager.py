"""Job management utilities."""

import json
from pathlib import Path
from typing import Optional


class JobManager:
    """Manage SKU analysis jobs and results."""
    
    @staticmethod
    def show_latest_job() -> None:
        """Show the latest job results."""
        from ..core.analyzer import SkuPatternAnalyzer
        
        analyzer = SkuPatternAnalyzer()
        latest_job = analyzer.get_latest_job_number()
        
        if latest_job is None:
            print("No jobs found.")
            return
        
        job_dir = Path("production_output") / str(latest_job)
        
        # Try both old and new metadata file formats
        metadata_file = job_dir / f"analysis_{latest_job}.json"
        if not metadata_file.exists():
            metadata_file = job_dir / f"job_metadata_{latest_job}.json"
        
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
            print(f"📁 Latest Job: {latest_job}")
            print(f"🕐 Created: {metadata['created_at']}")
            print(f"📄 Results: {job_dir}/")
            print(f"📊 Status: {metadata['status']}")
            
            # Handle both old and new metadata formats
            summary = metadata.get('summary', metadata.get('results', {}))
            if summary:
                print(f"   • Total SKUs: {summary.get('total_skus', 'N/A')}")
                print(f"   • Parent groups: {summary.get('parent_child_groups', 'N/A')}")
                if 'parent_skus' in summary:
                    parents = summary['parent_skus']
                    print(f"   • Parents: {', '.join(parents[:5])}" + ("..." if len(parents) > 5 else ""))
            
            # Check for template analysis
            template_analysis_file = job_dir / "flat_file_analysis" / "step1_template_columns.json"
            if template_analysis_file.exists():
                print(f"📋 Template Analysis: Available")
                try:
                    template_data = json.loads(template_analysis_file.read_text())
                    template_metadata = template_data.get('analysis_metadata', {})
                    print(f"   • Column mappings: {template_metadata.get('total_mappings', 'N/A')}")
                    
                    # Show requirement statistics if available
                    req_stats = template_metadata.get('requirement_statistics', {})
                    if req_stats:
                        print(f"   • Requirements: {req_stats.get('mandatory', 0)} mandatory, {req_stats.get('optional', 0)} optional, {req_stats.get('recommended', 0)} recommended")
                except Exception as e:
                    print(f"   • Template data: Error reading ({e})")
            else:
                print(f"📋 Template Analysis: Not available")
            
            # Check for split results
            split_metadata_file = job_dir / f"split_metadata_{latest_job}.json"
            if split_metadata_file.exists():
                split_metadata = json.loads(split_metadata_file.read_text())
                print(f"📦 Split Results:")
                print(f"   • Files created: {split_metadata['summary']['total_files_created']}")
                print(f"   • Rows exported: {split_metadata['summary']['total_rows_exported']}")
                print(f"   • Split completed: {split_metadata['created_at']}")
        else:
            print(f"📁 Latest Job: {latest_job} (metadata missing)")
    
    @staticmethod
    def show_split_status(job_id: str) -> None:
        """Show split operation status for a specific job."""
        job_dir = Path("production_output") / str(job_id)
        split_metadata_file = job_dir / f"split_metadata_{job_id}.json"
        
        if not split_metadata_file.exists():
            print(f"❌ No split results found for job {job_id}")
            return
        
        try:
            split_metadata = json.loads(split_metadata_file.read_text())
            print(f"📦 Split Results for Job {job_id}:")
            print(f"🕐 Created: {split_metadata['created_at']}")
            print(f"📊 Summary:")
            print(f"   • Parent groups: {split_metadata['summary']['total_parent_groups']}")
            print(f"   • Files created: {split_metadata['summary']['total_files_created']}")
            print(f"   • Total rows: {split_metadata['summary']['total_rows_exported']}")
            
            print(f"📄 Parent Group Details:")
            for parent_sku, details in split_metadata['parent_groups'].items():
                status_icon = "✅" if details['success'] else "❌"
                print(f"   {status_icon} {parent_sku}: {details['row_count']} rows -> {details['csv_file']}")
                
        except Exception as e:
            print(f"❌ Failed to read split metadata: {e}")