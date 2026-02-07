"""
MAIN ORCHESTRATOR
Membrane Filtration Analytics Pipeline

Executes all pipeline stages in sequence:
1. Ingestion → 2. Validation → 3. Preprocessing → 4. Feature Engineering
→ 5. Fouling Metrics → 6. Cleaning Analysis → 7. KPI Engine → 8. Dashboard

Usage:
    python main.py
    python main.py --config custom_config.yaml
    python main.py --skip-viz  # Skip visualization
"""

import sys
import os
import logging
import argparse
import time
from datetime import datetime
import yaml
import webbrowser
import io

# Fix Windows console encoding for Unicode
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.ingestion import run_ingestion
from src.validation import run_validation
from src.preprocessing import run_preprocessing
from src.feature_engineering import run_feature_engineering
from src.fouling_metrics import run_fouling_metrics
from src.cleaning_analysis import run_cleaning_analysis
from src.kpi_engine import run_kpi_engine
from src.dashboard_app import run_dashboard_app


def setup_logging(log_file: str = None, verbose: bool = True):
    """Setup logging configuration"""
    log_level = logging.INFO if verbose else logging.WARNING

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)


def load_config(config_path: str) -> dict:
    """Load pipeline configuration"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def print_banner():
    """Print pipeline banner"""
    banner = """
===============================================================
                                                               
        MEMBRANE FILTRATION ANALYTICS PIPELINE                
        PWN CapNF System - Modular Architecture               
                                                               
===============================================================
    """
    print(banner)


def print_stage_header(stage_num: int, stage_name: str):
    """Print stage header"""
    print(f"\n{'=' * 70}")
    print(f"  STAGE {stage_num}: {stage_name}")
    print(f"{'=' * 70}\n")


def run_pipeline(
    config_path: str = "config.yaml", skip_viz: bool = False, fast_mode: bool = False
):
    """
    Execute the complete pipeline

    Args:
        config_path: Path to configuration file
        skip_viz: Skip visualization stage
        fast_mode: Use faster settings (less analysis depth)
    """
    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config(config_path)
    run_flags = config["pipeline"]["run_modules"]

    # Track execution time
    pipeline_start = time.time()
    stage_times = {}

    try:
        # Stage 1: Ingestion
        if run_flags["ingestion"]:
            print_stage_header(1, "DATA INGESTION")
            stage_start = time.time()
            df_raw = run_ingestion(config_path)
            stage_times["ingestion"] = time.time() - stage_start
            logger.info(f"✓ Stage 1 completed in {stage_times['ingestion']:.2f}s")

        # Stage 2: Validation
        if run_flags["validation"]:
            print_stage_header(2, "DATA VALIDATION")
            stage_start = time.time()
            df_validated, validation_report = run_validation(config_path)
            stage_times["validation"] = time.time() - stage_start
            logger.info(f"✓ Stage 2 completed in {stage_times['validation']:.2f}s")

        # Stage 3: Preprocessing
        if run_flags["preprocessing"]:
            print_stage_header(3, "PREPROCESSING")
            stage_start = time.time()
            df_processed, df_viz = run_preprocessing(config_path)
            stage_times["preprocessing"] = time.time() - stage_start
            logger.info(f"✓ Stage 3 completed in {stage_times['preprocessing']:.2f}s")

        # Stage 4: Feature Engineering
        if run_flags["feature_engineering"]:
            print_stage_header(4, "FEATURE ENGINEERING")
            stage_start = time.time()
            df_features = run_feature_engineering(config_path)
            stage_times["feature_engineering"] = time.time() - stage_start
            logger.info(
                f"✓ Stage 4 completed in {stage_times['feature_engineering']:.2f}s"
            )

        # Stage 5: Fouling Metrics
        if run_flags["fouling_metrics"]:
            print_stage_header(5, "FOULING METRICS")
            stage_start = time.time()
            df_fouling, fouling_metadata = run_fouling_metrics(config_path)
            stage_times["fouling_metrics"] = time.time() - stage_start
            logger.info(f"✓ Stage 5 completed in {stage_times['fouling_metrics']:.2f}s")

        # Stage 6: Cleaning Analysis
        if run_flags["cleaning_analysis"]:
            print_stage_header(6, "CLEANING ANALYSIS")
            stage_start = time.time()
            df_cycles, cycles = run_cleaning_analysis(config_path)
            stage_times["cleaning_analysis"] = time.time() - stage_start
            logger.info(
                f"✓ Stage 6 completed in {stage_times['cleaning_analysis']:.2f}s"
            )

        # Stage 7: KPI Engine
        if run_flags["kpi_engine"]:
            print_stage_header(7, "KPI ENGINE")
            stage_start = time.time()
            kpis, alerts, forecasts = run_kpi_engine(config_path)
            stage_times["kpi_engine"] = time.time() - stage_start
            logger.info(f"✓ Stage 7 completed in {stage_times['kpi_engine']:.2f}s")

        # Stage 8: Dashboard
        if run_flags["dashboard"] and not skip_viz:
            print_stage_header(8, "DASHBOARD GENERATION")
            stage_start = time.time()
            viz_files = run_dashboard_app(config_path)
            stage_times["dashboard"] = time.time() - stage_start
            logger.info(f"✓ Stage 8 completed in {stage_times['dashboard']:.2f}s")

        # Pipeline complete
        pipeline_time = time.time() - pipeline_start

        print("\n" + "=" * 70)
        print("  PIPELINE EXECUTION SUMMARY")
        print("=" * 70)
        print(f"\n✓ Pipeline completed successfully!")
        print(
            f"  Total execution time: {pipeline_time:.2f}s ({pipeline_time / 60:.1f} min)\n"
        )
        print("  Stage execution times:")
        for stage, duration in stage_times.items():
            print(f"    {stage:.<30} {duration:>8.2f}s")

        print(f"\n  Output directory: {config['paths']['output_folder']}")
        print(f"  Plots directory: {config['paths']['plots_folder']}")

        # Show key results
        if run_flags["dashboard"] and not skip_viz:
            index_path = os.path.join(config["paths"]["plots_folder"], "index.html")
            print(f"\n  🎯 MAIN DASHBOARD: {index_path}")
            print(f"     Opening in your default browser...")
            # Open in default browser
            try:
                abs_path = os.path.abspath(index_path)
                webbrowser.open(f"file:///{abs_path}".replace("\\", "/"))
            except Exception as e:
                print(f"     ⚠️  Could not open browser automatically: {e}")
                print(f"     Please open manually: {index_path}")

        if run_flags["kpi_engine"]:
            print(f"\n  📊 KPIs calculated: {len(kpis)}")
            print(f"  ⚠️  Alerts generated: {len(alerts)}")
            if alerts:
                critical = sum(1 for a in alerts if a.severity == "critical")
                if critical > 0:
                    print(f"      ({critical} CRITICAL alerts)")

        if run_flags["cleaning_analysis"]:
            print(f"  🔄 Filtration cycles identified: {len(cycles)}")

        print("\n" + "=" * 70)
        print(
            f"  Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 70 + "\n")

        return True

    except Exception as e:
        logger.error(f"\n❌ Pipeline failed at stage: {e}")
        logger.exception("Full error traceback:")

        if not config["pipeline"]["continue_on_error"]:
            raise

        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Membrane Filtration Analytics Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Run full pipeline
  python main.py --config my_config.yaml  # Use custom config
  python main.py --skip-viz               # Skip visualization
  python main.py --fast                   # Fast mode (less detailed analysis)
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )

    parser.add_argument(
        "--skip-viz", action="store_true", help="Skip visualization stage"
    )

    parser.add_argument(
        "--fast", action="store_true", help="Fast mode: use faster settings"
    )

    parser.add_argument("--quiet", action="store_true", help="Reduce logging verbosity")

    args = parser.parse_args()

    # Setup logging
    log_file = "outputs/pipeline.log"
    setup_logging(log_file=log_file, verbose=not args.quiet)

    # Print banner
    print_banner()

    # Check config file exists
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found: {args.config}")
        return 1

    print(f"Configuration: {args.config}")
    if args.skip_viz:
        print("Visualization: DISABLED")
    if args.fast:
        print("Mode: FAST")
    print()

    # Run pipeline
    success = run_pipeline(
        config_path=args.config, skip_viz=args.skip_viz, fast_mode=args.fast
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
