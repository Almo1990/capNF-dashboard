"""Quick script to regenerate dashboard only (skip data processing stages)"""

import sys
import yaml
from src.dashboard_app import run_dashboard_app

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

print("=" * 70)
print("  REGENERATING DASHBOARD WITH DAILY BASELINE PERMEABILITY FORECAST")
print("=" * 70)
print()

try:
    viz_files = run_dashboard_app("config.yaml")
    print("\n" + "=" * 70)
    print("✅ DASHBOARD REGENERATION COMPLETE")
    print("=" * 70)
    print(f"\nMain dashboard: {viz_files.get('unified_dashboard', 'N/A')}")
    print(f"Permeability forecast: combined_data_plots/Permeability_TC_forecast.html")
    print()
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
