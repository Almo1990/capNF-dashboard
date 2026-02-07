import pandas as pd
import json

print("=" * 80)
print("PERMEABILITY DECLINE CALCULATION - NEW BASELINE METHOD")
print("=" * 80)

# Load the new data with baseline calculation
df = pd.read_parquet("outputs/05_fouling_metrics.parquet")
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])

# Load metadata
with open("outputs/fouling_metadata.json", "r") as f:
    metadata = json.load(f)

print("\n📊 BASELINE CALCULATION:")
print("-" * 80)
print(f"Baseline period: January 28, 2026 15:00 to February 4, 2026 15:00 (7 days)")
print(f"Method: Average of most stable period (lowest slope) within first week")
print(f"\nBaseline Permeability (SMA): {metadata['permeability_baseline']:.2f} LMH/bar")
print(f"Current Permeability (SMA):  {df['Permeability TC_SMA'].iloc[-1]:.2f} LMH/bar")
print(f"\nDecline from Baseline: {metadata['permeability_decline_from_baseline']:.2f}%")

print("\n" + "=" * 80)
print("COMPARISON: OLD vs NEW METHOD")
print("=" * 80)

print("\n❌ OLD METHOD (24h rolling window):")
print("   - Compared current permeability to 24 hours ago")
print("   - Result: 96.5% decline (8.33 → 0.29 LMH/bar)")
print("   - Issue: Captured transient drops, not true baseline performance")

print("\n✅ NEW METHOD (First-week baseline):")
print(f"   - Compared current to stable baseline from week 1")
print(
    f"   - Result: {metadata['permeability_decline_from_baseline']:.2f}% decline ({metadata['permeability_baseline']:.2f} → {df['Permeability TC_SMA'].iloc[-1]:.2f} LMH/bar)"
)
print("   - Benefit: True performance degradation vs initial condition")

print("\n" + "=" * 80)
print("ALERT STATUS")
print("=" * 80)

# Load alerts
with open("outputs/07_alerts.json", "r") as f:
    alerts = json.load(f)

decline_value = metadata["permeability_decline_from_baseline"]
if decline_value >= 20:
    print(f"🚨 CRITICAL: {decline_value:.1f}% ≥ 20% threshold")
elif decline_value >= 10:
    print(f"⚠️  WARNING: {decline_value:.1f}% ≥ 10% threshold")
else:
    print(f"✅ OK: {decline_value:.1f}% < 10% threshold (no alert)")

print(f"\nTotal alerts: {len(alerts)}")
print("=" * 80)
