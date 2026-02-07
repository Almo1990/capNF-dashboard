import pandas as pd

# Load the data
df = pd.read_parquet("outputs/05_fouling_metrics.parquet")
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])

# Find the max decline point
max_decline_time = pd.Timestamp("2026-02-03 14:51:40")
time_24h_ago = max_decline_time - pd.Timedelta(hours=24)

print(f"Max decline at: {max_decline_time}")
print(f"Looking back to: {time_24h_ago}")

# Find permeability 24h ago
df_filtered = df[
    (df["TimeStamp"] >= time_24h_ago - pd.Timedelta(minutes=5))
    & (df["TimeStamp"] <= time_24h_ago + pd.Timedelta(minutes=5))
].sort_values("TimeStamp")

if len(df_filtered) > 0:
    print(f"\nPermeability 24h ago (around {time_24h_ago}):")
    print(df_filtered[["TimeStamp", "Permeability TC"]].head(3))

    perm_24h_ago = df_filtered["Permeability TC"].iloc[0]
    perm_current = 0.29
    decline = ((perm_24h_ago - perm_current) / perm_24h_ago) * 100

    print(f"\n" + "=" * 60)
    print("STEP-BY-STEP CALCULATION")
    print("=" * 60)
    print(f"Permeability 24h ago: {perm_24h_ago:.2f} LMH/bar")
    print(f"Permeability current: {perm_current:.2f} LMH/bar")
    print(f"\nFormula: Decline % = ((Perm_start - Perm_current) / Perm_start) × 100")
    print(
        f"Decline % = (({perm_24h_ago:.2f} - {perm_current:.2f}) / {perm_24h_ago:.2f}) × 100"
    )
    print(f"Decline % = ({perm_24h_ago - perm_current:.2f} / {perm_24h_ago:.2f}) × 100")
    print(f"Decline % = {(perm_24h_ago - perm_current) / perm_24h_ago:.4f} × 100")
    print(f"Decline % = {decline:.2f}%")
    print("=" * 60)
