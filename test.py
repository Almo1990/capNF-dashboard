import pandas as pd

# Load raw data (before filtering)
# Load a specific TSV file
df_june = pd.read_csv("Data/2024-08-02_08-12-46 DataLog.tsv", sep="\t")
print(df_june["Permeability TC"].describe())
print(df_june["Permeability TC"].describe())
# Export visualization data to CSV
df_june.to_csv("outputs/01_raw.parquet.csv", index=False)
print("✓ Saved visualization data to CSV")
