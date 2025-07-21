import pandas as pd
from src.model_loader import load_models

# Paths
path = "data/baseline_features.csv"
df = pd.read_csv(path)
have = set(df["raw_name"])

models = load_models(models_dir="models")
needed = set()
for m in models.values():
    needed.update(m.feature_name())

missing = sorted(needed - have)

# Create rows for missing with default 0.0
df_missing = pd.DataFrame({
    "raw_name": missing,
    "baseline_value": [0.0] * len(missing)
})

# Combine, remove duplicates, sort, and save
df_full = pd.concat([df, df_missing], ignore_index=True)
df_full = df_full.drop_duplicates(subset="raw_name").sort_values("raw_name")
df_full.to_csv(path, index=False)

print(f"Appended {len(missing)} missing features to {path}.")
