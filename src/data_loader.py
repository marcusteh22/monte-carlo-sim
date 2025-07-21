import pandas as pd
import streamlit as st

@st.cache_data
def load_baseline_features(path="data/baseline_features.csv") -> pd.Series:
    """
    Load baseline feature values from CSV.
    Expects a CSV with columns:
      - raw_name
      - baseline_value
    Returns a pd.Series indexed by raw_name.
    """
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"Baseline features file not found: {path}")
        st.stop()

    # Ensure the first column is named 'raw_name'
    raw_col = df.columns[0]
    if raw_col != "raw_name":
        df = df.rename(columns={raw_col: "raw_name"})

    # Validate presence of 'baseline_value'
    if "baseline_value" not in df.columns:
        st.error("baseline_features.csv must contain a 'baseline_value' column.")
        st.stop()

    return pd.Series(df["baseline_value"].values, index=df["raw_name"])
