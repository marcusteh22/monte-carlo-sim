import pandas as pd
import streamlit as st

@st.cache_data
def load_feature_mapping(csv_path: str = "data/feature_mapping.csv") -> pd.DataFrame:
    """
    Load the full feature‑mapping table, with columns:
      - raw_name
      - asset_classes
      - friendly_label
      - ui_exposed
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Feature‑mapping file not found: {csv_path}")
        st.stop()

    required = {"raw_name", "asset_classes", "friendly_label", "ui_exposed"}
    if not required.issubset(df.columns):
        st.error(f"feature_mapping.csv must contain columns: {required}")
        st.stop()

    return df

def get_friendly_to_raw(df: pd.DataFrame) -> dict[str, str]:
    """
    Map user‑facing labels back to raw feature names.
    """
    return dict(zip(df["friendly_label"], df["raw_name"]))

def get_raw_to_friendly(df: pd.DataFrame) -> dict[str, str]:
    """
    Map raw feature names to user‑facing labels.
    """
    return dict(zip(df["raw_name"], df["friendly_label"]))

def get_ui_exposed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only those rows where ui_exposed == True.
    """
    return df[df["ui_exposed"]]
