import os
import streamlit as st
import lightgbm as lgb

# New mapping: asset → model filename (native format)
MODEL_FILES = {
    "Gold":   "gold_lgbm.txt",
    "Bonds":  "bonds_lgbm.txt",
    "Stocks": "stocks_lgbm.txt",
}

@st.cache_resource
def load_models(models_dir: str = "models") -> dict[str, lgb.Booster]:
    """
    Load and cache LightGBM Booster objects from text files.
    """
    models = {}
    for asset, fname in MODEL_FILES.items():
        path = os.path.join(models_dir, fname)
        if not os.path.exists(path):
            st.error(f"Model file not found: {path}")
            st.stop()
        # Load native LightGBM model
        booster = lgb.Booster(model_file=path)
        models[asset] = booster
    return models
