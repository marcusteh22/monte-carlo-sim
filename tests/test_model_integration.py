import sys, os
# Ensure src is on PYTHONPATH so modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pytest
import numpy as np
import pandas as pd
from model_loader import load_models
from simulator import run_portfolio_monte_carlo

# Step 1: Test that models load correctly and can be invoked

def test_load_models_return_structure():
    models = load_models(models_dir="models")
    # Models dict should contain exactly three keys
    expected_keys = {"Gold", "Bonds", "Stocks"}
    assert set(models.keys()) == expected_keys, f"Expected model keys {expected_keys}, got {set(models.keys())}"
    # Each model should be a LightGBM Booster with non-empty feature names
    for key, model in models.items():
        feat_names = model.feature_name()
        assert isinstance(feat_names, list), f"feature_name() for {key} should return a list"
        assert len(feat_names) > 0, f"Model {key} should have at least one feature name"


def test_run_monte_carlo_uses_models_exactly():
    models = load_models(models_dir="models")
    # Prepare a baseline Series of zeros for the union of all features
    all_feats = set()
    for m in models.values():
        all_feats.update(m.feature_name())
    baseline = pd.Series(0.0, index=list(all_feats))
    
    # Use only the Stocks model by naming the slot ending with 'Stocks'
    slot_label = "TEST Stocks"
    portfolio_weights = {slot_label: 1.0}
    
    # Run one sim without noise
    df = run_portfolio_monte_carlo(
        models=models,
        portfolio_weights=portfolio_weights,
        scenario_inputs={},
        baseline_features=baseline,
        n_sims=1,
        noise_scale=0.0,
    )
    # Expect a single row and two columns: 'TEST Stocks' and 'Portfolio'
    assert df.shape == (1, 2)
    assert slot_label in df.columns, f"Column {slot_label} missing in output"
    assert "Portfolio" in df.columns, "Column 'Portfolio' missing in output"
    
    # The value should equal the model prediction at zero inputs
    model_stock = models["Stocks"]
    zero_vec = np.zeros((1, len(model_stock.feature_name())))
    raw_expected = model_stock.predict(zero_vec)[0]
    # Simulator converts to decimal returns (percent/100)
    expected = raw_expected / 100.0
    # Since noise_scale=0, sim_returns == preds, so decimal conversion may depend on implementation
    # Here the simulator returns in percent units, so check equality
    result = df[slot_label].iloc[0]
    assert pytest.approx(expected, rel=1e-6) == result, \
        f"Expected model.predict zero input {expected}, got {result}"
    # Portfolio column should match same value
    assert pytest.approx(result, rel=1e-6) == df["Portfolio"].iloc[0], \
        "Portfolio column should equal the slot return when weight=1.0"
