import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
import pandas as pd
from data_loader import load_baseline_features
from model_loader import load_models


def test_baseline_loader_series_structure(tmp_path):
    # Create a temporary CSV file with two features
    csv = tmp_path / "b.csv"
    csv.write_text("raw_name,baseline_value\nfoo,1.2\nbar,-0.5\n")

    series = load_baseline_features(path=str(csv))
    assert isinstance(series, pd.Series)
    assert list(series.index) == ["foo", "bar"]
    assert list(series.values) == [1.2, -0.5]


def test_baseline_loader_covers_model_features():
    # Load real baselines
    baselines = load_baseline_features(path="data/baseline_features.csv")
    # Determine required features from all models
    models = load_models(models_dir="models")
    required = set()
    for m in models.values():
        required.update(m.feature_name())

    missing = required - set(baselines.index)
    assert not missing, f"Missing baseline values for features: {missing}"


def test_known_baseline_values():
    baselines = load_baseline_features(path="data/baseline_features.csv")
    # Spot-check known values
    assert pytest.approx(1.2321279948382504, rel=1e-9) == baselines['num__USD_GBP_GBR_ytd']
    assert pytest.approx(-0.4731426957301356, rel=1e-9) == baselines['num__USD_EUR_DEU_ytd']
    assert pytest.approx(3.7703322807812505, rel=1e-9) == baselines['num__CPI_Inflation_USA_pct']
