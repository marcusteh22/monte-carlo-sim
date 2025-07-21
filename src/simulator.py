# simulator.py

import numpy as np
import pandas as pd
import lightgbm as lgb

def run_portfolio_monte_carlo(
    models: dict[str, lgb.Booster],
    portfolio_weights: dict[str, float],
    scenario_inputs: dict[str, float],
    baseline_features: pd.Series,
    n_sims: int = 1000,
    noise_scale: float = 0.01,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Runs a Monte Carlo simulation of one‑period asset returns for each portfolio slot.

    Arguments:
      models             – mapping asset kind (“Stocks”, “Bonds”, “Gold”) to a LightGBM Booster
      portfolio_weights  – mapping full slot label (e.g. "US Stocks") to its weight (decimal)
      scenario_inputs    – mapping raw feature name to user‑override values (only non‑zero applied)
      baseline_features  – pd.Series(raw_name → baseline_value) covering ALL model features
      n_sims             – number of Monte Carlo draws per slot
      noise_scale        – standard deviation of Gaussian noise (in decimal returns)
      random_state       – optional seed for reproducible noise

    Returns:
      DataFrame of shape (n_sims, n_slots + 1) where last column “Portfolio” is the weighted sum.
    """
    rng = np.random.default_rng(random_state)
    sim_returns: dict[str, np.ndarray] = {}

    for full_label in portfolio_weights:
        model_key = full_label.split()[-1]
        model = models.get(model_key)
        if model is None:
            # Skip any slot without a matching model
            continue

        # 1) Get the model’s feature names
        feat_names = model.feature_name()

        # 2) Build the base feature vector from full baselines (fill missing with 0)
        base_vec = baseline_features.reindex(feat_names).fillna(0.0)

        # 3) Zero out all country dummies, then one-hot this slot’s country
        for fn in feat_names:
            if fn.startswith("cat__Country_"):
                base_vec[fn] = 0.0
        country_code = full_label.split()[0]
        dummy_col = f"cat__Country_{country_code}"
        if dummy_col in base_vec.index:
            base_vec[dummy_col] = 1.0

        # 4) Apply any non-zero scenario overrides
        for raw, val in scenario_inputs.items():
            if val == 0.0:
                continue
            if raw in base_vec.index:
                base_vec[raw] = val
            elif f"num__{raw}" in base_vec.index:
                base_vec[f"num__{raw}"] = val
            else:
                # fallback: substring match
                for fn in feat_names:
                    if raw.lower() in fn.lower():
                        base_vec[fn] = val

        # 5) Tile to (n_sims × n_features) and predict
        X = np.tile(base_vec.values, (n_sims, 1))
        preds_percent = model.predict(X)  # e.g. 1.8 → 1.8%

        # 6) Convert to decimal returns and add Gaussian noise
        decimal_rets = preds_percent / 100.0
        noise = rng.normal(scale=noise_scale, size=decimal_rets.shape)
        sim_returns[full_label] = decimal_rets + noise

    # 7) Combine into DataFrame and compute portfolio return
    df = pd.DataFrame(sim_returns)
    weights = pd.Series(portfolio_weights).reindex(df.columns).fillna(0.0)
    df["Portfolio"] = df.mul(weights, axis=1).sum(axis=1)
    return df


def run_price_paths(
    model: lgb.Booster,
    scenario_inputs: dict[str, float],
    n_periods: int = 100,
    n_paths: int = 30,
    initial_price: float = 100.0,
    noise_scale: float = 0.01,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Simulate price trajectories over n_periods for one asset model,
    centering at the raw model intercept so that 0 deviations produce flat paths.

    Arguments:
      model             – LightGBM Booster for a single asset kind
      scenario_inputs   – mapping raw feature → override (applied only if non-zero)
      n_periods         – number of steps to simulate
      n_paths           – number of independent trajectories
      initial_price     – starting price for every path
      noise_scale       – noise std dev (in decimal returns)
      random_state      – optional seed for reproducible noise
    Returns:
      DataFrame of shape (n_periods+1, n_paths) with simulated price paths.
    """
    rng = np.random.default_rng(random_state)

    # 1) Determine model intercept at zero inputs
    feat_names = model.feature_name()
    zero_vec = np.zeros((1, len(feat_names)))
    intercept_percent = model.predict(zero_vec)[0]

    # 2) Build baseline feature vector of zeros + scenario overrides
    base_vec = pd.Series(0.0, index=feat_names)
    for raw, val in scenario_inputs.items():
        if val == 0.0:
            continue
        if raw in base_vec.index:
            base_vec[raw] = val
        elif f"num__{raw}" in base_vec.index:
            base_vec[f"num__{raw}"] = val
        else:
            for fn in feat_names:
                if raw.lower() in fn.lower():
                    base_vec[fn] = val

    # 3) Initialize the paths DataFrame
    paths = pd.DataFrame(
        initial_price,
        index=range(n_periods + 1),
        columns=[f"Sim {i+1}" for i in range(n_paths)]
    )

    # 4) Simulate each path
    for path in paths.columns:
        price = initial_price
        for t in range(1, n_periods + 1):
            X = base_vec.values.reshape(1, -1)
            raw_pred = model.predict(X)[0]     # percent
            # remove intercept and convert to decimal
            ret = (raw_pred - intercept_percent) / 100.0
            ret += rng.normal(scale=noise_scale)
            price *= (1 + ret)
            paths.at[t, path] = price

    return paths
