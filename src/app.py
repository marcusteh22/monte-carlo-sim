import pandas as pd
from model_loader import load_models


import streamlit as st
import pandas as pd
from model_loader import load_models
from simulator import run_portfolio_monte_carlo, run_price_paths


from data_loader import load_baseline_features
from feature_mapping import (
    load_feature_mapping,
    get_ui_exposed,
    get_raw_to_friendly,
)

# ---------------------
# Global configuration & constants
# ---------------------
st.set_page_config(
    page_title="Geo‑Macro Monte Carlo",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Define these once so sidebar & main can both use them
countries   = ["US","CAN","DEU","FRA","GBR","ITA","JPN","CHN"]
asset_kinds = ["Bonds","Stocks","Gold"]
asset_options = ["Select an asset…"] + [
    f"{c} {k}" for c in countries for k in asset_kinds
]

st.title("📈 Monte Carlo Simulator")
st.text("Modelling Macroeconomics & Geopolitical Dynamics ")

# ---------------------
# Sidebar: Scenario Configuration
# ---------------------
st.sidebar.markdown("## Scenario Configuration")
country = st.sidebar.selectbox("Focus country", countries)

# Load data & mapping
baselines    = load_baseline_features()      # raw_name → baseline_value (now empty for 0 defaults)
df_map       = load_feature_mapping()        # full mapping table
ui_map       = get_ui_exposed(df_map)        # only ui_exposed rows
raw2friendly = get_raw_to_friendly(df_map)   # raw_name → friendly_label


# ===== Composite filter for generic + country-specific indicators =====
# Build a pattern matching any country suffix (_US_, _CAN_, … or ending _US, _CAN, …)
any_country_pattern = "|".join(
    [f"_{c}_" for c in countries] + [f"_{c}$" for c in countries]
)

# 1) Generic indicators (no country suffix at all)
generic_map = ui_map[~ui_map["raw_name"].str.contains(any_country_pattern)]

# 2) Country-specific indicators for the chosen country
country_pattern = f"_{country}_|_{country}$"
country_specific = ui_map[ui_map["raw_name"].str.contains(country_pattern)]

# Combine generic + country-specific into the final nine features
country_map = pd.concat([generic_map, country_specific], ignore_index=True)
# Create a mapping friendly_label → raw_name, prioritizing country-specific rows
friendly_to_raw = {}
# First add generic (if any) so country rows overwrite them
for _, row in ui_map[~ui_map["raw_name"].str.contains(f"_{country}_|_{country}$")].iterrows():
    friendly_to_raw[row["friendly_label"]] = row["raw_name"]
# Then add only the country-specific ones
for _, row in country_map.iterrows():
    friendly_to_raw[row["friendly_label"]] = row["raw_name"]

# Now friendly_to_raw has exactly one raw_name per friendly_label


# ---------------------
# Sidebar: Indicator Inputs
# ---------------------
st.sidebar.markdown("### Indicator Inputs")
scenario_inputs = {}
for label, raw in friendly_to_raw.items():
    val = st.sidebar.number_input(
        f"{label}",
        value=0.0,
        key=f"ni_{raw}"
    )
    scenario_inputs[raw] = val



# ---------------------
# Main panel: Portfolio Allocation 
# ---------------------
st.header("Portfolio Allocation")
NUM_SLOTS = 6  # updated

portfolio   = {}
total_alloc = 0.0

for i in range(1, NUM_SLOTS + 1):
    # Two columns: 4 units for dropdown, 1 unit for percentage
    col_asset, col_pct = st.columns([4, 1], gap="small")
    with col_asset:
        asset = st.selectbox(f"Asset {i}", asset_options, key=f"asset_{i}")
    with col_pct:
        pct = st.number_input(
            "",  # no label (we'll show % in header)
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            format="%.1f",
            key=f"wt_{i}"
        )
        st.markdown("%")  # label separator

    if asset != "Select an asset…":
        portfolio[asset] = pct
        total_alloc    += pct

st.markdown(f"**Total:** {total_alloc:.1f}%")
if abs(total_alloc - 100.0) > 1e-6:
    st.error("Portfolio weights must sum to 100%.")
    run_disabled = True
else:
    run_disabled = False

# ---------------------
# Sidebar: Run Button 
# ---------------------
st.sidebar.markdown("---")
run = st.sidebar.button("Run Monte Carlo", disabled=run_disabled)

# ---------------------
# Main panel: Simulation trigger/message
# ---------------------
# ---------------------
# Main panel: Simulation trigger/message
# ---------------------
if run:
    st.write("🔄 Running Monte Carlo price paths…")

    # ——————————————————————————————————————————————————————
    # 1) Build and run one‑period Monte Carlo returns
    models = load_models(models_dir="models")
    current_wts_simple = {
        lbl: pct/100.0
        for lbl, pct in portfolio.items()
        if lbl != "Select an asset…"
    }

    # Calculate baseline (no adjustments) for comparison if adjustments were made
    changed_indicators = {k: v for k, v in scenario_inputs.items() if v != 0.0}
    baseline_scenario = {}  # Empty scenario for baseline
    
    # Import numpy early for calculations
    import numpy as np
    
    if changed_indicators:
        # Calculate baseline returns for comparison
        df_ret_baseline = run_portfolio_monte_carlo(
            models=models,
            portfolio_weights=current_wts_simple,
            scenario_inputs=baseline_scenario,  # No adjustments
            baseline_features=baselines,
            n_sims=500,
            noise_scale=0.02,
            random_state=42
        ).drop(columns="Portfolio")
        
        means_baseline = df_ret_baseline.mean()
        exp_means_baseline = np.exp(means_baseline * 5)
        target_wts_baseline = exp_means_baseline / exp_means_baseline.sum()

    # Calculate current scenario returns
    df_ret = run_portfolio_monte_carlo(
        models=models,
        portfolio_weights=current_wts_simple,  # e.g. {"US Stocks":0.2, "FRA Stocks":0.1, …}
        scenario_inputs=scenario_inputs,
        baseline_features=baselines,
        n_sims=500,
        noise_scale=0.02,
        random_state=42  # Add fixed seed for consistent results
    ).drop(columns="Portfolio")

    # ——————————————————————————————————————————————————————
    # 2) Optimize portfolio allocation - use exponential weighting
    means = df_ret.mean()

    # Handle edge cases
    if len(means) == 0:
        target_wts = pd.Series([], dtype=float)
    else:
        # Use exponential weighting to amplify differences (numpy already imported above)
        exp_means = np.exp(means * 5)  # Amplify differences 5x
        target_wts = exp_means / exp_means.sum()
        


    # ——————————————————————————————————————————————————————
    # 3) Build recommendation rows, splitting duplicates
    # 3a) Count how many labels map to each model_key
    rec_rows = []
    for full_label, curr_pct in portfolio.items():
        if full_label == "Select an asset…":
            continue
        # Pull the target weight for this exact slot
        targ_frac = float(target_wts.get(full_label, 0.0))
        sugg_pct  = round(100 * targ_frac, 1)
        delta_pct = round(sugg_pct - curr_pct, 1)

        rec_rows.append({
            "Asset":        full_label,
            "Current %":    round(curr_pct, 1),
            "Suggested %":  sugg_pct,
            "Δ %":          delta_pct,
        })


    # ─── For each selected asset, plot price paths ───
    for asset_label in portfolio.keys():
        if asset_label == "Select an asset…":
            continue
        model_key = asset_label.split()[-1]
        model = models.get(model_key)
        if model is None:
            st.error(f"No model for '{model_key}'")
            continue

        st.subheader(f"{asset_label} Price Paths")
        df_paths = run_price_paths(
            model=model,
            scenario_inputs=scenario_inputs,
            n_periods=1,  # Single period to match optimization
            n_paths=100,
            initial_price=100.0,
            noise_scale=0.05
        )
        st.line_chart(df_paths)

        # Final‐price summary (now single-period)
        final_prices = df_paths.iloc[-1]
        st.markdown(f"**Final Price Summary for {asset_label}**")
        summary = pd.DataFrame({
            "Mean":   [final_prices.mean()],
            "5th %":  [final_prices.quantile(0.05)],
            "95th %": [final_prices.quantile(0.95)]
        }, index=[asset_label])
        st.table(summary)

    # ─── Show Recommendation ───
    st.write("")
    st.write("")
    st.write("")
    st.header("⚖️ Rebalance Recommendation")
    df_rec = pd.DataFrame(rec_rows).set_index("Asset")
    st.table(df_rec)

    # ——————————————————————————————————————————————————————
    # XAI: Simple Explanation of Allocation Logic
    # ——————————————————————————————————————————————————————
    st.write("")
    st.write("")
    st.write("")
    st.header("🧠 Why These Allocations?")
    
    if len(means) > 0:
        # Detect which scenario we're in
        changed_indicators = {k: v for k, v in scenario_inputs.items() if v != 0.0}
        
        # 1. Context Setting with Country and Scenario
        st.subheader("**Context:**")
        if changed_indicators:
            # Scenario 2: Some indicators changed
            st.write(f"Predictions for **{country}-focused** portfolio under **your adjusted conditions:**")
            
            # Show what user changed
            st.write("**Your adjustments:**")
            for raw_name, value in changed_indicators.items():
                # Convert raw name to friendly name
                friendly_name = next((label for label, raw in friendly_to_raw.items() if raw == raw_name), raw_name)
                st.write(f"• {friendly_name}: {value:+.2f}")
            
            st.write("• Other macro/geopolitical factors: baseline conditions")
            st.write("")
            st.write("")
            st.write("")

            # ——————————————————————————————————————————————————————
            # NEW: Impact Analysis Section
            # ——————————————————————————————————————————————————————
            st.write("")
            st.subheader("**Impact of Your Adjustments:**")
            st.write("*How your changes affected each asset:*")
            
            # Sort by current performance for consistent display
            sorted_assets = means.sort_values(ascending=False)
            
            for asset in sorted_assets.index:
                if asset in means_baseline.index:  # Make sure asset exists in baseline
                    # Calculate changes
                    baseline_return = means_baseline[asset] * 100
                    current_return = means[asset] * 100
                    return_change = current_return - baseline_return
                    
                    baseline_allocation = target_wts_baseline.get(asset, 0) * 100
                    current_allocation = target_wts.get(asset, 0) * 100
                    allocation_change = current_allocation - baseline_allocation
                    
                    # Format the impact message
                    if abs(return_change) < 0.05:  # Very small change
                        return_impact = "minimal change"
                        return_symbol = "➖"
                    elif return_change > 0:
                        return_impact = f"+{return_change:.1f}%"
                        return_symbol = "📈"
                    else:
                        return_impact = f"{return_change:.1f}%"
                        return_symbol = "📉"
                    
                    if abs(allocation_change) < 0.1:  # Very small allocation change
                        allocation_impact = "unchanged"
                        allocation_symbol = "➖"
                    elif allocation_change > 0:
                        allocation_impact = f"+{allocation_change:.1f}%"
                        allocation_symbol = "⬆️"
                    else:
                        allocation_impact = f"{allocation_change:.1f}%"
                        allocation_symbol = "⬇️"
                    
                    st.write(f"• **{asset}** {return_symbol}")
                    st.write(f"  - Expected return: {baseline_return:+.1f}% → {current_return:+.1f}% ({return_impact})")
                    st.write(f"  - Allocation: {baseline_allocation:.1f}% → {current_allocation:.1f}% ({allocation_impact}) {allocation_symbol}")
                    st.write("")
                    st.write("")
                    st.write("")
            
        else:
            # Scenario 1: No indicators touched
            st.write(f"Predictions for **{country}-focused** portfolio under **baseline macro/geopolitical conditions:**")
            st.write("• All indicators set to historical/default values")
            st.write("• You can adjust indicators in the sidebar to see how changing conditions affect allocations")
        
        st.write("")
        st.write("")
        st.write("")
        
        # 2. Performance Ranking
        sorted_assets = means.sort_values(ascending=False)
        st.subheader("**Predicted Performance Ranking:**")
        
        ranking_text = ""
        for i, (asset, return_val) in enumerate(sorted_assets.items()):
            if i > 0:
                ranking_text += " > "
            ranking_text += f"{asset} ({return_val*100:+.1f}%)"
        
        st.write(ranking_text)
        st.write("")
        st.write("")
        st.write("")
        
        # 3. Why These Specific Percentages
        st.subheader("**Breakdown:**")
        sorted_weights = target_wts.reindex(sorted_assets.index)
        
        for asset, weight in sorted_weights.items():
            return_pct = means[asset] * 100
            weight_pct = weight * 100
            
            if return_pct > 0:
                explanation = f"gets {weight_pct:.1f}% allocation because of its {return_pct:+.1f}% expected return"
            else:
                explanation = f"gets only {weight_pct:.1f}% allocation due to its {return_pct:+.1f}% expected loss"
            
            st.write(f"• **{asset}** {explanation}")
    
    else:
        st.write("*No assets selected for analysis*")

else:
    st.write("Configure and click ▶ Run Monte Carlo.")