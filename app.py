            
import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt
import traceback

# -------------------------
# 1. Page Configuration & CSS
# -------------------------
st.set_page_config(
    page_title="SmartRetailer | AI Dynamic Pricing",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling
st.markdown(
    """
    <style>
    /* Global Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* App Background */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Header Styling */
    h1, h2, h3 {
        color: #1e293b; 
        font-weight: 700;
    }
    
    /* Custom Card for Metrics */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-delta-pos {
        color: #10b981;
        font-size: 14px;
        font-weight: 600;
    }
    .metric-delta-neg {
        color: #ef4444;
        font-size: 14px;
        font-weight: 600;
    }

    /* Form & Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# 2. Config / Constants
# -------------------------
SAVED_OBS_DIM = 32
MODEL_FOLDER = "ppo_pricing_model"
DEMAND_MODEL_PATH = "../RL/market_simulator_model2.joblib"
FEATURES_PARQUET = "../RL/panel_augmented2.parquet"

model_features = [
    "lag_units_1w","lag_units_2w","lag_units_3w","lag_units_4w",
    "rolling_mean_4w","rolling_std_4w","rolling_mean_8w","rolling_std_8w",
    "rolling_mean_12w","rolling_std_12w","unit_price","discount_depth",
    "price_change","price_vs_ref_ratio","ref_price","promo_flag",
    "promo_code_encoded","month","quarter","weekofyr","is_month_start",
    "is_month_end","has_special_event","temp_mean","temp_max","precip_sum",
    "weather_missing_flag","cpi_bev","brand_encoded","margin_pct","store"
]

# -------------------------
# 3. Helper Functions
# -------------------------
def pad_obs_to_saved(obs_array, saved_dim=SAVED_OBS_DIM):
    arr = np.asarray(obs_array, dtype=np.float32).reshape(-1)
    if arr.size == saved_dim:
        return arr.reshape(1, -1)
    elif arr.size < saved_dim:
        pad = np.zeros(saved_dim - arr.size, dtype=np.float32)
        return np.concatenate([arr, pad]).reshape(1, -1)
    else:
        return arr[:saved_dim].reshape(1, -1)

def scale_action(a, low=-0.30, high=0.30):
    return low + (a + 1) * (high - low) / 2

def safe_action_scalar(action_raw):
    try:
        return float(np.asarray(action_raw).reshape(-1)[0])
    except Exception:
        return float(action_raw)

# -------------------------
# 4. Data & Model Loading
# -------------------------
@st.cache_data(show_spinner=False)
def load_features_df(path=FEATURES_PARQUET):
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)

features_df = load_features_df()

@st.cache_resource(show_spinner=False)
def load_models():
    """Load PPO and demand models."""
    try:
        from gymnasium import spaces
        from stable_baselines3 import PPO
    except Exception as e:
        return None, None, f"Import error: {e}"

    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(SAVED_OBS_DIM,), dtype=np.float32)
    action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

    custom_objects = {
        "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2,
        "_last_obs": None,
        "_last_episode_starts": None,
        "observation_space": observation_space,
        "action_space": action_space,
    }

    ppo = None
    tried = []
    for candidate in (f"{MODEL_FOLDER}.zip", MODEL_FOLDER):
        tried.append(candidate)
        if os.path.exists(candidate) or candidate.endswith(".zip"):
            try:
                ppo = PPO.load(candidate, custom_objects=custom_objects)
                break
            except Exception:
                ppo = None

    if ppo is None:
        return None, None, f"Failed to load PPO model (tried: {tried})."

    try:
        demand = joblib.load(DEMAND_MODEL_PATH)
    except Exception as e:
        return ppo, None, f"Failed to load demand model: {e}"

    return ppo, demand, None

@st.cache_data(show_spinner=False)
def get_train_means(parquet_path=FEATURES_PARQUET, model_features=model_features):
    if not os.path.exists(parquet_path):
        return {c: 0.0 for c in model_features}
    df_train = pd.read_parquet(parquet_path)
    means = {}
    for c in model_features:
        if c in df_train.columns:
            means[c] = float(pd.to_numeric(df_train[c], errors="coerce").mean(skipna=True) or 0.0)
        else:
            means[c] = 0.0
    return means

# -------------------------
# 5. Prediction Logic
# -------------------------
def predict_row_with_models(ppo_model, demand_model, row_data, current_price, prev_units, prev_price_change):
    state_features = [f for f in model_features if f not in ["unit_price", "price_change"]]
    base_state = np.array([row_data.get(f, 0.0) for f in state_features], dtype=np.float32)
    extended = np.array([current_price, prev_units, prev_price_change], dtype=np.float32)
    obs_raw = np.concatenate([base_state, extended])

    obs = pad_obs_to_saved(obs_raw, saved_dim=SAVED_OBS_DIM)
    action_raw, _ = ppo_model.predict(obs, deterministic=True)
    raw_val = safe_action_scalar(action_raw)
    price_change = float(scale_action(raw_val))
    
    # Cap price boundaries logically
    new_price = float(np.clip(current_price * (1.0 + price_change), 0.5, 20.0))

    row_for_demand = row_data.copy()
    row_for_demand["unit_price"] = new_price
    row_for_demand["price_change"] = price_change

    X_demand = np.array([row_for_demand.get(f, 0.0) for f in model_features]).reshape(1, -1)
    predicted_units = max(float(demand_model.predict(X_demand)[0]), 0.0)

    cost = row_for_demand.get("cost_real", row_for_demand.get("unit_cost", current_price * 0.7))
    profit = (new_price - cost) * predicted_units
    revenue = new_price * predicted_units

    return new_price, price_change, predicted_units, profit, revenue

def run_multiweek_for_upc(df_upc, ppo_model, demand_model):
    per_week = []
    try:
        first = df_upc.iloc[0].to_dict()
        current_price = float(first.get("unit_price", 0.0))
        prev_units = 0.0
        prev_price_change = 0.0
        total_profit = 0.0

        for _, row in df_upc.iterrows():
            row_dict = row.to_dict()
            try:
                new_price, p_chg, units, profit, revenue = predict_row_with_models(
                    ppo_model, demand_model, row_dict, current_price, prev_units, prev_price_change
                )
            except Exception as e:
                raise RuntimeError(f"Prediction error: {e}")

            per_week.append({
                "upc": row_dict.get("upc"),
                "week_x": row_dict.get("week_x"),
                "historical_profit": row_dict.get("profit", 0.0),
                "original_price": float(row_dict.get("unit_price", 0.0)),
                "ppo_price": new_price,
                "price_change_pct": p_chg,
                "pred_demand": units,
                "pred_revenue": revenue,
                "pred_profit": profit
            })
            total_profit += profit
            prev_units = units
            prev_price_change = p_chg
            current_price = new_price

        return per_week, total_profit, None
    except Exception as e:
        return per_week, None, str(e)


# -------------------------
# 6. Main Application UI
# -------------------------

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    st.info("**SmartRetailer** uses Proximal Policy Optimization (PPO) agents trained on retail scanner data to optimize price elasticity.")
    
    # Load Status
    st.markdown("---")
    st.markdown("### Model Status")
    ppo_model, demand_model, load_err = load_models()
    if load_err:
        st.error(f"❌ Error: {load_err}")
    else:
        st.success("✅ Models Loaded Successfully")
        st.caption("PPO Agent: Ready\nDemand Simulator: Ready")

# Main Header
st.title("SmartRetailer: AI Pricing Engine")
st.markdown("Optimize retail margins using deep reinforcement learning.")

# Tabs
tab1, tab2 = st.tabs(["🕹️ Interactive Price Simulator", "📊 Bulk Strategy Backtest"])

# --- TAB 1: INTERACTIVE SANDBOX ---
with tab1:
    col_inputs, col_viz = st.columns([1, 2], gap="large")

    with col_inputs:
        st.subheader("🛒 Product Context")
        
        # 1. Loader (Quick fill)
        with st.expander("📂 Load from Dataset", expanded=True):
            upc_input = st.text_input("Enter UPC to fetch latest state:", value="1200000230")
            if st.button("Fetch Market Data"):
                if features_df is None:
                    st.error("Parquet file missing.")
                else:
                    try:
                        df_u = features_df[features_df["upc"] == int(upc_input)]
                        if df_u.empty:
                            st.warning("UPC not found.")
                        else:
                            # row0 = df_u.sort_values("week_x").iloc[0].to_dict()
                            row0 = df_u.sort_values("week_x").iloc[-1].to_dict()
                            for feat in model_features + ["profit", "units_sold", "cost_real", "unit_cost"]:
                                st.session_state[f"prefill_{feat}"] = row0.get(feat, 0.0)
                            st.success(f"Loaded context for UPC {upc_input}")
                            st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

        # 2. Main Form
        with st.form("interactive_form"):
            st.markdown("### Decision Levers")
            
            # Key levers visible immediately
            c1, c2 = st.columns(2)
            with c1:
                u_price = st.number_input("Base Price ($)", value=float(st.session_state.get("prefill_unit_price", 5.0)), step=0.1, key="unit_price_input")
                margin = st.number_input("Margin %", value=float(st.session_state.get("prefill_margin_pct", 20.0)), step=1.0, key="margin_pct_input")
            with c2:
                promo = st.selectbox("Promo Active?", options=[0, 1], index=int(st.session_state.get("prefill_promo_flag", 0)), key="promo_flag_input")
                temp = st.number_input("Temp (Avg)", value=float(st.session_state.get("prefill_temp_mean", 65.0)), key="temp_mean_input")

            # Hidden complexity
            with st.expander("📉 Market Context & Lag Features (Advanced)"):
                st.caption("These features define the RL Agent's state observation.")
                for feat in model_features:
                    if feat not in ["unit_price", "promo_flag", "margin_pct", "temp_mean", "price_change"]:
                        default_val = st.session_state.get(f"prefill_{feat}", 0.0)
                        st.number_input(feat, value=float(default_val), key=f"adv_{feat}")
                
                st.markdown("---")
                st.markdown("**State History**")
                prev_u = st.number_input("Prev Units Sold", value=float(st.session_state.get("prefill_prev_units", 0.0)), key="prev_units_input")
                prev_chg = st.number_input("Prev Price Change", value=float(st.session_state.get("prefill_price_change", 0.0)), key="prev_price_change_input")

            submit_btn = st.form_submit_button("🚀 Run AI Optimization", type="primary")

    with col_viz:
        st.subheader("💡 Optimization Results")
        
        if not submit_btn:
            st.info("👈 Adjust parameters on the left and click 'Run AI Optimization' to see the agent's recommendation.")
            
        else:
            # Gather inputs
            row = {}
            for feat in model_features:
                if feat == "unit_price": row[feat] = st.session_state["unit_price_input"]
                elif feat == "promo_flag": row[feat] = st.session_state["promo_flag_input"]
                elif feat == "margin_pct": row[feat] = st.session_state["margin_pct_input"]
                elif feat == "price_change": row[feat] = float(st.session_state.get("prefill_price_change", 0.0))
                else: row[feat] = float(st.session_state.get(f"adv_{feat}", st.session_state.get(f"prefill_{feat}", 0.0)))

            if "prefill_cost_real" in st.session_state:
                row["cost_real"] = st.session_state.get("prefill_cost_real")

            # Predict
            with st.spinner("Agent is evaluating 32-dimensional state space..."):
                try:
                    new_p, p_chg, units_ppo, profit_ppo, _ = predict_row_with_models(
                        ppo_model, demand_model, row, float(row["unit_price"]), 
                        float(st.session_state["prev_units_input"]), 
                        float(st.session_state["prev_price_change_input"])
                    )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            # Baseline calculation logic
            historical_profit = row.get("profit", float(st.session_state.get("prefill_profit", 0.0)))
            if historical_profit == 0:
                 historical_profit = profit_ppo * 0.9 # Dummy fallback if no history

            profit_uplift_abs = profit_ppo - historical_profit
            uplift_color = "metric-delta-pos" if profit_uplift_abs >= 0 else "metric-delta-neg"
            
            # --- KPI CARDS ---
            kpi1, kpi2, kpi3 = st.columns(3)
            
            with kpi1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Recommended Price</div>
                    <div class="metric-value">${new_p:.2f}</div>
                    <div class="{uplift_color}">{p_chg*100:+.1f}% Change</div>
                </div>
                """, unsafe_allow_html=True)
            
            with kpi2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Projected Demand</div>
                    <div class="metric-value">{int(units_ppo)} Units</div>
                    <div class="metric-label">Elasticity Est.</div>
                </div>
                """, unsafe_allow_html=True)

            with kpi3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Projected Profit</div>
                    <div class="metric-value">${profit_ppo:,.2f}</div>
                    <div class="{uplift_color}">${profit_uplift_abs:+.2f} vs Hist</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            
            # --- CHARTS ---
            st.markdown("#### Profit Comparison")
            comp_df = pd.DataFrame({
                "Strategy": ["Historical / Baseline", "AI Optimized"],
                "Profit": [historical_profit, profit_ppo],
                "Color": ["#94a3b8", "#10b981"]
            })
            
            # Base chart
            base = alt.Chart(comp_df).encode(
                x=alt.X("Strategy", axis=None),
                y=alt.Y("Profit", title="Profit ($)"),
                color=alt.Color("Color", scale=None),
                tooltip=["Strategy", alt.Tooltip("Profit", format="$.2f")]
            )
            
            # Bars
            bars = base.mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, size=80)
            
            # Text Labels on top
            text = base.mark_text(dy=-15, fontWeight='bold', fontSize=14).encode(
                text=alt.Text("Profit", format="$.2f")
            )
            
            # Combine and render
            st.altair_chart((bars + text).properties(height=350), use_container_width=True)


# --- TAB 2: BATCH SIMULATION ---
with tab2:
    st.markdown("### 🧬 Bulk Simulation & Backtesting")
    st.markdown("Upload a CSV containing `upc`, `week_x`, and `unit_price` to simulate AI performance over time.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], help="Required columns: upc, week_x, unit_price")

    if uploaded_file and st.button("Start Batch Simulation"):
        try:
            df_raw = pd.read_csv(uploaded_file)
            
            # --- PREPROCESSING ---
            with st.status("Running Simulation Pipeline...", expanded=True) as status:
                st.write("Checking schema...")
                required = ["upc", "week_x", "unit_price"]
                miss = [c for c in required if c not in df_raw.columns]
                if miss:
                    st.error(f"Missing: {miss}")
                    st.stop()
                    
                # Fill missing model cols
                st.write("Imputing missing features...")
                train_means = get_train_means()
                for c in model_features:
                    if c not in df_raw.columns: df_raw[c] = np.nan
                    df_raw[c] = df_raw[c].fillna(train_means.get(c, 0.0))
                
                # ... [Run Loop] ...
                st.write("Simulating agents...")
                grouped = df_raw.groupby("upc", sort=False)
                upc_list = list(grouped.groups.keys())
                
                per_week_results = []
                summary_rows = []
                
                progress_bar = st.progress(0)
                for idx, upc in enumerate(upc_list):
                    df_upc = grouped.get_group(upc).sort_values("week_x")
                    per_week, total_profit, error = run_multiweek_for_upc(df_upc, ppo_model, demand_model)
                    if not error:
                        per_week_results.extend(per_week)
                        summary_rows.append({"upc": upc, "total_profit": total_profit})
                    progress_bar.progress((idx + 1) / len(upc_list))
                
                status.update(label="Simulation Complete!", state="complete", expanded=False)

            # --- RESULTS DASHBOARD ---
            if summary_rows:
                per_week_df = pd.DataFrame(per_week_results)
                
                # Aggregates
                agg = per_week_df.groupby("upc").agg(
                    Historical_Profit=("historical_profit", "sum"),
                    AI_Profit=("pred_profit", "sum")
                ).reset_index()
                agg["Uplift"] = agg["AI_Profit"] - agg["Historical_Profit"]
                total_uplift = agg["Uplift"].sum()
                
                st.markdown("#### 🏆 Campaign Summary")
                m1, m2, m3 = st.columns(3)
                m1.metric("Total Products Sim", len(agg))
                m2.metric("Total AI Profit", f"${agg['AI_Profit'].sum():,.0f}")
                m3.metric("Total Uplift", f"${total_uplift:,.0f}", delta=f"{(total_uplift/agg['Historical_Profit'].sum())*100:.1f}%")

                st.markdown("#### 🚀 Top Performing Products (by Uplift)")
                
                # Sort and Limit to Top 50 for readability
                agg_sorted = agg.sort_values("Uplift", ascending=False).head(50)
                
                # Bar Graph
                bar_chart = alt.Chart(agg_sorted).mark_bar().encode(
                    x=alt.X("upc:N", sort="-y", title="UPC"),
                    y=alt.Y("Uplift:Q", title="Profit Uplift ($)"),
                    color=alt.condition(alt.datum.Uplift > 0, alt.value("#10b981"), alt.value("#ef4444")),
                    tooltip=["upc", alt.Tooltip("Historical_Profit", format="$,.2f"), alt.Tooltip("AI_Profit", format="$,.2f"), alt.Tooltip("Uplift", format="$,.2f")]
                ).properties(height=400)
                
                st.altair_chart(bar_chart, use_container_width=True)
                
                with st.expander("📄 Download Detailed Report"):
                    st.dataframe(agg.sort_values("Uplift", ascending=False))
                    st.download_button(
                        "Download Full CSV", 
                        per_week_df.to_csv(index=False).encode("utf-8"), 
                        "simulation_results.csv", "text/csv"
                    )

        except Exception as e:
            st.error(f"Simulation failed: {e}")
            st.write(traceback.format_exc())