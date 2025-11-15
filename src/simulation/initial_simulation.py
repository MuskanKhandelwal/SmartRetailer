import numpy as np
import pandas as pd
import pickle   # for saving/loading models
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

features_df = pd.read_parquet("panel_augmented2.parquet") #data file
model_filepath = "market_simulator_model2.joblib" #model file
demand_model = joblib.load(model_filepath)

print(features_df.columns.tolist())


model_features = [
    "lag_units_1w",
    "lag_units_2w",
    "lag_units_3w",
    "lag_units_4w",
    "rolling_mean_4w",
    "rolling_std_4w",
    "rolling_mean_8w",
    "rolling_std_8w",
    "rolling_mean_12w",
    "rolling_std_12w",
    "unit_price",
    "discount_depth",
    "price_change",
    "price_vs_ref_ratio",
    "ref_price",
    "promo_flag",
    "promo_code_encoded",
    "month",
    "quarter",
    "weekofyr",
    "is_month_start",
    "is_month_end",
    "has_special_event",
    "temp_mean",
    "temp_max",
    "precip_sum",
    "weather_missing_flag",
    "cpi_bev",
    "brand_encoded",
    "margin_pct",
    "store"
]



class PricingSimEnv:
    def __init__(self,
                 demand_model,
                 features_df,
                 upc_id,
                 week_col='week_x',
                 price_bounds=(0.5, 10.0),
                 price_change_bounds=(-0.10, +0.10),
                 noise_std=0.1):
        self.demand_model = demand_model
        self.features_df = features_df.copy()
        self.upc_id = upc_id
        self.week_col = week_col
        self.price_bounds = price_bounds
        self.price_change_bounds = price_change_bounds
        self.noise_std = noise_std

        # Precompute valid weeks list for this UPC
        df_upc = self.features_df[self.features_df["upc"] == self.upc_id]
        self.weeks_for_upc = sorted(df_upc[self.week_col].unique())
        self.reset()

    def reset(self, start_week=None):
        if start_week is None:
            self.week_idx_ptr = 0
        else:
            # find the index of start_week in weeks_for_upc, default to 0 if not found
            try:
                self.week_idx_ptr = self.weeks_for_upc.index(start_week)
            except ValueError:
                self.week_idx_ptr = 0

        self.current_week = self.weeks_for_upc[self.week_idx_ptr]
        row0 = self.features_df[
            (self.features_df["upc"] == self.upc_id) &
            (self.features_df[self.week_col] == self.current_week)
        ].iloc[0]
        self.current_price = row0["unit_price"]
        self.done = False
        return self._get_state(row0)

    def _get_state(self, row):
        state = {feat: row[feat] for feat in model_features if feat in row.index}
        state["unit_price"] = self.current_price
        state["price_change"] = 0.0
        return state

    def step(self, action):
        change_frac = np.clip(action,
                              self.price_change_bounds[0],
                              self.price_change_bounds[1])
        new_price = self.current_price * (1 + change_frac)
        new_price = float(np.clip(new_price,
                                  self.price_bounds[0],
                                  self.price_bounds[1]))

        # Use row for current week
        row = self.features_df[
            (self.features_df["upc"] == self.upc_id) &
            (self.features_df[self.week_col] == self.current_week)
        ].iloc[0]

        model_input = row.copy()
        model_input["unit_price"] = new_price
        model_input["price_change"] = change_frac
        # (Optional) Recalculate discount_depth etc here if needed.

        X = model_input[model_features].values.reshape(1, -1)
        base_units = self.demand_model.predict(X)[0]
        units_sold = base_units * (1 + np.random.normal(0, self.noise_std))
        units_sold = max(units_sold, 0.0)

        cost = row.get("cost_real", row.get("unit_cost", 0))
        profit = (new_price - cost) * units_sold
        reward = profit
        # reward = new_price * units_sold

        # Update price and move to next week in list
        self.current_price = new_price
        self.week_idx_ptr += 1

        if self.week_idx_ptr >= len(self.weeks_for_upc):
            self.done = True
            next_state = None
        else:
            self.current_week = self.weeks_for_upc[self.week_idx_ptr]
            next_row = self.features_df[
                (self.features_df["upc"] == self.upc_id) &
                (self.features_df[self.week_col] == self.current_week)
            ].iloc[0]
            next_state = self._get_state(next_row)
            self.done = False

        #info = {"units_sold": units_sold, "price": new_price}
        info = {"units_sold": units_sold, "price": new_price, "profit": profit}
        return next_state, reward, self.done, info

class SimpleRuleBasedAgent:
    def __init__(self, threshold_units=5.0, increase_price_action=0.05, decrease_price_action=-0.05):
        self.threshold_units = threshold_units
        self.increase_price_action = increase_price_action
        self.decrease_price_action = decrease_price_action
        self.last_units_sold = None

    def choose_action(self, state, info):
        # This agent uses information from the previous step (info)
        if self.last_units_sold is not None:
            if self.last_units_sold > self.threshold_units:
                # If units sold were high, increase price
                action = self.increase_price_action
            else:
                # If units sold were low, decrease price
                action = self.decrease_price_action
        else:
            # If it's the first step, take a neutral action (no price change)
            action = 0.0

        # Update the last_units_sold for the next step
        self.last_units_sold = info.get("units_sold", None)

        return action

upc_test = features_df["upc"].unique()[0]

# 2. Find its real historical price range
upc_prices = features_df[features_df["upc"] == upc_test]["unit_price"]
min_price = upc_prices.min()
max_price = upc_prices.max()

# (Optional) Add a little padding to the bounds
sim_min_price = max(0.5, min_price * 0.8) # e.g., 20% below min, but not < $0.50
sim_max_price = max_price * 1.2         # e.g., 20% above max

print(f"--- Setting up simulation for UPC: {upc_test} ---")
print(f"Historical Price Range: (${min_price:.2f}, ${max_price:.2f})")
print(f"Simulation Price Bounds: (${sim_min_price:.2f}, ${sim_max_price:.2f})")


env = PricingSimEnv(demand_model, features_df, upc_test,
                    price_bounds=(sim_min_price, sim_max_price), # <-- Use realistic bounds
                    price_change_bounds=(-0.10, +0.10),
                    noise_std=0.1)
state = env.reset()
print("Initial state:", state)

# Simulate 10 steps with random actions
random_history = []
print("\n--- Random Agent Simulation ---")
env.reset() # Reset environment for the random agent simulation
for step in range(20):
    action = np.random.choice([-0.10, -0.05, 0.0, 0.05, 0.10])
    next_state, reward, done, info = env.step(action)
    print(f"Step {step}: action {action:.2f}, price {info['price']:.2f}, reward {reward:.2f}, units_sold {info['units_sold']:.1f}")
    random_history.append({
        "step": step,
        "action": action,
        "price": info["price"],
        "reward": reward,
        "units_sold": info["units_sold"]
    })
    if done:
        print("Environment finished.")
        break

# Instantiate the agent
agent = SimpleRuleBasedAgent(threshold_units=3.0, increase_price_action=0.03, decrease_price_action=-0.03)

# Reset the environment
state = env.reset()
print("Initial state:", state)

# Simulate 20 steps using the rule-based agent
history = []
info = {} # Initialize info for the first step
for step in range(20):
    action = agent.choose_action(state, info)
    next_state, reward, done, info = env.step(action)
    print(f"Step {step}: action {action:.2f}, price {info['price']:.2f}, reward {reward:.2f}, units_sold {info['units_sold']:.1f}")
    history.append({
        "step": step,
        "action": action,
        "price": info["price"],
        "reward": reward,
        "units_sold": info["units_sold"]
    })
    if done:
        print("Environment finished.")
        break



# --- 1. Define All Agents ---
action_space = [-0.10, -0.05, 0.0, 0.05, 0.10]

# Agent 1: Random (as a class for consistency)
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    def choose_action(self, state, current_row, current_price):
        return np.random.choice(self.action_space)

# Agent 2: Rule-Based
class SimpleRuleBasedAgent:
    def __init__(self, action_space, threshold_units=5.0, increase_action=0.05, decrease_action=-0.05):
        self.action_space = action_space
        self.threshold = threshold_units
        self.increase_action = increase_action
        self.decrease_action = decrease_action
        self.last_units_sold = None

    def choose_action(self, state, current_row, current_price):
        if self.last_units_sold is None:
            action = 0.0 # Start with no change
        elif self.last_units_sold > self.threshold:
            action = self.increase_action
        else:
            action = self.decrease_action

        # Ensure the chosen action is in the allowed space
        if action not in self.action_space:
            action = 0.0 # Default to no change

        return action

    def update(self, info):
        # This agent needs to be updated with the results *after* the step
        self.last_units_sold = info.get("units_sold")

# Agent 3: Historical (for baseline)
class HistoricalAgent:
    def choose_action(self, state, current_row, current_price):
        # This agent just "chooses" the action that results
        # in the historical price. We'll just log the historical profit.
        return "HISTORICAL"

# Create the Simulation Runner
def run_simulation(env, agent):
    state = env.reset()
    total_reward = 0
    history = []

    while True:
        # Get the *full row data* for the current week
        current_row = env.features_df[
            (env.features_df["upc"] == env.upc_id) &
            (env.features_df[env.week_col] == env.current_week)
        ].iloc[0]

        if isinstance(agent, HistoricalAgent):
            # Special case: Just log the historical data
            # Use the historical unit_price and margin_pct to get cost
            hist_price = current_row["unit_price"]
            hist_margin_pct = current_row["margin_pct"]
            hist_units = current_row["units_sold"]

            # Use the official formula
            hist_unit_cost = hist_price * (1 - (hist_margin_pct / 100))
            profit = (hist_price - hist_unit_cost) * hist_units

            reward = profit
            action = 0.0 # No action, just advance time
            next_state, _, done, _ = env.step(action)
            info = {"price": hist_price, "units_sold": hist_units, "profit": profit}

        else:
            # All other agents
            action = agent.choose_action(state, current_row, env.current_price)
            next_state, reward, done, info = env.step(action)

        total_reward += reward
        history.append(info)

        # Update agents that need it
        if hasattr(agent, 'update'):
            agent.update(info)

        if done:
            break
        state = next_state

    return total_reward, pd.DataFrame(history)

# Run All Simulations
print(f"\n--- Running Full Simulations for UPC: {upc_test} ---")

# Set noise to 0.0 for a fair, repeatable comparison
env.noise_std = 0.0

# Run Historical (Baseline)
historical_agent = HistoricalAgent()
hist_profit, hist_df = run_simulation(env, historical_agent)
print(f"Total Historical (Static) Profit: ${hist_profit:,.2f}")

# Run Random Agent
random_agent = RandomAgent(action_space)
rand_profit, rand_df = run_simulation(env, random_agent)
print(f"Total Random Agent Profit: ${rand_profit:,.2f}")

# Run Rule-Based Agent
# Use the historical mean as the threshold
rule_threshold = hist_df["units_sold"].mean()
rule_agent = SimpleRuleBasedAgent(action_space, threshold_units=rule_threshold)
rule_profit, rule_df = run_simulation(env, rule_agent)
print(f"Total Rule-Based Agent Profit (Threshold={rule_threshold:.1f}): ${rule_profit:,.2f}")

# Plot The Final "Payoff" Slide
results = {
    "Historical (Static)": hist_profit,
    "Random Agent": rand_profit,
    "Rule-Based Agent": rule_profit
}

plt.figure(figsize=(10, 6))
colors = ['grey', 'lightblue', 'orange']
bars = plt.bar(results.keys(), results.values(), color=colors)
plt.title(f"Phase 1 Profit Comparison (UPC {upc_test})", fontsize=16)
plt.ylabel("Total Accumulated Profit ($)", fontsize=12)
plt.xticks(fontsize=12)

# Add labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval,
             f"${yval:,.0f}", ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()
# Add this code after your simulation runs:
print(f"Total Historical Units Sold: {hist_df['units_sold'].sum():.0f}")
print(f"Total Rule-Based Units Sold: {rule_df['units_sold'].sum():.0f}")