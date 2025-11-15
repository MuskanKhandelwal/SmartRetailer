import numpy as np
import pandas as pd


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
    "store",
]


class MultiUPCEnv:
    def __init__(
        self,
        demand_model,
        features_df: pd.DataFrame,
        upc_list,
        price_change_bounds=(-0.1, 0.1),
        noise_std: float = 0.02,
        min_price: float = 0.5,
        max_price: float = 10.0,
    ):
        self.demand_model = demand_model
        self.features_df = features_df
        self.upc_list = np.array(upc_list)
        self.price_change_bounds = price_change_bounds
        self.noise_std = noise_std
        self.min_price = min_price
        self.max_price = max_price

        self.current_upc = None
        self.weeks = None
        self.week_idx_ptr = 0
        self.current_price = None
        self.done = False

    def reset(self):
        self.current_upc = np.random.choice(self.upc_list)
        df_upc = self.features_df[self.features_df["upc"] == self.current_upc]
        self.weeks = sorted(df_upc["week_x"].unique())
        self.week_idx_ptr = 0

        first_row = df_upc[df_upc["week_x"] == self.weeks[self.week_idx_ptr]].iloc[0]
        self.current_price = float(first_row["unit_price"])
        self.done = False

        return self._build_state(first_row)

    def _build_state(self, row: pd.Series) -> dict:
        state = {feat: float(row.get(feat, 0.0)) for feat in model_features}
        state["unit_price"] = float(self.current_price)
        state["price_change"] = 0.0
        return state

    def step(self, action_index: int, action_space):
        df_upc = self.features_df[self.features_df["upc"] == self.current_upc]

        change_frac = float(action_space[action_index])
        change_frac = np.clip(
            change_frac, self.price_change_bounds[0], self.price_change_bounds[1]
        )

        new_price = float(
            np.clip(
                self.current_price * (1.0 + change_frac),
                self.min_price,
                self.max_price,
            )
        )

        row = df_upc[df_upc["week_x"] == self.weeks[self.week_idx_ptr]].iloc[0]
        model_input = row.copy()
        model_input["unit_price"] = new_price
        model_input["price_change"] = change_frac

        X = model_input[model_features].values.reshape(1, -1)
        base_units = float(self.demand_model.predict(X)[0])

        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std)
            units_sold = base_units * (1.0 + noise)
        else:
            units_sold = base_units

        units_sold = max(units_sold, 0.0)

        cost = row.get("cost_real", row.get("unit_cost", 0.0))
        cost = float(cost) if cost is not None else 0.0

        profit = (new_price - cost) * units_sold

        reward = np.tanh(profit / 2000.0) * 10.0

        self.current_price = new_price
        self.week_idx_ptr += 1

        if self.week_idx_ptr >= len(self.weeks):
            self.done = True
            next_state = None
        else:
            next_row = df_upc[df_upc["week_x"] == self.weeks[self.week_idx_ptr]].iloc[0]
            next_state = self._build_state(next_row)

        info = {
            "profit": profit,
            "price": new_price,
            "units": units_sold,
            "upc": self.current_upc,
            "week_x": self.weeks[self.week_idx_ptr - 1],
        }
        return next_state, reward, self.done, info
