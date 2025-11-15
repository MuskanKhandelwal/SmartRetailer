import requests
import pandas as pd
from meteostat import Stations, Daily, Point
import pandas as pd


FRED_API_KEY = "a476d28e33a82a8fcfc70210cae879c8"  # get a key

def fetch_cpi_beverage():
    # series ID for nonalcoholic beverages CPI (NSA)
    SERIES = "CUUR0000SAF114"
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": SERIES,
        "api_key": FRED_API_KEY,
        "file_type": "json"
    }
    resp = requests.get(url, params=params)
    data = resp.json()["observations"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.rename(columns={"date":"date", "value":"cpi_bev"})
    return df[["date","cpi_bev"]]




def fetch_weather_weekly():
    """
    Fetch weekly weather aggregated data (temp_mean, temp_max, precip_sum)
    using Meteostat’s Point interface (interpolates from nearby stations)
    to reduce missing gaps.
    """
    # define point for Chicago (lat, lon)
    # you can also specify altitude if you know it
    point = Point(41.88, -87.62)  # lat, lon

    start = pd.to_datetime("1990-01-01")
    end   = pd.to_datetime("1997-12-31")
    daily = Daily(point, start, end).fetch().reset_index()

    # Convert to weekly by period
    daily["week"] = daily["time"].dt.to_period("W").apply(lambda r: r.start_time)
    wk = daily.groupby("week").agg(
        temp_mean   = ("tavg", "mean"),
        temp_max    = ("tmax", "max"),
        precip_sum  = ("prcp", "sum")
    ).reset_index()

    # Ensure week is datetime
    wk["week"] = pd.to_datetime(wk["week"])
    return wk



def process_data():
    print("Loading cleaned panel …")
    panel = pd.read_parquet("../softdrinks_cleaned_timestamp2.parquet")

    # ensure date columns are datetime
    panel["start_date"] = pd.to_datetime(panel["start_date"], errors="coerce")
    panel["timestamp"]  = pd.to_datetime(panel["timestamp"], errors="coerce")
    panel["panel_week"] = panel["start_date"].dt.to_period("W").apply(lambda r: r.start_time)
    print("Fetching CPI …")
    cpi = fetch_cpi_beverage()
    cpi["month"] = cpi["date"].dt.to_period("M")

    print("Merging CPI …")
    panel["month"] = panel["start_date"].dt.to_period("M")
    panel = panel.merge(cpi[["month", "cpi_bev"]], on="month", how="left")

    # deflation / real price & cost
    base = cpi.loc[cpi["date"].dt.year == 1997, "cpi_bev"].mean()
    print("Base CPI:", base)
    panel["price_real"] = panel["unit_price"] / (panel["cpi_bev"] / base)
    panel["cost_real"]  = panel["unit_cost"]  / (panel["cpi_bev"] / base)

    print("Fetching weather …")
    wk_weather = fetch_weather_weekly()
    print("Weather weeks:", len(wk_weather))

    print("=== Weather table sample weeks ===")
    print(wk_weather.head(20))

    print("=== Panel weeks sample (after computing panel_week) ===")
    print(panel["panel_week"].drop_duplicates().sort_values().head(20))

    common = set(panel["panel_week"].dropna().unique()) & set(wk_weather["week"].dropna().unique())
    print("Number of overlapping weeks:", len(common))
    print("Some overlapping weeks (first few):", list(common)[:5])

    print("Merging weather …")
    # merge matching start_date to week
    # panel = panel.merge(wk_weather, left_on="start_date", right_on="week", how="left")
    print("Merging weather using panel_week → week")
    panel = panel.merge(
        wk_weather,
        left_on="panel_week",
        right_on="week",
        how="left"
    )
    print("After merge, missing fraction temp_mean:", panel["temp_mean"].isna().mean())
    print("Feature engineering …")
    panel["month"] = panel["timestamp"].dt.month
    panel["weekofyr"] = panel["timestamp"].dt.isocalendar().week.astype(int)
    panel["lag_units_1w"] = panel.groupby(["store", "upc"])["units_sold"].shift(1)
    panel["ref_price"] = panel.groupby(["upc", "timestamp"])["price_real"].transform("median")

    # handle missing weather: impute or fill
    print("Filling missing weather …")
    panel["temp_mean"] = panel["temp_mean"].fillna(method="ffill").fillna(method="bfill")
    panel["temp_max"] = panel["temp_max"].fillna(panel["temp_mean"])
    panel["precip_sum"] = panel["precip_sum"].fillna(0.0)
    panel["weather_missing_flag"] = panel["temp_mean"].isna().astype(int)

    # define week_y properly (optional)
    panel["week_y"] = panel["start_date"].dt.to_period("W").apply(lambda r: r.start_time)

    print("Dropping nulls …")
    panel = panel.dropna(subset=["price_real", "lag_units_1w"])
    print("Rows after dropna:", len(panel))

    print("Saving augmented parquet …")
    panel.to_parquet("panel_augmented.parquet", index=False)
    print("Saved panel_augmented.parquet")
    return panel

if __name__ == "__main__":
    panel = process_data()
