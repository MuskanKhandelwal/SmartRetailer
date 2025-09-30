import pandas as pd
panel = pd.read_csv("softdrinks_cleaned.csv")
wk = pd.read_csv("dominicks_weeks.csv", parse_dates=["start","end"])
panel = panel.merge(wk.rename(columns={"start":"start_date","end":"end_date"}), on="week", how="left")
panel["timestamp"] = panel["start_date"]

panel.to_csv("softdrinks_cleaned_timestamp.csv", index=False)