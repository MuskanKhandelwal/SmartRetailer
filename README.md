# SmartRetailer

> Multi-agent dynamic pricing on public retail data with a simulator seeded by real transactions.

## Project Summary

Small retailers often rely on static markups and ad-hoc promos, missing revenue when demand shifts with seasonality, inflation, or competitor moves. This project prototypes **learned pricing agents** that experiment safely in a **market simulator** and choose prices that increase revenue while respecting **fairness caps**. We benchmark against **static** and **rule-based** baselines.

This repo is tailored to **Dominick’s weekly store-level** scanner data. We build a clean soft-drinks panel from the classic files (upcsdr.csv, wsdr.csv, dominicks_weeks.csv), then will run baselines and agents. Agents try small price changes in a sandbox and keep those that help revenue without violating caps.


### Preprocessing 

This repository processes Dominick’s Finer Foods scanner data to create a **clean, enriched panel dataset** for soft drinks, integrating:

* **Product & movement data** (`upcsdr.csv`, `wsdr.csv`)
* **Macroeconomic data** from FRED (CPI for nonalcoholic beverages)
* **Weather data** (temperature & precipitation from Meteostat)
* **Store-week calendar alignment** (`dominicks_weeks.csv`)

The resulting dataset is structured for econometric or machine learning analysis of price, promotion, and demand dynamics.

---

##  Pipeline Summary

### 1️`smart_read_csv()` — Robust CSV Loader

Reads CSV files with multiple fallback encodings (`utf-8`, `ISO-8859-1`, `cp1252`), automatically skipping malformed lines.

```python
upc = smart_read_csv("upcsdr.csv")
mov = smart_read_csv("wsdr.csv")
```

---

### 2️ UPC & Movement Data Cleaning

Key computed variables:

| Column       | Description                                |
| ------------ | ------------------------------------------ |
| `unit_price` | Price per unit                             |
| `unit_cost`  | Cost per unit (computed via profit margin) |
| `margin_pct` | Gross margin percent                       |
| `revenue`    | Weekly revenue                             |
| `promo_flag` | 1 if on promotion                          |
| `brand`      | Extracted brand name                       |
| `pack`       | Package size (e.g., 12OZ, 2L, 6PK)         |

Outliers (1st–99th percentile) in price are removed, and the cleaned dataset is saved as:

```bash
softdrinks_cleaned.csv
```

---

### 3️ Add Week Timestamps

Merges with the Dominick’s week calendar to attach `start_date`, `end_date`, and `timestamp`:

```python
panel = pd.read_csv("softdrinks_cleaned.csv")
wk = pd.read_csv("dominicks_weeks.csv", parse_dates=["start","end"])
panel = panel.merge(wk.rename(columns={"start":"start_date","end":"end_date"}), on="week", how="left")
panel["timestamp"] = panel["start_date"]
panel.to_csv("softdrinks_cleaned_timestamp.csv", index=False)
```

---

### 4️ `process_data()` — CPI & Weather Augmentation

#### 🧾 CPI Data (FRED)

Fetches **nonalcoholic beverage CPI (series `CUUR0000SAF114`)** and merges by month.

Computes:

* `price_real` = inflation-adjusted price
* `cost_real` = inflation-adjusted cost

#### 🌦 Weather Data (Meteostat)

Fetches daily weather for Chicago and aggregates weekly:

* Mean temperature (`temp_mean`)
* Max temperature (`temp_max`)
* Weekly precipitation (`precip_sum`)

Merged by week with the retail panel.

#### 🧠 Feature Engineering

Adds:

* `month`, `weekofyr`
* Lagged demand (`lag_units_1w`)
* Reference price (`ref_price`)
* Weather imputation & missing flags

Saves final enriched dataset:

```bash
panel_augmented.parquet
```

---

## 📁 Output Files

| File                               | Description                         |
| ---------------------------------- | ----------------------------------- |
| `softdrinks_cleaned.csv`           | Cleaned soft drink transactions     |
| `softdrinks_cleaned_timestamp.csv` | Cleaned data with weekly timestamps |
| `panel_augmented.parquet`          | CPI- and weather-augmented dataset  |

---

## 🔑 API Keys

This script uses:

* **FRED API** → Register for a free key at [https://fred.stlouisfed.org](https://fred.stlouisfed.org)
* **Meteostat** → Works without key for public use

Store your FRED API key in an environment variable or edit the script line:

```python
FRED_API_KEY = "your_api_key_here"
```

---

## 💻 Setup & Usage

### 1️⃣ Install dependencies

```bash
pip install pandas requests meteostat pyarrow
```

### 2️⃣ Run scripts

```bash
python preprocess_softdrinks.py     # generates softdrinks_cleaned.csv
python add_timestamp.py             # merges calendar weeks
python augment_panel.py             # adds CPI & weather features
```

### 3️⃣ Output verification

All intermediate and final datasets will print head samples and dimensions to confirm correctness.

---

