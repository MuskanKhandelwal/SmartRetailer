import pandas as pd
import re

def smart_read_csv(path, **kwargs):
    for enc in ("cp1252", "ISO-8859-1", "utf-8"):
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                engine="c",           # supports low_memory
                low_memory=False,     # avoids mixed dtypes
                on_bad_lines="skip",  # pandas>=1.3
                **kwargs
            )
        except UnicodeDecodeError:
            continue
        except ValueError as e:
            if "on_bad_lines" in str(e):
                return pd.read_csv(
                    path,
                    encoding=enc,
                    engine="c",
                    low_memory=False,
                    **kwargs
                )
            raise

    for enc in ("cp1252", "ISO-8859-1", "utf-8"):
        try:
            return pd.read_csv(
                path,
                encoding=enc,
                engine="python",
                on_bad_lines="skip",
                **kwargs
            )
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode with tried encodings")


upc = smart_read_csv("upcsdr.csv")
mov = smart_read_csv("wsdr.csv")


upc = upc.rename(columns=str.lower)
mov = mov.rename(columns=str.lower)

upc["upc"] = pd.to_numeric(upc["upc"], errors="coerce")
mov["upc"] = pd.to_numeric(mov["upc"], errors="coerce")


upc = upc[upc["upc"] != 179].copy()
mov = mov[mov["upc"] != 179].copy()


kw = ["COLA","SODA","SOFT","DRINK","PEPSI","COKE","COCA","SPRITE",
      "FANTA","7UP","DIET","ROOT BEER","GINGER ALE","MOUNTAIN DEW","DR PEPPER"]
mask_kw = upc["descrip"].str.upper().str.contains("|".join(kw), na=False)
upc_sdr = upc[mask_kw].copy()

df = mov.merge(upc_sdr[["upc","descrip","size","com_code"]] \
               if "com_code" in upc_sdr.columns else upc_sdr[["upc","descrip","size"]],
               on="upc", how="inner")


df["sale"] = df["sale"].replace("", pd.NA)
df = df.dropna(subset=["sale"])
df = df[(df["ok"] == 1) & (df["price"] > 0) & (df["qty"] > 0)].copy()
df["unit_price"] = df["price"] / df["qty"]                 
df["units_sold"] = df["move"].astype("int64")
df["revenue"]    = df["price"] * df["move"] / df["qty"]    # weekly revenue
# profit is a margin percentage in many Dominick’s extractions; negative is possible on deep deals
df["unit_cost"]  = df["unit_price"] * (1 - df["profit"]/100.0)
df["margin_pct"] = 100 * (1 - (df["unit_cost"] / df["unit_price"])).clip(lower=-200, upper=200)
df["promo_flag"]  = df["sale"].notna().astype(int)
df["promo_code"]  = df["sale"].fillna("")


BRANDS = ["COCA","COKE","PEPSI","SPRITE","FANTA","7UP","MOUNTAIN DEW","DR PEPPER",
          "A&W","SCHWEPPES","RC","SIERRA MIST","CRYSTAL PEPSI"]
def get_brand(s):
    s = (s or "").upper()
    for b in BRANDS:
        if b in s: return b
    return "OTHER"

def get_pack(size, desc):
    s = (size if isinstance(size,str) else "") + " " + (desc if isinstance(desc,str) else "")
    s = s.upper()
    m = re.search(r"(\d+)\s*(OZ|OZS|OZ\.)", s)
    if m: return f"{m.group(1)}OZ"
    m = re.search(r"(\d+)\s*(L|LTR|LITRE|LITER)", s)
    if m: return f"{m.group(1)}L"
    m = re.search(r"(\d+)\s*PK", s)
    if m: return f"{m.group(1)}PK"
    return None

df["brand"] = df["descrip"].apply(get_brand)
df["pack"]  = df.apply(lambda r: get_pack(r.get("size"), r.get("descrip")), axis=1)


q_lo, q_hi = df["unit_price"].quantile([0.01, 0.99])
df = df[df["unit_price"].between(q_lo, q_hi)]
panel = df[[
    "store","week","upc","descrip","brand","pack","size",
    "unit_price","unit_cost","margin_pct","units_sold","revenue",
    "promo_flag","promo_code","profit","sale"
]].sort_values(["store","upc","week"]).reset_index(drop=True)

print(panel.head())

panel.to_csv("softdrinks_cleaned.csv", index=False)