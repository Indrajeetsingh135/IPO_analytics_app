import pandas as pd
import time
from nsepython import nse_eq
from tqdm import tqdm

# -------- CONFIG --------
file = "IPO-PastIssue-01-01-2015-to-13-08-2025.xlsx"

# -------- LOAD --------
df = pd.read_excel(file)

# Symbol is in column D (0-based index 3)
SYM_COL = df.columns[3]

# Columns to add (Data Builder 4)
cols = [
    "TRADED_VOLUME_LAKHS",
    "TRADED_VALUE_CR",
    "TOTAL_MCAP_CR",
    "ANNUALISED_VOLATILITY",
    "WEEK52_HIGH",
    "WEEK52_HIGH_DATE",
    "WEEK52_LOW",
    "WEEK52_LOW_DATE",
]
for c in cols:
    if c not in df.columns:
        df[c] = None

def clean_sym(x):
    return str(x).strip().upper()

for i in tqdm(range(len(df)), desc="DB4: NSE trade info"):
    sym = clean_sym(df.iat[i, 3])
    if not sym or sym.lower() == "nan":
        continue
    try:
        d = nse_eq(sym)
        sec = d.get("securityInfo", {}) or {}
        pri = d.get("priceInfo", {}) or {}
        w52 = pri.get("weekHighLow", {}) or {}
        meta = d.get("metadata", {}) or {}

        vol_lakhs = sec.get("tradeVolumeInLakhs")
        val_cr    = sec.get("tradeValueInCr")
        # total mcap sometimes reported as totalMarketCap, else try metadata.mktCap
        mcap_cr   = sec.get("totalMarketCap")
        if mcap_cr is None:
            mcap_cr = meta.get("mktCap")

        vol_annual = sec.get("annualisedVolatility")

        hi_val  = w52.get("max")
        hi_date = w52.get("maxDate")
        lo_val  = w52.get("min")
        lo_date = w52.get("minDate")

        # fill only if empty (so we don't overwrite existing values)
        if pd.isna(df.at[i, "TRADED_VOLUME_LAKHS"]): df.at[i, "TRADED_VOLUME_LAKHS"] = vol_lakhs
        if pd.isna(df.at[i, "TRADED_VALUE_CR"]):     df.at[i, "TRADED_VALUE_CR"]     = val_cr
        if pd.isna(df.at[i, "TOTAL_MCAP_CR"]):       df.at[i, "TOTAL_MCAP_CR"]       = mcap_cr
        if pd.isna(df.at[i, "ANNUALISED_VOLATILITY"]): df.at[i, "ANNUALISED_VOLATILITY"] = vol_annual
        if pd.isna(df.at[i, "WEEK52_HIGH"]):         df.at[i, "WEEK52_HIGH"]         = hi_val
        if pd.isna(df.at[i, "WEEK52_HIGH_DATE"]):    df.at[i, "WEEK52_HIGH_DATE"]    = hi_date
        if pd.isna(df.at[i, "WEEK52_LOW"]):          df.at[i, "WEEK52_LOW"]          = lo_val
        if pd.isna(df.at[i, "WEEK52_LOW_DATE"]):     df.at[i, "WEEK52_LOW_DATE"]     = lo_date

        time.sleep(0.40)  # be polite to NSE
    except:
        time.sleep(0.60)
        continue

# -------- SAVE (same file) --------
df.to_excel(file, index=False)
print("Saved:", file)
