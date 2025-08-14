import pandas as pd
import yfinance as yf
from tqdm import tqdm

file = "IPO-PastIssue-01-01-2015-to-13-08-2025.xlsx"
df = pd.read_excel(file)

m = {c.strip().replace(" ","").upper(): c for c in df.columns}
sym_col = df.columns[3]

want = ["DATEOFLISTING","DATE1W","DATE1M","DATE3M","DATE6M","DATE1Y"]
avail = [m.get(k) for k in want]

for c in ["PRICE_LISTING","PRICE_1W","PRICE_1M","PRICE_3M","PRICE_6M","PRICE_1Y","PRICE_TODAY"]:
    if c not in df.columns:
        df[c] = None

def norm_sym(s):
    s = str(s).strip().upper()
    return s if "." in s else s + ".NS"

def parse_date(x):
    return pd.to_datetime(x, errors="coerce", dayfirst=False)

def price_on_or_after(tkr, dt):
    if pd.isna(dt): return None
    try:
        dt = parse_date(dt)
        start = (dt - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
        end   = (dt + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        h = yf.download(tkr, start=start, end=end, interval="1d", auto_adjust=True, progress=False)
        if h is None or h.empty: return None
        h = h.reset_index()
        h["Date"] = pd.to_datetime(h["Date"]).dt.normalize()
        target = dt.normalize()
        row = h[h["Date"] >= target]
        if row.empty:
            row = h[h["Date"] <= target]
            if row.empty: return None
        return float(row.iloc[0]["Close"])
    except:
        return None

def price_today(tkr):
    try:
        t = yf.Ticker(tkr)
        info = getattr(t, "info", {}) or {}
        p = info.get("currentPrice")
        if p: return float(p)
        h = t.history(period="5d", auto_adjust=True)
        if not h.empty: return float(h["Close"].iloc[-1])
        return None
    except:
        return None

for i in tqdm(range(len(df)), desc="Fetching prices"):
    tkr = norm_sym(df.iat[i, 3])

    d_list = parse_date(df.at[i, avail[0]]) if avail[0] in df.columns else pd.NaT
    d_1w   = parse_date(df.at[i, avail[1]]) if avail[1] in df.columns else pd.NaT
    d_1m   = parse_date(df.at[i, avail[2]]) if avail[2] in df.columns else pd.NaT
    d_3m   = parse_date(df.at[i, avail[3]]) if avail[3] in df.columns else pd.NaT
    d_6m   = parse_date(df.at[i, avail[4]]) if avail[4] in df.columns else pd.NaT
    d_1y   = parse_date(df.at[i, avail[5]]) if avail[5] in df.columns else pd.NaT

    df.at[i,"PRICE_LISTING"] = price_on_or_after(tkr, d_list)
    df.at[i,"PRICE_1W"]      = price_on_or_after(tkr, d_1w)
    df.at[i,"PRICE_1M"]      = price_on_or_after(tkr, d_1m)
    df.at[i,"PRICE_3M"]      = price_on_or_after(tkr, d_3m)
    df.at[i,"PRICE_6M"]      = price_on_or_after(tkr, d_6m)
    df.at[i,"PRICE_1Y"]      = price_on_or_after(tkr, d_1y)
    df.at[i,"PRICE_TODAY"]   = price_today(tkr)

df.to_excel(file, index=False)
print("Saved:", file)
