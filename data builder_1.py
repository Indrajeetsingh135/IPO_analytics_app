import pandas as pd
import yfinance as yf
from nsepython import nsefetch
from tqdm import tqdm
import re, time

file = "IPO-PastIssue-01-01-2015-to-13-08-2025.xlsx"
df = pd.read_excel(file)

df.columns = [str(c).strip() for c in df.columns]
if "Symbol" in df.columns and "SYMBOL" not in df.columns:
    df.rename(columns={"Symbol": "SYMBOL"}, inplace=True)

df = df.drop_duplicates(subset=["COMPANY NAME"]).reset_index(drop=True)

for c in ["NSE_SECTOR","NSE_INDUSTRY","NSE_BASIC_INDUSTRY","NSE_MACRO",
          "LOW_PRICE","HIGH_PRICE","PE","PB","EV_EBITDA","NET_PROFIT_MARGIN","ROE","DEBT_TO_EQUITY"]:
    if c not in df.columns:
        df[c] = None

if "PRICE RANGE" in df.columns:
    pat = re.compile(r"\d+(?:\.\d+)?")
    for i in tqdm(range(len(df)), desc="Price range", unit="row"):
        s = str(df.at[i,"PRICE RANGE"]).replace(",", "")
        if (pd.isna(df.at[i,"LOW_PRICE"]) or pd.isna(df.at[i,"HIGH_PRICE"])) and s and s.lower() != "nan":
            nums = pat.findall(s)
            if nums:
                a = float(nums[0]); b = float(nums[-1])
                lo, hi = (a, b) if a <= b else (b, a)
                if pd.isna(df.at[i,"LOW_PRICE"]): df.at[i,"LOW_PRICE"] = lo
                if pd.isna(df.at[i,"HIGH_PRICE"]): df.at[i,"HIGH_PRICE"] = hi

def clean_sym(s):
    return str(s).strip().upper()

def yf_sym(s):
    s = clean_sym(s)
    return s if "." in s else s + ".NS"

def get_tax(sym):
    try:
        d = nsefetch(f"https://www.nseindia.com/api/quote-equity?symbol={sym}")
        ind = d.get("industryInfo", {}) or {}
        meta = d.get("metadata", {}) or {}
        return [
            ind.get("sector") or meta.get("sector"),
            ind.get("industry") or meta.get("industry"),
            ind.get("basicIndustry") or meta.get("basicIndustry"),
            ind.get("macro") or meta.get("macro")
        ]
    except:
        return [None, None, None, None]

def get_ratios(sym):
    try:
        t = yf.Ticker(yf_sym(sym))
        info = getattr(t, "info", {}) or {}
        fin = getattr(t, "financials", pd.DataFrame())
        pe = info.get("trailingPE")
        pb = info.get("priceToBook")
        ev = info.get("enterpriseToEbitda")
        roe = info.get("returnOnEquity")
        de = info.get("debtToEquity")
        margin = None
        if not fin.empty and {"Net Income","Total Revenue"}.issubset(fin.index):
            ni = fin.loc["Net Income"].dropna()
            rv = fin.loc["Total Revenue"].dropna()
            if len(ni) > 0 and len(rv) > 0 and float(rv.iloc[0]) != 0:
                margin = float(ni.iloc[0]) / float(rv.iloc[0])
        return [pe, pb, ev, margin, roe, de]
    except:
        return [None, None, None, None, None, None]

for i in tqdm(range(len(df)), desc="NSE taxonomy", unit="row"):
    s = clean_sym(df.at[i,"SYMBOL"])
    if any(pd.isna(df.at[i,c]) for c in ["NSE_SECTOR","NSE_INDUSTRY","NSE_BASIC_INDUSTRY","NSE_MACRO"]):
        a, b, c, d = get_tax(s)
        if pd.isna(df.at[i,"NSE_SECTOR"]): df.at[i,"NSE_SECTOR"] = a
        if pd.isna(df.at[i,"NSE_INDUSTRY"]): df.at[i,"NSE_INDUSTRY"] = b
        if pd.isna(df.at[i,"NSE_BASIC_INDUSTRY"]): df.at[i,"NSE_BASIC_INDUSTRY"] = c
        if pd.isna(df.at[i,"NSE_MACRO"]): df.at[i,"NSE_MACRO"] = d
        time.sleep(0.40)

for i in tqdm(range(len(df)), desc="Yahoo ratios", unit="row"):
    s = clean_sym(df.at[i,"SYMBOL"])
    if any(pd.isna(df.at[i,c]) for c in ["PE","PB","EV_EBITDA","NET_PROFIT_MARGIN","ROE","DEBT_TO_EQUITY"]):
        pe, pb, ev, margin, roe, de = get_ratios(s)
        if pd.isna(df.at[i,"PE"]): df.at[i,"PE"] = pe
        if pd.isna(df.at[i,"PB"]): df.at[i,"PB"] = pb
        if pd.isna(df.at[i,"EV_EBITDA"]): df.at[i,"EV_EBITDA"] = ev
        if pd.isna(df.at[i,"NET_PROFIT_MARGIN"]): df.at[i,"NET_PROFIT_MARGIN"] = margin
        if pd.isna(df.at[i,"ROE"]): df.at[i,"ROE"] = roe
        if pd.isna(df.at[i,"DEBT_TO_EQUITY"]): df.at[i,"DEBT_TO_EQUITY"] = de
        time.sleep(0.10)

df.to_excel(file, index=False)
print("Saved:", file)
