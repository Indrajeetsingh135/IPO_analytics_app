import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt

st.set_page_config(page_title="IPO Analytics", layout="wide")
st.title("IPO Analytics")

DATAFILE = "IPO-PastIssue-01-01-2015-to-13-08-2025.xlsx"

def dts(s):
    x = pd.to_datetime(s, errors="coerce")
    if x.isna().mean() > 0.8 and pd.to_numeric(s, errors="coerce").notna().mean() > 0.5:
        x = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit="D", origin="1899-12-30", errors="coerce")
    return x

def fcol(df, cands):
    m = {c.lower(): c for c in df.columns}
    for c in cands:
        if c and c.lower() in m: return m[c.lower()]
    return None

def to_prop(s):
    s = s.astype(str).str.strip().str.replace('%','',regex=False)
    s = pd.to_numeric(s, errors='coerce').replace([np.inf,-np.inf], np.nan)
    s.loc[s.abs() >= 2] = s.loc[s.abs() >= 2] / 100.0
    return s

def pct_slider(label, s, default=None):
    x = s.replace([np.inf,-np.inf], np.nan).dropna()
    if x.empty:
        lo, hi = st.slider(label, -500, 500, (0,100), format="%d%%")
        return lo/100, hi/100
    lo = int(np.floor(x.min()*100)); hi = int(np.ceil(x.max()*100))
    if lo == hi: lo, hi = lo-1, hi+1
    lo = max(-500, min(500, lo)); hi = max(-500, min(500, hi))
    if default is None: default = (lo, hi)
    d0 = max(lo, min(default[0], hi)); d1 = max(lo, min(default[1], hi))
    if d0 > d1: d0, d1 = lo, hi
    a, b = st.slider(label, lo, hi, (d0, d1), format="%d%%")
    return a/100, b/100

def ratio_ui(name, col, s, key):
    opts = ["None","Between (%)","Above average","Below average","Top 10 highest","Top 10 lowest","Sort high→low","Sort low→high"]
    ch = st.selectbox(f"{name} – action", opts, index=0, key=f"{key}_a")
    s2 = s.replace([np.inf,-np.inf], np.nan)
    mask, sorti = None, None
    if ch == "Between (%)":
        lo, hi = pct_slider(f"{name} (Between, %)", s, None)
        mask = s2.between(lo, hi)
    elif ch == "Above average":
        m = s2.mean(skipna=True)
        mask = s2 >= m
        st.caption(f"Average = {(m*100):.2f}%")
    elif ch == "Below average":
        m = s2.mean(skipna=True)
        mask = s2 <= m
        st.caption(f"Average = {(m*100):.2f}%")
    elif ch == "Top 10 highest":
        idx = s2.dropna().nlargest(10).index
        mask = s2.index.isin(idx)
    elif ch == "Top 10 lowest":
        idx = s2.dropna().nsmallest(10).index
        mask = s2.index.isin(idx)
    elif ch == "Sort high→low":
        sorti = (col, False)
    elif ch == "Sort low→high":
        sorti = (col, True)
    return mask, sorti

def fdate(x):
    if pd.isna(x): return "—"
    try: return pd.to_datetime(x).date().isoformat()
    except: return str(x)

def fmoney(x):
    if pd.isna(x): return "—"
    try: return f"{float(x):,.2f}"
    except: return str(x)

def fratio(x):
    if pd.isna(x): return "—"
    try: return f"{float(x):.2f}"
    except: return str(x)

def fpct_cell(v):
    s = to_prop(pd.Series([v])).iloc[0]
    if pd.isna(s): return "—"
    return f"{s*100:.2f}%"

def render_details(row, m):
    st.markdown("**Overview**")
    ov = [
        ("Symbol", row.get(m["SYMBOL"], "—")),
        ("Security type", row.get(m["SECURITY_TYPE"], "—")),
        ("Issue price (₹)", fmoney(row.get(m["ISSUE_PRICE"])) if m["ISSUE_PRICE"] else "—"),
        ("Price range", row.get(m["PRICE_RANGE"], "—") if m["PRICE_RANGE"] else "—"),
        ("Issue start date", fdate(row.get(m["ISSUE_START"])) if m["ISSUE_START"] else "—"),
        ("Issue end date", fdate(row.get(m["ISSUE_END"])) if m["ISSUE_END"] else "—"),
        ("Listing date", fdate(row.get(m["LISTING_DATE"])) if m["LISTING_DATE"] else "—"),
    ]
    for k,v in ov: st.markdown(f"- **{k}:** {v}")
    st.markdown("**NSE Classification**")
    tax = [
        ("Sector", row.get(m["SECTOR"], "—") if m["SECTOR"] else "—"),
        ("Industry", row.get(m["INDUSTRY"], "—") if m["INDUSTRY"] else "—"),
        ("Basic industry", row.get(m["BASIC_INDUSTRY"], "—") if m["BASIC_INDUSTRY"] else "—"),
        ("Macro", row.get(m["MACRO"], "—") if m["MACRO"] else "—"),
    ]
    for k,v in tax: st.markdown(f"- **{k}:** {v}")
    st.markdown("**Core ratios**")
    cr = [
        ("P/E", fratio(row.get(m["PE"])) if m["PE"] else "—"),
        ("P/B", fratio(row.get(m["PB"])) if m["PB"] else "—"),
        ("EV/EBITDA", fratio(row.get(m["EVEBITDA"])) if m["EVEBITDA"] else "—"),
        ("Net profit margin", fpct_cell(row.get(m["NPM"])) if m["NPM"] else "—"),
        ("ROE", fpct_cell(row.get(m["ROE"])) if m["ROE"] else "—"),
        ("Debt to equity", fratio(row.get(m["DEBT_EQ"])) if m["DEBT_EQ"] else "—"),
    ]
    for k,v in cr: st.markdown(f"- **{k}:** {v}")
    st.markdown("**Returns**")
    rr = [
        ("Listing gain", fpct_cell(row.get(m["LISTING_GAIN"])) if m["LISTING_GAIN"] else "—"),
        ("1-week return", fpct_cell(row.get(m["R1W"])) if m["R1W"] else "—"),
        ("1-month return", fpct_cell(row.get(m["R1M"])) if m["R1M"] else "—"),
        ("3-month return", fpct_cell(row.get(m["R3M"])) if m["R3M"] else "—"),
        ("6-month return", fpct_cell(row.get(m["R6M"])) if m["R6M"] else "—"),
        ("1-year return", fpct_cell(row.get(m["R1Y"])) if m["R1Y"] else "—"),
        ("Current return", fpct_cell(row.get(m["RCURR"])) if m["RCURR"] else "—"),
    ]
    for k,v in rr: st.markdown(f"- **{k}:** {v}")
    st.markdown("**52-week**")
    ft = [
        ("52-week high (₹)", fmoney(row.get(m["W52H"])) if m["W52H"] else "—"),
        ("High date", fdate(row.get(m["W52H_DATE"])) if m["W52H_DATE"] else "—"),
        ("52-week low (₹)", fmoney(row.get(m["W52L"])) if m["W52L"] else "—"),
        ("Low date", fdate(row.get(m["W52L_DATE"])) if m["W52L_DATE"] else "—"),
    ]
    for k,v in ft: st.markdown(f"- **{k}:** {v}")

p = Path(DATAFILE)
if not p.exists():
    st.error(f"Data file not found: {DATAFILE}")
    st.stop()
df = pd.read_excel(p)

COL_COMPANY       = fcol(df, ["COMPANY NAME","COMPANY","NAME"])
COL_SYMBOL        = fcol(df, ["SYMBOL"])
COL_SECURITY_TYPE = fcol(df, ["SECURITY TYPE","SECURITY_TYPE"])
COL_ISSUE_PRICE   = fcol(df, ["ISSUE PRICE","ISSUE_PRICE"])
COL_PRICE_RANGE   = fcol(df, ["PRICE RANGE","PRICE_RANGE"])
COL_ISSUE_START   = fcol(df, ["ISSUE START DATE","ISSUE_START_DATE","ISSUE START"])
COL_ISSUE_END     = fcol(df, ["ISSUE END DATE","ISSUE_END_DATE","ISSUE END"])
COL_LISTING_DATE  = fcol(df, ["DATE OF LISTING","LISTING DATE"])
if COL_LISTING_DATE is None and df.shape[1] >= 20: COL_LISTING_DATE = df.columns[19]
COL_SECTOR         = fcol(df, ["NSE_SECTOR","SECTOR"])
COL_INDUSTRY       = fcol(df, ["NSE_INDUSTRY","INDUSTRY"])
COL_BASIC_INDUSTRY = fcol(df, ["NSE_BASIC_INDUSTRY","BASIC_INDUSTRY"])
COL_MACRO          = fcol(df, ["NSE_MACRO","MACRO"])
COL_SERIES         = fcol(df, ["SERIES","SERIES_NAME"])
COL_PE        = fcol(df, ["PE"])
COL_PB        = fcol(df, ["PB"])
COL_EVEBITDA  = fcol(df, ["EV_EBITDA"])
COL_NPM       = fcol(df, ["NET_PROFIT_MARGIN"])
COL_ROE       = fcol(df, ["ROE"])
COL_DEBT_EQ   = fcol(df, ["DEBT_TO_EQUITY"])
COL_LISTING_GAIN = fcol(df, ["LISTING_GAIN"])
COL_R1W          = fcol(df, ["1W_RETURN"])
COL_R1M          = fcol(df, ["1M_RETURN"])
COL_R3M          = fcol(df, ["3M_RETURN"])
COL_R6M          = fcol(df, ["6M_RETURN"])
COL_R1Y          = fcol(df, ["1Y_RETURN"])
COL_RCURR        = fcol(df, ["CURRENT RETURN","CURRENT_RETURN"])
COL_52H      = fcol(df, ["WEEK52_HIGH"])
COL_52H_DATE = fcol(df, ["WEEK52_HIGH_DATE"])
COL_52L      = fcol(df, ["WEEK52_LOW"])
COL_52L_DATE = fcol(df, ["WEEK52_LOW_DATE"])

for c in [COL_ISSUE_START,COL_ISSUE_END,COL_LISTING_DATE,COL_52H_DATE,COL_52L_DATE]:
    if c: df[c] = dts(df[c])

RATIO6_COLS = [c for c in [COL_PE,COL_PB,COL_EVEBITDA,COL_NPM,COL_ROE,COL_DEBT_EQ] if c]
RET_COLS    = [c for c in [COL_LISTING_GAIN,COL_R1W,COL_R1M,COL_R3M,COL_R6M,COL_R1Y,COL_RCURR] if c]

cols_map = {
    "SYMBOL": COL_SYMBOL, "SECURITY_TYPE": COL_SECURITY_TYPE, "ISSUE_PRICE": COL_ISSUE_PRICE, "PRICE_RANGE": COL_PRICE_RANGE,
    "ISSUE_START": COL_ISSUE_START, "ISSUE_END": COL_ISSUE_END, "LISTING_DATE": COL_LISTING_DATE,
    "SECTOR": COL_SECTOR, "INDUSTRY": COL_INDUSTRY, "BASIC_INDUSTRY": COL_BASIC_INDUSTRY, "MACRO": COL_MACRO,
    "PE": COL_PE, "PB": COL_PB, "EVEBITDA": COL_EVEBITDA, "NPM": COL_NPM, "ROE": COL_ROE, "DEBT_EQ": COL_DEBT_EQ,
    "LISTING_GAIN": COL_LISTING_GAIN, "R1W": COL_R1W, "R1M": COL_R1M, "R3M": COL_R3M, "R6M": COL_R6M, "R1Y": COL_R1Y, "RCURR": COL_RCURR,
    "W52H": COL_52H, "W52H_DATE": COL_52H_DATE, "W52L": COL_52L, "W52L_DATE": COL_52L_DATE, "SERIES": COL_SERIES
}

tab1, tab2 = st.tabs(["IPO Screener","IPO Insights"])

with tab1:
    with st.expander("Layer 1: Security Type & Date", expanded=True):
        if COL_SECURITY_TYPE and df[COL_SECURITY_TYPE].notna().any():
            sec_types = ["All"] + sorted(df[COL_SECURITY_TYPE].dropna().astype(str).unique())
            sel_sec = st.selectbox("Security Type", sec_types, index=0)
        else:
            sel_sec = "All"
        mode = st.radio("Date filter", ["Issue Window (From–To)","Listing Date (After)"], index=0 if (COL_ISSUE_START and COL_ISSUE_END) else 1, horizontal=True)
        dmask = pd.Series(True, index=df.index)
        if mode == "Issue Window (From–To)" and COL_ISSUE_START and COL_ISSUE_END:
            mn = pd.to_datetime(df[COL_ISSUE_START]).min(); mx = pd.to_datetime(df[COL_ISSUE_END]).max()
            a,b = st.columns(2)
            dfrom = a.date_input("From (Issue Start ≥)", value=mn.date() if pd.notna(mn) else None, key="i_from")
            dto   = b.date_input("To (Issue End ≤)", value=mx.date() if pd.notna(mx) else None, key="i_to")
            if dfrom: dmask &= (df[COL_ISSUE_START] >= pd.to_datetime(dfrom))
            if dto:   dmask &= (df[COL_ISSUE_END]   <= pd.to_datetime(dto) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        elif COL_LISTING_DATE:
            dafter = st.date_input("After (Listing Date ≥)", key="l_after")
            if dafter: dmask &= (df[COL_LISTING_DATE] >= pd.to_datetime(dafter))

    mask = pd.Series(True, index=df.index)
    if sel_sec != "All" and COL_SECURITY_TYPE: mask &= (df[COL_SECURITY_TYPE].astype(str) == sel_sec)
    mask &= dmask

    with st.expander("Layer 2: NSE Taxonomy", expanded=True):
        if COL_SECTOR:
            v = st.multiselect("Sector", sorted(df[COL_SECTOR].dropna().astype(str).unique()))
            if v: mask &= df[COL_SECTOR].astype(str).isin(v)
        if COL_INDUSTRY:
            v = st.multiselect("Industry", sorted(df[COL_INDUSTRY].dropna().astype(str).unique()))
            if v: mask &= df[COL_INDUSTRY].astype(str).isin(v)
        if COL_BASIC_INDUSTRY:
            v = st.multiselect("Basic Industry", sorted(df[COL_BASIC_INDUSTRY].dropna().astype(str).unique()))
            if v: mask &= df[COL_BASIC_INDUSTRY].astype(str).isin(v)
        if COL_MACRO:
            v = st.multiselect("Macro", sorted(df[COL_MACRO].dropna().astype(str).unique()))
            if v: mask &= df[COL_MACRO].astype(str).isin(v)

    sorts = []
    with st.expander("Layer 3: Core Ratios", expanded=True):
        for c in RATIO6_COLS:
            s = to_prop(df[c])
            m, sr = ratio_ui(c, c, s, f"core_{c}")
            if m is not None: mask &= m
            if sr is not None: sorts.append(sr)
    with st.expander("Layer 4: Return Ratios", expanded=True):
        for c in RET_COLS:
            s = to_prop(df[c])
            m, sr = ratio_ui(c, c, s, f"ret_{c}")
            if m is not None: mask &= m
            if sr is not None: sorts.append(sr)

    out = df[mask].copy()
    if sorts:
        by = [c for c,_ in sorts]; asc = [a for _,a in sorts]
        out = out.sort_values(by=by, ascending=asc, kind="mergesort")
    for c in (RATIO6_COLS+RET_COLS):
        if c in out.columns: out[c+"_percent"] = (to_prop(out[c])*100).round(2)
    st.success(f"Matches: {len(out):,}")
    st.dataframe(out.reset_index(drop=True), use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="ipo_screener_filtered.csv", mime="text/csv")

with tab2:
    st.subheader("Company Insights (by Symbol)")
    sym = st.text_input("Enter Symbol (exact)").strip().upper()
    sel = None
    if sym and COL_SYMBOL:
        m = df[df[COL_SYMBOL].astype(str).str.upper().str.strip() == sym]
        if m.empty: st.warning("No matching symbol found.")
        else: sel = m.iloc[0]
    if sel is not None: render_details(sel, cols_map)
    st.markdown("---")
    st.subheader("Comparison Tool")
    if "cmp" not in st.session_state: st.session_state.cmp = []
    c1,c2,c3 = st.columns([2,1,1])
    ns = c1.text_input("Add symbol", key="add_sym").strip().upper()
    if c2.button("Add"):
        if ns:
            ok = COL_SYMBOL and (df[COL_SYMBOL].astype(str).str.upper().str.strip() == ns).any()
            if not ok: st.warning(f"Symbol '{ns}' not found.")
            elif ns not in st.session_state.cmp: st.session_state.cmp.append(ns)
    if c3.button("Clear all"): st.session_state.cmp = []
    if st.session_state.cmp:
        st.caption("Comparing: " + ", ".join(st.session_state.cmp))
        rows = []
        for s in st.session_state.cmp:
            mm = df[df[COL_SYMBOL].astype(str).str.upper().str.strip() == s]
            if not mm.empty: rows.append(mm.iloc[0])
        k = len(rows); ncols = 3
        for i in range(0,k,ncols):
            cols = st.columns(min(ncols, k-i))
            for j,col in enumerate(cols):
                with col: render_details(rows[i+j], cols_map)
    st.markdown("---")
    st.subheader("Market Stats & Trends")
    if COL_LISTING_DATE: year = df[COL_LISTING_DATE].dt.year
    elif COL_ISSUE_START: year = df[COL_ISSUE_START].dt.year
    else: year = pd.Series(np.nan, index=df.index)
    stats = df.copy(); stats["YEAR"] = year
    for rc in RET_COLS: stats[rc+"_p"] = to_prop(stats[rc])
    isz = fcol(df, ["ISSUE SIZE","ISSUE_SIZE","OFFER SIZE","ISSUE AMOUNT","TOTAL ISSUE SIZE"])
    if isz: stats["_sz"] = pd.to_numeric(stats[isz], errors="coerce"); met = "Average issue size"
    else:
        stats["_sz"] = pd.to_numeric(stats[COL_ISSUE_PRICE], errors="coerce") if COL_ISSUE_PRICE else np.nan; met = "Average issue price"
    g = stats.dropna(subset=["YEAR"]).groupby("YEAR")
    yearly = pd.DataFrame({
        "Number of IPOs": g.size(),
        met: g["_sz"].mean(),
        "Avg listing gain (%)": g[COL_LISTING_GAIN+"_p"].mean()*100 if COL_LISTING_GAIN else np.nan,
        "Avg current return (%)": g[COL_RCURR+"_p"].mean()*100 if COL_RCURR else np.nan,
    }).reset_index().sort_values("YEAR")
    st.markdown("**IPO Yearly Trends**")
    st.dataframe(yearly.reset_index(drop=True), use_container_width=True, hide_index=True)
    if COL_SECTOR and COL_RCURR:
        sp = stats[[COL_SECTOR, COL_RCURR+"_p"]].dropna().groupby(COL_SECTOR)[COL_RCURR+"_p"].median().sort_values(ascending=False).reset_index()
        sp = sp.rename(columns={COL_SECTOR:"Sector", COL_RCURR+"_p":"Median current return (%)"})
        sp["Median current return (%)"] = (sp["Median current return (%)"]*100).round(2)
        c1,c2 = st.columns(2)
        with c1: st.markdown("**Top 10 sectors**"); st.dataframe(sp.head(10), use_container_width=True, hide_index=True)
        with c2: st.markdown("**Bottom 10 sectors**"); st.dataframe(sp.tail(10).sort_values("Median current return (%)"), use_container_width=True, hide_index=True)
    if COL_SECTOR and COL_RCURR:
        heat = (stats.dropna(subset=["YEAR"])
                .groupby(["YEAR", COL_SECTOR])[COL_RCURR+"_p"]
                .median().reset_index().rename(columns={COL_SECTOR:"Sector", COL_RCURR+"_p":"Median current return (%)"}))
        heat["Median current return (%)"] = heat["Median current return (%)"]*100
        ch = (alt.Chart(heat).mark_rect()
              .encode(x=alt.X("YEAR:O", title="Year"),
                      y=alt.Y("Sector:N", sort="-x"),
                      color=alt.Color("Median current return (%):Q"),
                      tooltip=["YEAR:O","Sector:N", alt.Tooltip("Median current return (%):Q", format=".2f")])
              .properties(height=400))
        st.markdown("**Year × Sector Performance (Median Current Return Heatmap)**")
        st.altair_chart(ch, use_container_width=True)
    st.markdown("**Counts by Category**")
    c1,c2 = st.columns(2)
    if COL_SECURITY_TYPE:
        cs = stats[COL_SECURITY_TYPE].value_counts(dropna=False).reset_index()
        cs.columns = ["Security type","Count"]
        with c1: st.dataframe(cs, use_container_width=True, hide_index=True)
    if COL_SERIES:
        css = stats[COL_SERIES].value_counts(dropna=False).reset_index()
        css.columns = ["Series","Count"]
        with c2: st.dataframe(css, use_container_width=True, hide_index=True)
