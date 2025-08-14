# IPO Analytics

A simple Streamlit app with two tabs: **IPO Screener** and **IPO Insights**.  
It works on one Excel file and lets you filter, look up symbols, compare companies, and see market stats.

## What it does

### 1) IPO Screener
- **Security Type** filter and **Date** filter  
  - Date: either **Issue window (From–To)** or **Listing Date (After)**  
- **NSE taxonomy** filters: Sector, Industry, Basic Industry, Macro  
- **Ratios & Returns** (nothing applied by default)  
  - For each metric choose an action:  
    - **None** (do nothing)  
    - **Between (%)** (shows a slider)  
    - **Above average / Below average**  
    - **Top 10 highest / Top 10 lowest**  
    - **Sort high→low / Sort low→high**  
- Handles mixed percent formats like `45`, `0.45`, `45%` and cleans `NaN/±∞` safely  
- Results table hides the index and can be downloaded as CSV

**Core ratios used:** `PE, PB, EV_EBITDA, NET_PROFIT_MARGIN, ROE, DEBT_TO_EQUITY`  
**Return ratios used:** `LISTING_GAIN, 1W_RETURN, 1M_RETURN, 3M_RETURN, 6M_RETURN, 1Y_RETURN, CURRENT RETURN`

### 2) IPO Insights
- **Lookup by Symbol** (exact match as in the Excel’s `SYMBOL` column)
- Shows a clean detail list: overview, taxonomy, core ratios, **returns**, 52-week high/low
- **Comparison Tool:** add multiple symbols and see the same details **side-by-side**
- **Market Stats & Trends**
  - **IPO yearly trends:** number of IPOs, average issue size/price, avg listing gain, avg current return
  - **Best/Worst sectors:** median current return by sector
  - **Heatmap:** Year × Sector median current return
  - **Counts by category:** Security Type (and Series if present)



