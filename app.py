import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from live_data import fetch as fetch_live

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Structured Products & Options Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── SESSION STATE (passage de données entre onglets) ─────────────────────────
for key, default in [("bs_spot", 100.0), ("bs_sigma", 20.0), ("bs_ticker", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

#MainMenu, footer, header[data-testid="stHeader"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"]  { display: none !important; }

html, body, .stApp {
    background: #eef3f8 !important;
    font-family: Inter, Arial, sans-serif !important;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}

/* ── Tabs → dark navbar ── */
.stTabs > div:first-child { background: #0d1b2a; }
div[data-baseweb="tab-list"] {
    background: #0b1b2a !important;
    padding: 0 40px !important;
    gap: 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    overflow-x: auto !important;
}
button[data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(255,255,255,0.5) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1.3px !important;
    padding: 16px 20px !important;
    border-radius: 0 !important;
    border: none !important;
    white-space: nowrap !important;
}
button[data-baseweb="tab"]:hover { color: rgba(255,255,255,0.8) !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: white !important; }
div[data-baseweb="tab-highlight"] { background-color: white !important; height: 2px !important; }
div[data-baseweb="tab-border"]    { display: none !important; }
div[data-baseweb="tab-panel"]     { padding: 0 !important; background: #eef3f8 !important; }

div[data-testid="stMetric"],
div[data-testid="stDataFrame"],
div[data-testid="stPlotlyChart"] {
    border-radius: 10px !important;
}

/* ── Inputs ── */
.stNumberInput label, .stSelectbox label, .stTextInput label, .stToggle label {
    font-size: 13px !important; font-weight: 600 !important; color: #374151 !important;
}

.stButton button {
    border-radius: 999px !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
}

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #eef3f8; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── FINANCIAL FUNCTIONS ───────────────────────────────────────────────────────

def bs_d1_d2(S, K, T, sigma, r, q=0.0):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None, None
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)

def bs_price(S, K, T, sigma, r, q=0.0, otype="Call"):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    if d1 is None:
        return 0.0
    if otype == "Call":
        return S * np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)

def bs_greeks(S, K, T, sigma, r, q=0.0, otype="Call"):
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)
    if d1 is None:
        return {k: 0.0 for k in ["Delta","Gamma","Vega","Theta","Rho"]}
    pdf1 = norm.pdf(d1); sqT = np.sqrt(T)
    gamma = np.exp(-q*T)*pdf1/(S*sigma*sqT)
    vega  = S*np.exp(-q*T)*pdf1*sqT/100
    if otype == "Call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        theta = (-np.exp(-q*T)*S*pdf1*sigma/(2*sqT) - r*K*np.exp(-r*T)*norm.cdf(d2)  + q*S*np.exp(-q*T)*norm.cdf(d1))/365
        rho   = K*T*np.exp(-r*T)*norm.cdf(d2)/100
    else:
        delta = -np.exp(-q*T)*norm.cdf(-d1)
        theta = (-np.exp(-q*T)*S*pdf1*sigma/(2*sqT) + r*K*np.exp(-r*T)*norm.cdf(-d2) - q*S*np.exp(-q*T)*norm.cdf(-d1))/365
        rho   = -K*T*np.exp(-r*T)*norm.cdf(-d2)/100
    return {"Delta":delta,"Gamma":gamma,"Vega":vega,"Theta":theta,"Rho":rho}

# ── YFINANCE FUNCTIONS ────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data(ticker: str):
    """
    Market-data loader aligned with the working project.
    It uses live_data.fetch(), which returns Yahoo Finance data plus hist/returns.
    """
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker vide.")

    data = fetch_live(ticker, period="1y")

    if not data:
        raise ValueError(
            f"Impossible de charger '{ticker}'. "
            "Copie bien live_data.py dans le même dossier que app.py, "
            "puis lance avec le même environnement Python que le projet qui marche."
        )

    hist = data.get("hist")
    if hist is None or hist.empty or len(hist) < 5:
        raise ValueError(f"'{ticker}' a été trouvé mais l'historique Yahoo est vide.")

    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)

    hist = hist.dropna(subset=["Close"]).copy()
    if hist.empty or len(hist) < 5:
        raise ValueError(f"Historique inexploitable pour '{ticker}'.")

    return hist, {
        "name": data.get("ticker", ticker),
        "price": float(data["price"]),
        "prev_close": float(data["prev_close"]),
        "mkt_cap": float(data.get("market_cap") or 0.0),
        "high_52w": float(data.get("high_52w") or hist["High"].max()),
        "low_52w": float(data.get("low_52w") or hist["Low"].min()),
        "currency": data.get("currency") or "USD",
    }

def calc_hist_vols(hist: pd.DataFrame):
    """Return log returns and annualised historical vols (30/60/90/252d)."""
    log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    vols = {}
    for w in [30, 60, 90, 252]:
        tail = log_ret.tail(w)
        vols[f"{w}d"] = float(tail.std() * np.sqrt(252) * 100) if len(tail) >= 5 else None
    rolling_30 = log_ret.rolling(30).std() * np.sqrt(252) * 100
    return log_ret, rolling_30, vols

@st.cache_data(ttl=900, show_spinner=False)
def fetch_option_surface(ticker: str, max_expiries: int = 8):
    """Load Yahoo option-chain IVs and return a clean surface dataframe."""
    ticker = ticker.upper().strip()
    if not ticker:
        raise ValueError("Ticker is empty.")

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist is None or hist.empty:
        raise ValueError(f"No recent price found for {ticker}.")

    spot = float(hist["Close"].dropna().iloc[-1])
    expiries = list(tk.options or [])[:max_expiries]
    if not expiries:
        raise ValueError(f"No listed option expiries found for {ticker}.")

    rows = []
    today = pd.Timestamp.today().normalize()
    for expiry in expiries:
        try:
            chain = tk.option_chain(expiry)
        except Exception:
            continue

        expiry_ts = pd.Timestamp(expiry)
        dte = max((expiry_ts - today).days, 1)
        for opt_type, df in (("Call", chain.calls), ("Put", chain.puts)):
            if df is None or df.empty:
                continue
            cols = ["strike", "impliedVolatility", "volume", "openInterest"]
            tmp = df[[c for c in cols if c in df.columns]].copy()
            tmp = tmp.rename(columns={"impliedVolatility": "iv"})
            tmp["type"] = opt_type
            tmp["expiry"] = expiry
            tmp["dte"] = dte
            tmp["maturity"] = dte / 365.0
            rows.append(tmp)

    if not rows:
        raise ValueError(f"Yahoo returned no usable option chain for {ticker}.")

    surface = pd.concat(rows, ignore_index=True)
    surface = surface.dropna(subset=["strike", "iv"])
    surface = surface[(surface["iv"] > 0.01) & (surface["iv"] < 5.0)]
    surface = surface[(surface["strike"] > spot * 0.45) & (surface["strike"] < spot * 1.75)]
    if surface.empty:
        raise ValueError(f"Option chain for {ticker} is empty after cleaning.")

    surface["moneyness"] = surface["strike"] / spot
    surface["iv_pct"] = surface["iv"] * 100
    return surface, spot, expiries

@st.cache_data(ttl=900, show_spinner=False)
def fetch_estimated_surface(ticker: str, max_expiries: int = 8):
    """Fallback surface from historical vol when listed option IV is unavailable."""
    hist, minfo = fetch_market_data(ticker)
    log_ret, _, hvols = calc_hist_vols(hist)

    spot = float(minfo["price"])
    base_vol = hvols.get("30d") or hvols.get("60d") or hvols.get("90d") or hvols.get("252d")
    if not base_vol:
        base_vol = float(log_ret.std() * np.sqrt(252) * 100)
    base_vol = max(base_vol / 100, 0.05)

    dtes = np.array([30, 60, 90, 180, 270, 365, 540, 730])[:max_expiries]
    moneyness_grid = np.linspace(0.7, 1.3, 25)
    rows = []
    today = pd.Timestamp.today().normalize()
    for dte in dtes:
        term_adj = 0.015 * np.exp(-dte / 365)
        for m in moneyness_grid:
            skew = -0.10 * np.log(m)
            convexity = 0.18 * (np.log(m) ** 2)
            iv = max(base_vol + term_adj + skew + convexity, 0.01)
            expiry = (today + pd.Timedelta(days=int(dte))).date().isoformat()
            for opt_type in ("Call", "Put"):
                rows.append({
                    "strike": spot * m,
                    "iv": iv,
                    "volume": np.nan,
                    "openInterest": np.nan,
                    "type": opt_type,
                    "expiry": expiry,
                    "dte": int(dte),
                    "maturity": dte / 365,
                    "moneyness": m,
                    "iv_pct": iv * 100,
                })

    return pd.DataFrame(rows), spot, [r["expiry"] for r in rows[::len(moneyness_grid) * 2]]

# ── PLOTLY THEME ──────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Arial, sans-serif", size=12, color="#374151"),
    margin=dict(l=50, r=30, t=48, b=44),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
)

def fig_style(fig, title=""):
    fig.update_layout(**BASE_LAYOUT,
        title=dict(text=title, font=dict(size=14, color="#0f172a")),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
    )
    return fig

# ── UI HELPERS ────────────────────────────────────────────────────────────────

def metric_card(label, value, color="#0f172a"):
    st.markdown(f"""
    <div style="background:white;border-radius:10px;padding:18px 16px;
    box-shadow:0 1px 4px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04);
    text-align:center;margin-bottom:8px;">
      <div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
      color:#94a3b8;margin-bottom:6px;">{label}</div>
      <div style="font-size:22px;font-weight:800;color:{color};">{value}</div>
    </div>""", unsafe_allow_html=True)

def info_box(text):
    st.markdown(f"""
    <div style="background:#eff6ff;border-left:4px solid #3b82f6;border-radius:6px;
    padding:14px 18px;font-size:13px;color:#1e3a5f;margin:14px 0;line-height:1.6;">
    {text}</div>""", unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#0b1b2a;padding:18px 40px 0 40px;">
  <div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:16px;gap:20px;flex-wrap:wrap;">
    <div>
      <div style="font-size:10px;color:rgba(255,255,255,0.42);letter-spacing:2.6px;font-weight:800;text-transform:uppercase;margin-bottom:5px;">QUANTITATIVE FINANCE - STUDENT PROJECT</div>
      <div style="font-size:18px;font-weight:800;color:white;">Structured Products &amp; Options Dashboard</div>
    </div>
    <div style="display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end;">
      <span style="background:rgba(255,255,255,0.11);color:rgba(255,255,255,0.88);border-radius:20px;padding:5px 12px;font-size:11px;font-weight:800;border:1px solid rgba(255,255,255,0.15);">Black-Scholes</span>
      <span style="background:rgba(255,255,255,0.11);color:rgba(255,255,255,0.88);border-radius:20px;padding:5px 12px;font-size:11px;font-weight:800;border:1px solid rgba(255,255,255,0.15);">13 Strategies</span>
      <span style="background:rgba(255,255,255,0.11);color:rgba(255,255,255,0.88);border-radius:20px;padding:5px 12px;font-size:11px;font-weight:800;border:1px solid rgba(255,255,255,0.15);">Autocall</span>
      <span style="background:rgba(255,255,255,0.11);color:rgba(255,255,255,0.88);border-radius:20px;padding:5px 12px;font-size:11px;font-weight:800;border:1px solid rgba(255,255,255,0.15);">BRC</span>
      <span style="background:rgba(59,130,246,0.26);color:#bfdbfe;border-radius:20px;padding:5px 12px;font-size:11px;font-weight:800;border:1px solid rgba(59,130,246,0.45);">Live Market</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
# TABS ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "OVERVIEW", "LIVE DATA", "BLACK-SCHOLES", "STRATEGIES",
    "AUTOCALL", "BRC", "MC PRICER", "SENSITIVITIES", "LEARNINGS"
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="background:linear-gradient(135deg,#10233f 0%,#173d70 54%,#0f5774 100%);
    border-radius:16px;padding:56px 48px;margin-bottom:28px;color:white;box-shadow:0 18px 45px rgba(16,35,63,0.18);">
      <div style="font-size:10px;color:rgba(255,255,255,0.48);letter-spacing:2.6px;font-weight:800;text-transform:uppercase;margin-bottom:18px;">
        QUANTITATIVE FINANCE - INTERACTIVE PROJECT
      </div>
      <h1 style="font-size:42px;font-weight:800;color:white;margin:0 0 20px 0;line-height:1.12;">
        Structured Products<br>&amp; Options Dashboard
      </h1>
      <p style="font-size:15px;color:rgba(255,255,255,0.72);max-width:680px;line-height:1.75;margin-bottom:30px;">
        Full pricing and risk framework: Black-Scholes replication, live Greeks, option strategies,
        Autocall and BRC mechanics, Monte Carlo pricing and real-time market data.
      </p>
      <div style="display:flex;gap:8px;flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.92);border-radius:20px;padding:6px 14px;font-size:12px;font-weight:800;">Black-Scholes</span>
        <span style="background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.92);border-radius:20px;padding:6px 14px;font-size:12px;font-weight:800;">Greeks</span>
        <span style="background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.92);border-radius:20px;padding:6px 14px;font-size:12px;font-weight:800;">Strategies</span>
        <span style="background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.92);border-radius:20px;padding:6px 14px;font-size:12px;font-weight:800;">Autocall</span>
        <span style="background:rgba(255,255,255,0.15);color:rgba(255,255,255,0.92);border-radius:20px;padding:6px 14px;font-size:12px;font-weight:800;">BRC</span>
        <span style="background:rgba(59,130,246,0.32);color:#bfdbfe;border-radius:20px;padding:6px 14px;font-size:12px;font-weight:800;">Live Market Data</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (c1, "5",  "#3b82f6", "Greeks Live",       "Delta, Gamma, Vega, Theta, Rho"),
        (c2, "13", "#22c55e", "Option Strategies",  "Straddle, Condor, Butterfly and more"),
        (c3, "MC", "#f97316", "Structured Pricer",  "GBM simulation for Autocall and BRC"),
        (c4, "LIVE", "#3b82f6", "Live Market Data", "Real prices and historical vol via yfinance"),
    ]
    for col, num, color, label, desc in stats:
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:28px 24px;min-height:154px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06),0 8px 22px rgba(15,35,63,0.06);
            border-left:4px solid {color};">
              <div style="font-size:42px;font-weight:800;color:{color};line-height:1;margin-bottom:14px;">{num}</div>
              <div style="font-size:15px;font-weight:800;color:#0f172a;margin-bottom:7px;">{label}</div>
              <div style="font-size:13px;color:#64748b;line-height:1.45;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    links = [
        (c1, "Live Data",      "Real stock prices, historical vol, returns distribution", "#3b82f6"),
        (c2, "Black-Scholes",  "European pricing with dividends and full Greeks",          "#8b5cf6"),
        (c3, "Strategies",     "Classic strategies with live payoff and P&L",             "#22c55e"),
        (c4, "Autocall / BRC", "Monte Carlo simulation of structured products",           "#f97316"),
    ]
    for col, title, desc, color in links:
        with col:
            st.markdown(f"""
            <div style="background:white;border-radius:10px;padding:22px 20px;min-height:96px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);border-top:3px solid {color};">
              <div style="font-size:14px;font-weight:800;color:{color};margin-bottom:8px;">{title}</div>
              <div style="font-size:13px;color:#64748b;line-height:1.5;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:24px;font-weight:800;color:#0f172a;">Live Market Data</div>
    <div style="font-size:14px;color:#64748b;margin-bottom:24px;">
        Real-time prices and historical volatility via yfinance — use any ticker to pre-fill the BS Pricer</div>
    """, unsafe_allow_html=True)

    # ── Upgrade hint ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#fefce8;border-left:4px solid #eab308;border-radius:6px;
    padding:10px 16px;font-size:13px;color:#713f12;margin-bottom:16px;">
    <strong>Si les données ne chargent pas</strong> — dans ton terminal (Ctrl+C pour arrêter d'abord) :
    <br><code style="background:#fef9c3;padding:2px 6px;border-radius:4px;">pip install --upgrade yfinance curl_cffi</code>
    puis <code style="background:#fef9c3;padding:2px 6px;border-radius:4px;">streamlit run app.py</code>
    </div>""", unsafe_allow_html=True)

    # ── Ticker input ──────────────────────────────────────────────────────
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        ticker_input = st.text_input(
            "Ticker symbol",
            placeholder="AAPL, MC.PA, ^FCHI, BNP.PA, NVDA…",
            label_visibility="collapsed",
        )
    with col_btn:
        load_btn = st.button("Load Data", type="primary", use_container_width=True)

    # Quick picks
    st.markdown("""
    <div style="margin:-4px 0 20px 0;font-size:12px;color:#94a3b8;font-weight:600;
    letter-spacing:0.5px;text-transform:uppercase;">Quick picks</div>""",
    unsafe_allow_html=True)
    qcols = st.columns(9)
    quick = ["AAPL","MSFT","NVDA","TSLA","MC.PA","BNP.PA","TTE.PA","^FCHI","^GSPC"]
    for qc, tk_q in zip(qcols, quick):
        with qc:
            if st.button(tk_q, key=f"qp_{tk_q}", use_container_width=True):
                ticker_input = tk_q
                load_btn = True

    # ── Fetch & display ───────────────────────────────────────────────────
    ticker_to_use = ticker_input.strip() if ticker_input else ""

    if load_btn and ticker_to_use:
        with st.spinner(f"Fetching data for **{ticker_to_use.upper()}**…"):
            try:
                hist, minfo = fetch_market_data(ticker_to_use)
            except Exception as _err:
                hist, minfo = None, None
                st.error(f"**{ticker_to_use.upper()}** — {_err}")

        if hist is not None and minfo is not None:
            log_ret, rolling_vol, hvols = calc_hist_vols(hist)
            price     = minfo["price"]
            pct_chg   = (price / minfo["prev_close"] - 1) * 100 if minfo["prev_close"] else 0
            chg_color = "#22c55e" if pct_chg >= 0 else "#ef4444"
            chg_sign  = "+" if pct_chg >= 0 else ""
            vol_30    = hvols["30d"] or 20.0
            ccy       = minfo["currency"]

            # ── Store in session state for BS Pricer ──────────────────────
            st.session_state["bs_spot"]   = round(price, 4)
            st.session_state["bs_sigma"]  = round(vol_30, 2)
            st.session_state["bs_ticker"] = minfo["name"]

            # ── Top metric cards ──────────────────────────────────────────
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:20px 28px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-bottom:20px;
            display:flex;align-items:center;gap:32px;">
              <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
                color:#94a3b8;margin-bottom:4px;">TICKER</div>
                <div style="font-size:28px;font-weight:800;color:#0f172a;">{minfo['name']}</div>
              </div>
              <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
                color:#94a3b8;margin-bottom:4px;">LAST PRICE</div>
                <div style="font-size:28px;font-weight:800;color:#0f172a;">{price:,.2f} {ccy}</div>
              </div>
              <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
                color:#94a3b8;margin-bottom:4px;">1D CHANGE</div>
                <div style="font-size:28px;font-weight:800;color:{chg_color};">{chg_sign}{pct_chg:.2f}%</div>
              </div>
              <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
                color:#94a3b8;margin-bottom:4px;">52W HIGH</div>
                <div style="font-size:28px;font-weight:800;color:#0f172a;">{minfo['high_52w']:,.2f}</div>
              </div>
              <div>
                <div style="font-size:11px;font-weight:700;letter-spacing:1px;text-transform:uppercase;
                color:#94a3b8;margin-bottom:4px;">52W LOW</div>
                <div style="font-size:28px;font-weight:800;color:#0f172a;">{minfo['low_52w']:,.2f}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Historical vol cards ──────────────────────────────────────
            vc1, vc2, vc3, vc4 = st.columns(4)
            vol_labels = [("30d","30-Day HV","#3b82f6"),("60d","60-Day HV","#8b5cf6"),
                          ("90d","90-Day HV","#f97316"),("252d","1-Year HV","#22c55e")]
            for col, (key, lbl, clr) in zip([vc1,vc2,vc3,vc4], vol_labels):
                with col:
                    v = hvols[key]
                    metric_card(lbl, f"{v:.1f}%" if v else "N/A", clr)

            # ── CTA: send to BS Pricer ────────────────────────────────────
            st.markdown(f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
            padding:14px 20px;margin:16px 0;display:flex;align-items:center;
            justify-content:space-between;">
              <span style="font-size:14px;color:#1e40af;">
                <strong>✓ Parameters loaded.</strong>
                Spot = <strong>{price:,.2f}</strong> and
                σ (30d) = <strong>{vol_30:.1f}%</strong>
                are now pre-filled in the <strong>Black-Scholes</strong> tab.
              </span>
            </div>""", unsafe_allow_html=True)

            # ── Price chart ───────────────────────────────────────────────
            fig_price = go.Figure()
            fig_price.add_trace(go.Candlestick(
                x=hist.index, open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"],
                increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                name="Price",
            ))
            fig_style(fig_price, f"{minfo['name']} — 1-Year Price History")
            fig_price.update_xaxes(title="Date", rangeslider_visible=False)
            fig_price.update_yaxes(title=f"Price ({ccy})")
            st.plotly_chart(fig_price, use_container_width=True)

            # ── Returns & Rolling vol ────────────────────────────────────
            ch_l, ch_r = st.columns(2)

            with ch_l:
                fig_ret = go.Figure()
                fig_ret.add_trace(go.Histogram(
                    x=log_ret * 100, nbinsx=60,
                    marker_color="#3b82f6", opacity=0.8, name="Daily returns"
                ))
                mu = float(log_ret.mean() * 100)
                fig_ret.add_vline(x=mu, line_dash="dash", line_color="#0f172a",
                                  annotation_text=f"Mean {mu:.2f}%")
                fig_style(fig_ret, "Daily Log-Returns Distribution (%)")
                fig_ret.update_xaxes(title="Daily Return (%)")
                fig_ret.update_yaxes(title="Count")
                st.plotly_chart(fig_ret, use_container_width=True)

            with ch_r:
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(
                    x=rolling_vol.index, y=rolling_vol.values,
                    line=dict(color="#f97316", width=2), name="Rolling 30d Vol",
                    fill="tozeroy", fillcolor="rgba(249,115,22,0.08)"
                ))
                fig_vol.add_hline(y=vol_30, line_dash="dash", line_color="#3b82f6",
                                  annotation_text=f"Current {vol_30:.1f}%")
                fig_style(fig_vol, "Rolling 30-Day Historical Volatility (%)")
                fig_vol.update_xaxes(title="Date")
                fig_vol.update_yaxes(title="Annualised Vol (%)")
                st.plotly_chart(fig_vol, use_container_width=True)

            # ── Summary stats table ───────────────────────────────────────
            daily_ret = log_ret * 100
            ann_ret   = float(log_ret.mean() * 252 * 100)
            sharpe    = ann_ret / vol_30 if vol_30 > 0 else 0
            max_dd    = float(((hist["Close"] / hist["Close"].cummax()) - 1).min() * 100)
            skew      = float(daily_ret.skew())
            kurt      = float(daily_ret.kurt())
            var_95    = float(np.percentile(daily_ret, 5))

            stats_df = pd.DataFrame({
                "Metric": ["Annualised Return","Hist. Vol (30d)","Hist. Vol (252d)",
                           "Sharpe (annualised)","Max Drawdown (1Y)","Skewness","Kurtosis","VaR 95% (1d)"],
                "Value":  [f"{ann_ret:.2f}%", f"{vol_30:.2f}%", f"{hvols['252d']:.2f}%" if hvols['252d'] else "N/A",
                           f"{sharpe:.2f}", f"{max_dd:.2f}%", f"{skew:.3f}", f"{kurt:.3f}", f"{var_95:.2f}%"],
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

    elif not ticker_to_use:
        st.markdown("""
        <div style="background:white;border-radius:12px;padding:48px;text-align:center;
        box-shadow:0 1px 4px rgba(0,0,0,0.06);">
          <div style="font-size:40px;margin-bottom:16px;">📡</div>
          <div style="font-size:17px;font-weight:700;color:#0f172a;margin-bottom:8px;">
            Enter a ticker to load real market data</div>
          <div style="font-size:14px;color:#64748b;">
            Examples: <strong>AAPL</strong> · <strong>MC.PA</strong> · <strong>BNP.PA</strong> ·
            <strong>^FCHI</strong> · <strong>NVDA</strong></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — BLACK-SCHOLES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:24px;font-weight:800;color:#0f172a;">Black-Scholes Option Pricer</div>
    <div style="font-size:14px;color:#64748b;margin-bottom:24px;">European options with continuous dividends — live Greeks</div>
    """, unsafe_allow_html=True)

    # ── Market loader banner (if data was loaded from Live Data tab) ──────
    if st.session_state.get("bs_ticker"):
        ticker_lbl = st.session_state["bs_ticker"]
        spot_lbl   = st.session_state["bs_spot"]
        sigma_lbl  = st.session_state["bs_sigma"]
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
        padding:12px 20px;margin-bottom:20px;display:flex;align-items:center;gap:12px;">
          <span style="font-size:18px;">📡</span>
          <span style="font-size:13px;color:#166534;">
            Market data loaded from <strong>{ticker_lbl}</strong> —
            Spot = <strong>{spot_lbl:,.2f}</strong>,
            σ (30d HV) = <strong>{sigma_lbl:.1f}%</strong>.
            Parameters pre-filled below.
          </span>
        </div>""", unsafe_allow_html=True)

    col_param, col_res = st.columns([1, 2], gap="large")

    with col_param:
        S     = st.number_input("Spot Price (S)",        value=float(st.session_state["bs_spot"]),  min_value=0.01, step=1.0,   key="bs_S")
        K     = st.number_input("Strike Price (K)",      value=float(st.session_state["bs_spot"]),  min_value=0.01, step=1.0,   key="bs_K")
        T     = st.number_input("Time to Maturity (yr)", value=1.0,  min_value=0.01, max_value=10.0, step=0.05,     key="bs_T")
        sigma = st.number_input("Volatility σ (%)",      value=float(st.session_state["bs_sigma"]), min_value=0.1,  max_value=200.0, step=0.5, key="bs_sigma_inp") / 100
        r     = st.number_input("Risk-free Rate (%)",    value=3.0,  min_value=-5.0, max_value=20.0, step=0.1,     key="bs_r") / 100
        q     = st.number_input("Dividend Yield (%)",    value=0.0,  min_value=0.0,  max_value=20.0, step=0.1,     key="bs_q") / 100
        otype = st.selectbox("Option Type", ["Call","Put"], key="bs_otype")

    call_p = bs_price(S, K, T, sigma, r, q, "Call")
    put_p  = bs_price(S, K, T, sigma, r, q, "Put")
    gk     = bs_greeks(S, K, T, sigma, r, q, otype)
    d1, d2 = bs_d1_d2(S, K, T, sigma, r, q)

    with col_res:
        pc, pp = st.columns(2)
        with pc: metric_card("Call Price", f"{call_p:.4f}", "#22c55e")
        with pp: metric_card("Put Price",  f"{put_p:.4f}", "#ef4444")

        g_cols = st.columns(5)
        gcolors = {"Delta":"#3b82f6","Gamma":"#8b5cf6","Vega":"#f97316","Theta":"#ef4444","Rho":"#22c55e"}
        for col, name in zip(g_cols, ["Delta","Gamma","Vega","Theta","Rho"]):
            with col: metric_card(name, f"{gk[name]:.4f}", gcolors[name])

        if d1 is not None:
            st.markdown(f"""<div style="background:white;border-radius:10px;padding:14px 20px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);font-size:13px;color:#374151;margin-bottom:16px;
            display:flex;gap:32px;">
            <span><strong>d₁</strong> = {d1:.4f}</span>
            <span><strong>d₂</strong> = {d2:.4f}</span>
            <span><strong>N(d₁)</strong> = {norm.cdf(d1 if otype=='Call' else -d1):.4f}</span>
            </div>""", unsafe_allow_html=True)

    spot_range = np.linspace(max(0.1, S*0.4), S*1.8, 400)
    call_pay   = np.maximum(spot_range - K, 0)
    put_pay    = np.maximum(K - spot_range, 0)

    t1, t2, t3, t4 = st.tabs(["Payoff","P&L","Price vs Vol","Price vs Spot"])
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=call_pay, name="Call", line=dict(color="#22c55e",width=2.5)))
        fig.add_trace(go.Scatter(x=spot_range, y=put_pay,  name="Put",  line=dict(color="#ef4444",width=2.5)))
        fig.add_vline(x=S, line_dash="dash", line_color="#94a3b8", annotation_text="Spot")
        fig.add_vline(x=K, line_dash="dot",  line_color="#3b82f6", annotation_text="Strike")
        fig_style(fig,"Payoff at Maturity"); fig.update_xaxes(title="Spot at Maturity"); fig.update_yaxes(title="Payoff")
        st.plotly_chart(fig, use_container_width=True)
    with t2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=call_pay-call_p, name="Call P&L", line=dict(color="#22c55e",width=2.5)))
        fig.add_trace(go.Scatter(x=spot_range, y=put_pay -put_p,  name="Put P&L",  line=dict(color="#ef4444",width=2.5)))
        fig.add_hline(y=0, line_dash="dash", line_color="#0f172a", line_width=1)
        fig.add_vline(x=S, line_dash="dash", line_color="#94a3b8", annotation_text="Spot")
        fig_style(fig,"P&L at Maturity"); fig.update_xaxes(title="Spot at Maturity"); fig.update_yaxes(title="P&L")
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        vols = np.linspace(0.01,1.0,250)
        fig  = go.Figure()
        fig.add_trace(go.Scatter(x=vols*100, y=[bs_price(S,K,T,v,r,q,"Call") for v in vols], name="Call", line=dict(color="#22c55e",width=2.5)))
        fig.add_trace(go.Scatter(x=vols*100, y=[bs_price(S,K,T,v,r,q,"Put")  for v in vols], name="Put",  line=dict(color="#ef4444",width=2.5)))
        fig.add_vline(x=sigma*100, line_dash="dash", line_color="#f97316", annotation_text="σ")
        fig_style(fig,"Option Price vs Volatility"); fig.update_xaxes(title="Volatility (%)"); fig.update_yaxes(title="Price")
        st.plotly_chart(fig, use_container_width=True)
    with t4:
        sps = np.linspace(max(0.1,S*0.4),S*1.8,300)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sps, y=[bs_price(s,K,T,sigma,r,q,"Call") for s in sps], name="Call", line=dict(color="#22c55e",width=2.5)))
        fig.add_trace(go.Scatter(x=sps, y=[bs_price(s,K,T,sigma,r,q,"Put")  for s in sps], name="Put",  line=dict(color="#ef4444",width=2.5)))
        fig.add_vline(x=S, line_dash="dash", line_color="#94a3b8", annotation_text="Spot")
        fig.add_vline(x=K, line_dash="dot",  line_color="#3b82f6", annotation_text="Strike")
        fig_style(fig,"Option Price vs Spot"); fig.update_xaxes(title="Spot"); fig.update_yaxes(title="Price")
        st.plotly_chart(fig, use_container_width=True)

    info_box("Volatility always increases option value (Vega > 0). Theta erodes time value daily. "
             "Delta is the hedge ratio; Gamma the cost of rehedging. "
             "Higher rates increase calls and decrease puts.")
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:24px;font-weight:800;color:#0f172a;">Option Strategies Simulator</div>
    <div style="font-size:14px;color:#64748b;margin-bottom:24px;">Payoff & P&L profiles for 13 classic strategies</div>
    """, unsafe_allow_html=True)

    STRATS = {
        "Long Call":        {"desc":"Buy a call. Max loss = premium. Unlimited upside above K + premium.",   "market":"Bullish",            "strikes":["K1"]},
        "Long Put":         {"desc":"Buy a put. Protection or bearish bet. Max loss = premium.",             "market":"Bearish",            "strikes":["K1"]},
        "Short Call":       {"desc":"Sell a call. Collect premium, unlimited risk above K + premium.",       "market":"Bearish / Income",   "strikes":["K1"]},
        "Short Put":        {"desc":"Sell a put. Collect premium, risk below K − premium.",                  "market":"Bullish / Income",   "strikes":["K1"]},
        "Covered Call":     {"desc":"Hold underlying + sell call. Enhanced yield, capped upside.",           "market":"Mildly Bullish",     "strikes":["K1"]},
        "Protective Put":   {"desc":"Hold underlying + buy put. Downside protection.",                       "market":"Bullish with Hedge", "strikes":["K1"]},
        "Bull Call Spread": {"desc":"Buy call K1, sell call K2. Limited profit, limited cost.",              "market":"Moderately Bullish", "strikes":["K1","K2"]},
        "Bear Put Spread":  {"desc":"Buy put K2, sell put K1. Limited profit, limited cost.",                "market":"Moderately Bearish", "strikes":["K1","K2"]},
        "Straddle":         {"desc":"Buy call + put at same strike. Profits from large moves either way.",   "market":"High Volatility",    "strikes":["K1"]},
        "Strangle":         {"desc":"Buy OTM call K2 + OTM put K1. Cheaper than straddle.",                 "market":"High Volatility",    "strikes":["K1","K2"]},
        "Butterfly Spread": {"desc":"Buy K1, sell 2×K2, buy K3. Profits from low volatility near K2.",      "market":"Low Volatility",     "strikes":["K1","K2","K3"]},
        "Iron Condor":      {"desc":"Sell put K2, buy put K1, sell call K3, buy call K4. Net credit range.","market":"Range-bound",        "strikes":["K1","K2","K3","K4"]},
        "Calendar Spread":  {"desc":"Buy long-dated call, sell short-dated call same strike. Vega play.",   "market":"Neutral / Vol Play", "strikes":["K1"]},
    }
    mkt_color = {
        "Bullish":"#22c55e","Bearish":"#ef4444","Bearish / Income":"#f87171",
        "Bullish / Income":"#86efac","Mildly Bullish":"#6ee7b7","Bullish with Hedge":"#34d399",
        "Moderately Bullish":"#4ade80","Moderately Bearish":"#f97316","High Volatility":"#8b5cf6",
        "Low Volatility":"#3b82f6","Range-bound":"#0ea5e9","Neutral / Vol Play":"#a78bfa",
    }

    col_l, col_r = st.columns([1, 2], gap="large")
    with col_l:
        strategy = st.selectbox("Strategy", list(STRATS.keys()))
        info_s   = STRATS[strategy]
        mc       = mkt_color.get(info_s["market"], "#64748b")
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:18px;
        box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-bottom:16px;">
          <div style="font-size:13px;color:#374151;margin-bottom:12px;line-height:1.5;">{info_s['desc']}</div>
          <span style="background:{mc}22;color:{mc};border-radius:20px;
          padding:3px 10px;font-size:12px;font-weight:700;">{info_s['market']}</span>
        </div>""", unsafe_allow_html=True)

        S0_s  = st.number_input("Spot", value=float(st.session_state["bs_spot"]), min_value=0.01, step=1.0, key="st_S0")
        T_s   = st.number_input("Maturity (yr)", value=0.5, min_value=0.01, max_value=5.0, step=0.05, key="st_T")
        sig_s = st.number_input("Volatility (%)", value=float(st.session_state["bs_sigma"]), min_value=0.1, max_value=150.0, step=0.5, key="st_sig") / 100
        r_s   = st.number_input("Risk-free Rate (%)", value=3.0, step=0.1, key="str_r") / 100
        q_s   = st.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, step=0.1, key="str_q") / 100
        notional = st.number_input("Notional", value=1.0, min_value=0.01, step=1.0, key="st_notional")

        st.markdown("**Strike(s)**")
        defs_s = {"K1":S0_s*0.90,"K2":S0_s,"K3":S0_s*1.05,"K4":S0_s*1.10}
        sv = {k: st.number_input(f"Strike {k}", value=round(defs_s[k],2), min_value=0.01, step=1.0, key=f"st_strike_{k}")
              for k in info_s["strikes"]}
        K1s=sv.get("K1",S0_s*0.9); K2s=sv.get("K2",S0_s); K3s=sv.get("K3",S0_s*1.05); K4s=sv.get("K4",S0_s*1.10)
        manual_s = st.toggle("Override premiums manually", key="st_manual")

    def pr(K_,ot): return bs_price(S0_s,K_,T_s,sig_s,r_s,q_s,ot)
    ap = {"cK1":pr(K1s,"Call"),"pK1":pr(K1s,"Put"),"cK2":pr(K2s,"Call"),"pK2":pr(K2s,"Put"),
          "cK3":pr(K3s,"Call"),"pK3":pr(K3s,"Put"),"cK4":pr(K4s,"Call"),"pK4":pr(K4s,"Put")}
    if manual_s:
        with col_l:
            for k in ap: ap[k]=st.number_input(k, value=round(ap[k],4), min_value=0.0, step=0.01, format="%.4f")

    spts = np.linspace(max(0.1,S0_s*0.4), S0_s*1.8, 500)
    def strat_pnl(name, sp):
        c1=np.maximum(sp-K1s,0); c2=np.maximum(sp-K2s,0); c3=np.maximum(sp-K3s,0); c4=np.maximum(sp-K4s,0)
        p1=np.maximum(K1s-sp,0); p2=np.maximum(K2s-sp,0)
        if name=="Long Call":        return c1, c1-ap["cK1"]
        if name=="Long Put":         return p1, p1-ap["pK1"]
        if name=="Short Call":       return -c1, ap["cK1"]-c1
        if name=="Short Put":        return -p1, ap["pK1"]-p1
        if name=="Covered Call":     return sp-S0_s-c2+ap["cK2"], sp-S0_s-c2+ap["cK2"]
        if name=="Protective Put":   return sp-S0_s+p1-ap["pK1"], sp-S0_s+p1-ap["pK1"]
        if name=="Bull Call Spread": return c1-c2, c1-c2-(ap["cK1"]-ap["cK2"])
        if name=="Bear Put Spread":  return p2-p1, p2-p1-(ap["pK2"]-ap["pK1"])
        if name=="Straddle":         return c1+p1, c1+p1-ap["cK1"]-ap["pK1"]
        if name=="Strangle":         return c2+p1, c2+p1-ap["cK2"]-ap["pK1"]
        if name=="Butterfly Spread": net=ap["cK1"]-2*ap["cK2"]+ap["cK3"]; return c1-2*c2+c3,(c1-2*c2+c3)-net
        if name=="Calendar Spread":  return np.zeros_like(sp),(ap["cK1"]*0.4)-c1+c1*0.4
        if name=="Iron Condor":
            net_cr=ap["cK3"]-ap["cK4"]+ap["pK2"]-ap["pK1"]
            pay=-c3+c4+p2-p1; return pay, pay+net_cr
        return np.zeros_like(sp), np.zeros_like(sp)

    pay_s, pnl_s = strat_pnl(strategy, spts)
    pay_s *= notional; pnl_s *= notional
    maxp = float(np.max(pnl_s)); maxl = float(np.min(pnl_s))
    be   = [float(spts[i]) for i in np.where(np.diff(np.sign(pnl_s)))[0]]

    with col_r:
        m1,m2,m3=st.columns(3)
        with m1: metric_card("Max Profit", f"{maxp:.2f}" if maxp<9e5 else "Unlimited","#22c55e")
        with m2: metric_card("Max Loss",   f"{maxl:.2f}","#ef4444")
        with m3: metric_card("Break-Even(s)", " / ".join([f"{b:.1f}" for b in be]) if be else "N/A","#3b82f6")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spts, y=pay_s, name="Payoff", line=dict(color="#3b82f6",width=2.5)))
        fig.add_trace(go.Scatter(x=spts, y=pnl_s, name="P&L",    line=dict(color="#22c55e",width=2,dash="dash")))
        fig.add_hline(y=0, line_color="#0f172a", line_dash="dash", line_width=1)
        fig.add_vline(x=S0_s, line_dash="dot", line_color="#94a3b8", annotation_text="Spot")
        for b in be:
            fig.add_vline(x=b, line_dash="dot", line_color="#f97316",
                          annotation_text=f"BE {b:.1f}", annotation_position="top right")
        fig_style(fig, f"{strategy} — Payoff & P&L at Maturity")
        fig.update_xaxes(title="Spot at Maturity"); fig.update_yaxes(title="P&L / Payoff")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(pd.DataFrame({
            "Strategy":[strategy],"Outlook":[info_s["market"]],
            "Max Profit":[f"{maxp:.2f}" if maxp<9e5 else "Unlimited"],"Max Loss":[f"{maxl:.2f}"],
            "Break-Even(s)":[" / ".join([f"{b:.1f}" for b in be]) if be else "N/A"],
        }), use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — AUTOCALL
# ═════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:24px;font-weight:800;color:#0f172a;">Autocall Simulator</div>
    <div style="font-size:14px;color:#64748b;margin-bottom:16px;">Monte Carlo simulation — GBM lognormal paths</div>
    """, unsafe_allow_html=True)
    info_box("An <strong>autocall</strong> is automatically redeemed if the underlying exceeds the recall barrier "
             "at an observation date. The coupon remunerates for the downside risk (embedded put), the volatility "
             "sold implicitly, and the illiquidity.")

    col_p, col_r = st.columns([1,2], gap="large")
    with col_p:
        S0_ac  = st.number_input("Initial Spot S₀", value=float(st.session_state["bs_spot"]), min_value=0.01, step=1.0, key="ac_s0")
        S_ac   = st.number_input("Current Spot",    value=float(st.session_state["bs_spot"]), min_value=0.01, step=1.0, key="ac_sc")
        sig_ac = st.number_input("Volatility (%)",  value=float(st.session_state["bs_sigma"]), min_value=0.1, max_value=150.0, step=0.5, key="ac_sig") / 100
        r_ac   = st.number_input("Risk-free Rate (%)", value=3.0, step=0.1, key="ac_r") / 100
        q_ac   = st.number_input("Dividend Yield (%)",value=2.0, min_value=0.0, step=0.1, key="ac_q") / 100
        mat_ac = st.number_input("Maturity (years)",  value=3.0, min_value=0.25, max_value=10.0, step=0.25, key="ac_mat")
        freq   = st.selectbox("Observation Frequency",["Quarterly","Semi-annual","Annual"], key="ac_freq")
        rb_ac  = st.number_input("Recall Barrier (% of S₀)",      value=100.0, min_value=50.0, max_value=150.0, step=1.0, key="ac_rb") / 100
        cpn_ac = st.number_input("Coupon per Period (%)",          value=5.0,   min_value=0.0,  max_value=50.0,  step=0.5, key="ac_cpn") / 100
        pb_ac  = st.number_input("Protection Barrier (% of S₀)",  value=60.0,  min_value=10.0, max_value=100.0, step=1.0, key="ac_pb") / 100
        n_ac   = st.number_input("Simulations",                    value=5000,  min_value=500,  max_value=50000, step=500,  key="ac_n")
        sd_ac  = st.number_input("Seed (0=random)",                value=42,    min_value=0,    step=1,          key="ac_seed")
        run_ac = st.button("Run Simulation", type="primary", key="ac_run")

    with col_r:
        if run_ac:
            ppy    = {"Quarterly":4,"Semi-annual":2,"Annual":1}[freq]
            tot_p  = int(mat_ac * ppy)
            obs_t  = [i/ppy for i in range(1, tot_p+1)]
            rng    = np.random.default_rng(sd_ac if sd_ac>0 else None)
            ns     = max(int(mat_ac*252),50)
            dt     = mat_ac / ns
            Z      = rng.standard_normal((int(n_ac), ns))
            paths  = S_ac * np.exp(np.cumsum((r_ac-q_ac-0.5*sig_ac**2)*dt + sig_ac*np.sqrt(dt)*Z, axis=1))
            obs_i  = [max(0,min(int(round(t*ns/mat_ac))-1,ns-1)) for t in obs_t]
            payoffs= np.zeros(int(n_ac)); rec_p = np.full(int(n_ac),-1)
            for i in range(int(n_ac)):
                rcld=False
                for pi,(idx,t) in enumerate(zip(obs_i,obs_t)):
                    if paths[i,idx] >= rb_ac*S0_ac:
                        payoffs[i]=1.0+cpn_ac*(pi+1); rec_p[i]=pi+1; rcld=True; break
                if not rcld:
                    sf=paths[i,-1]
                    payoffs[i]=(1.0+cpn_ac*tot_p) if sf>=pb_ac*S0_ac else (sf/S0_ac+cpn_ac*tot_p)

            pr_rc=np.mean(rec_p>0); pr_lc=np.mean(payoffs<1.0)
            avg_t=np.mean(rec_p[rec_p>0]/ppy) if (rec_p>0).any() else None

            m1,m2,m3,m4=st.columns(4)
            with m1: metric_card("Mean Payoff",       f"{np.mean(payoffs):.4f}","#3b82f6")
            with m2: metric_card("Prob. Recall",      f"{pr_rc*100:.1f}%","#22c55e")
            with m3: metric_card("Prob. Capital Loss",f"{pr_lc*100:.1f}%","#ef4444")
            with m4: metric_card("Avg Recall Time",   f"{avg_t:.2f}y" if avg_t else "N/A","#f97316")

            t_ax=np.linspace(0,mat_ac,ns)
            fig_t=go.Figure()
            for i in range(min(60,int(n_ac))):
                fig_t.add_trace(go.Scatter(x=t_ax,y=paths[i],mode="lines",
                    line=dict(width=0.6,color="#3b82f6"),opacity=0.3,showlegend=False))
            fig_t.add_hline(y=rb_ac*S0_ac,line_color="#22c55e",line_dash="dash",annotation_text=f"Recall {rb_ac*100:.0f}%")
            fig_t.add_hline(y=pb_ac*S0_ac,line_color="#ef4444",line_dash="dash",annotation_text=f"Protection {pb_ac*100:.0f}%")
            fig_style(fig_t,"Simulated Underlying Paths (sample)")
            fig_t.update_xaxes(title="Time (years)"); fig_t.update_yaxes(title="Level")
            st.plotly_chart(fig_t,use_container_width=True)

            ch1,ch2=st.columns(2)
            with ch1:
                fig_h=go.Figure()
                fig_h.add_trace(go.Histogram(x=payoffs,nbinsx=60,marker_color="#3b82f6",opacity=0.8))
                fig_h.add_vline(x=1.0,line_dash="dash",line_color="#0f172a")
                fig_style(fig_h,"Distribution of Payoffs")
                fig_h.update_xaxes(title="Payoff"); fig_h.update_yaxes(title="Count")
                st.plotly_chart(fig_h,use_container_width=True)
            with ch2:
                fig_rc=go.Figure()
                fig_rc.add_trace(go.Bar(
                    x=[f"T{i+1} ({obs_t[i]:.2f}y)" for i in range(tot_p)],
                    y=[np.mean(rec_p==i+1)*100 for i in range(tot_p)],
                    marker_color="#22c55e"))
                fig_style(fig_rc,"Recall Probability by Observation Date")
                fig_rc.update_xaxes(title="Observation"); fig_rc.update_yaxes(title="Probability (%)")
                st.plotly_chart(fig_rc,use_container_width=True)

            st.dataframe(pd.DataFrame({
                "Scenario":["Early Recall","Full Term — Above Barrier","Below Protection Barrier"],
                "Condition":[f"Spot ≥ {rb_ac*100:.0f}% at any obs.",
                             f"Spot ≥ {pb_ac*100:.0f}% at maturity",
                             f"Spot < {pb_ac*100:.0f}% at maturity"],
                "Payoff":["100% + coupons × n periods",
                          f"100% + {cpn_ac*tot_p*100:.1f}% (full coupons)",
                          "% performance + coupons (capital at risk)"],
                "Probability":[f"{pr_rc*100:.1f}%",
                               f"{(1-pr_rc-pr_lc)*100:.1f}%",
                               f"{pr_lc*100:.1f}%"],
            }), use_container_width=True, hide_index=True)
        else:
            st.info("Set parameters and click **Run Simulation**.")
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 5 — BRC
# ═════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:24px;font-weight:800;color:#0f172a;">Barrier Reverse Convertible</div>
    <div style="font-size:14px;color:#64748b;margin-bottom:16px;">BRC payoff simulation and scenario analysis</div>
    """, unsafe_allow_html=True)
    info_box("A <strong>BRC</strong> pays a high coupon but exposes the investor to capital loss if the underlying "
             "closes below the protection barrier at maturity. Economically = zero-coupon bond + short down-and-in put.")

    cp,cr=st.columns([1,2],gap="large")
    with cp:
        S0_b  = st.number_input("Initial Spot S₀",          value=float(st.session_state["bs_spot"]), min_value=0.01, step=1.0, key="b_s0")
        Sc_b  = st.number_input("Current Spot",             value=float(st.session_state["bs_spot"]), min_value=0.01, step=1.0, key="b_sc")
        Kp_b  = st.number_input("Strike (% of S₀)",         value=100.0, min_value=1.0, max_value=150.0, step=1.0,  key="b_K")   / 100
        bar_b = st.number_input("Protection Barrier (% S₀)",value=70.0,  min_value=1.0, max_value=100.0, step=1.0,  key="b_bar") / 100
        cpn_b = st.number_input("Coupon (%)",                value=8.0,   min_value=0.0, max_value=50.0,  step=0.5,  key="b_cpn") / 100
        T_b   = st.number_input("Maturity (years)",          value=1.0,   min_value=0.1, max_value=5.0,   step=0.1,  key="b_T")
        sig_b = st.number_input("Volatility (%)",            value=float(st.session_state["bs_sigma"]), min_value=0.1, max_value=150.0, step=0.5, key="b_sig") / 100
        r_b   = st.number_input("Risk-free Rate (%)",        value=3.0,   step=0.1, key="b_r") / 100
        q_b   = st.number_input("Dividend Yield (%)",        value=0.0,   min_value=0.0, step=0.1, key="b_q") / 100
        n_b   = st.number_input("Simulations",               value=10000, min_value=1000, max_value=100000, step=1000, key="b_n")
        sd_b  = st.number_input("Seed (0=random)",           value=42,    min_value=0,   step=1, key="b_sd")
        run_b = st.button("Run Simulation", type="primary", key="b_run")

    with cr:
        sr=np.linspace(max(0.01,S0_b*0.3),S0_b*1.5,500)
        pa=np.where(sr>=Kp_b*S0_b,1+cpn_b,np.where(sr>=bar_b*S0_b,1+cpn_b,sr/S0_b+cpn_b))
        fig_p=go.Figure()
        fig_p.add_trace(go.Scatter(x=sr,y=pa,name="BRC Payoff",
            line=dict(color="#3b82f6",width=2.5),fill="tozeroy",fillcolor="rgba(59,130,246,0.05)"))
        fig_p.add_hline(y=1.0,line_dash="dash",line_color="#0f172a",annotation_text="Capital = 100%")
        fig_p.add_hline(y=1+cpn_b,line_dash="dot",line_color="#22c55e",annotation_text=f"Max = {(1+cpn_b)*100:.1f}%")
        fig_p.add_vline(x=bar_b*S0_b,line_dash="dash",line_color="#ef4444",annotation_text=f"Barrier {bar_b*100:.0f}%")
        fig_p.add_vline(x=Kp_b*S0_b, line_dash="dot", line_color="#94a3b8",annotation_text=f"Strike {Kp_b*100:.0f}%")
        fig_style(fig_p,"BRC Payoff at Maturity (analytical)")
        fig_p.update_xaxes(title="Spot at Maturity"); fig_p.update_yaxes(title="Payoff (per unit nominal)")
        st.plotly_chart(fig_p,use_container_width=True)

        if run_b:
            rng_b=np.random.default_rng(sd_b if sd_b>0 else None)
            ns_b=max(int(T_b*252),50)
            Z_b=rng_b.standard_normal((int(n_b),ns_b))
            pths=Sc_b*np.exp(np.cumsum((r_b-q_b-0.5*sig_b**2)*(T_b/ns_b)+sig_b*np.sqrt(T_b/ns_b)*Z_b,axis=1))
            sf_b=pths[:,-1]
            pb=np.where(sf_b>=Kp_b*S0_b,1+cpn_b,np.where(sf_b>=bar_b*S0_b,1+cpn_b,sf_b/S0_b+cpn_b))
            pl_=np.mean(pb<1.0); pb_=np.mean(sf_b<bar_b*S0_b); mp_=np.mean(pb)
            m1,m2,m3,m4=st.columns(4)
            with m1: metric_card("Mean Payoff",     f"{mp_:.4f}","#3b82f6")
            with m2: metric_card("Exp. Return",     f"{(mp_-1)*100:.2f}%","#22c55e" if mp_>=1 else "#ef4444")
            with m3: metric_card("Prob. Below Barrier",f"{pb_*100:.1f}%","#ef4444")
            with m4: metric_card("Prob. Capital Loss", f"{pl_*100:.1f}%","#ef4444")
            fig_bh=go.Figure()
            fig_bh.add_trace(go.Histogram(x=pb,nbinsx=60,marker_color="#3b82f6",opacity=0.8))
            fig_bh.add_vline(x=1.0,line_dash="dash",line_color="#0f172a")
            fig_bh.add_vline(x=1+cpn_b,line_dash="dot",line_color="#22c55e")
            fig_style(fig_bh,"Distribution of BRC Payoffs")
            fig_bh.update_xaxes(title="Payoff"); fig_bh.update_yaxes(title="Count")
            st.plotly_chart(fig_bh,use_container_width=True)
            st.dataframe(pd.DataFrame({
                "Scenario":["Favorable","Neutral","Adverse"],
                "Condition":[f"Final ≥ Strike ({Kp_b*100:.0f}%)",
                             "Final between Barrier & Strike",
                             f"Final < Barrier ({bar_b*100:.0f}%)"],
                "Payoff":[f"100% + {cpn_b*100:.1f}% coupon",
                          f"100% + {cpn_b*100:.1f}% coupon",
                          f"S_T/S₀ + {cpn_b*100:.1f}% coupon"],
                "Probability":[f"{np.mean(sf_b>=Kp_b*S0_b)*100:.1f}%",
                               f"{np.mean((sf_b>=bar_b*S0_b)&(sf_b<Kp_b*S0_b))*100:.1f}%",
                               f"{pb_*100:.1f}%"],
            }), use_container_width=True, hide_index=True)
        else:
            st.info("Set parameters and click **Run Simulation**.")
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 6 — MC PRICER / PRODUCT DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:24px;font-weight:800;color:#0f172a;">Structured Product Decomposition</div>
    <div style="font-size:14px;color:#64748b;margin-bottom:16px;">Pedagogical breakdown of structured product building blocks</div>
    """, unsafe_allow_html=True)
    st.markdown("""<div style="background:#fffbeb;border-left:4px solid #f59e0b;border-radius:6px;
    padding:14px 18px;font-size:13px;color:#78350f;margin-bottom:20px;">
    <strong>Note:</strong> Simplified pedagogical approximations — not exact institutional replication portfolios.</div>""",
    unsafe_allow_html=True)

    prod=st.selectbox("Select Product",["Capital-Protected Note","Reverse Convertible","Autocall (simplified)"])
    cd,cv=st.columns([1,2],gap="large")
    with cd:
        S_dc =st.number_input("Spot S₀",          value=float(st.session_state["bs_spot"]),  min_value=0.01, step=1.0, key="dc_s")
        r_dc =st.number_input("Risk-free Rate (%)",value=3.0, step=0.1, key="dc_r") / 100
        T_dc =st.number_input("Maturity (years)",  value=3.0, min_value=0.5, max_value=10.0, step=0.5, key="dc_T")
        sig_dc=st.number_input("Volatility (%)",   value=float(st.session_state["bs_sigma"]), min_value=0.1, step=0.5, key="dc_sig") / 100
        q_dc =st.number_input("Dividend Yield (%)",value=0.0, min_value=0.0, step=0.1, key="dc_q") / 100

    zc=np.exp(-r_dc*T_dc); budget=1.0-zc
    with cd:
        st.markdown(f"""<div style="background:white;border-radius:12px;padding:20px;
        box-shadow:0 1px 4px rgba(0,0,0,0.06);margin-top:16px;">
          <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
            <span style="font-size:13px;color:#64748b;">Zero-Coupon Bond</span>
            <span style="font-weight:700;color:#0f172a;">{zc:.4f}</span></div>
          <div style="display:flex;justify-content:space-between;">
            <span style="font-size:13px;color:#64748b;">Option Budget</span>
            <span style="font-weight:700;color:#22c55e;">{budget:.4f}</span></div>
        </div>""", unsafe_allow_html=True)

    sr_dc=np.linspace(S_dc*0.4,S_dc*1.8,400)
    with cv:
        if prod=="Capital-Protected Note":
            cp_dc=bs_price(S_dc,S_dc,T_dc,sig_dc,r_dc,q_dc,"Call")
            part=min(budget/cp_dc,2.5) if cp_dc>0 else 0
            pay_dc=zc*100+part*np.maximum(sr_dc-S_dc,0)
            st.markdown(f"""**Capital-Protected Note = Zero-Coupon Bond + Call Option**

| Component | Value |
|---|---|
| Zero-Coupon Bond | {zc:.4f} — guarantees capital |
| ATM Call Option  | {cp_dc:.4f} — captures upside |
| Participation Rate | **{part:.2f}×** |

**Investor:** Floor = {zc*100:.1f}%, participates {part:.2f}× above S₀.
""")
            fig_dc=go.Figure()
            fig_dc.add_trace(go.Scatter(x=sr_dc,y=pay_dc,name="CPN",line=dict(color="#3b82f6",width=2.5)))
            fig_dc.add_hline(y=100*zc,line_dash="dash",line_color="#22c55e",annotation_text=f"Floor {zc*100:.1f}%")

        elif prod=="Reverse Convertible":
            pp_dc=bs_price(S_dc,S_dc,T_dc,sig_dc,r_dc,q_dc,"Put")
            impl_cpn=pp_dc/zc
            pay_dc=100*np.where(sr_dc>=S_dc,1.0,sr_dc/S_dc)
            st.markdown(f"""**Reverse Convertible = Zero-Coupon Bond + Short ATM Put**

| Component | Value |
|---|---|
| Zero-Coupon Bond | {zc:.4f} |
| Short ATM Put    | +{pp_dc:.4f} — funds the coupon |
| Implied Coupon   | **{impl_cpn*100:.2f}%** |

**Investor:** Coupon received, but bears put risk below strike.
""")
            fig_dc=go.Figure()
            fig_dc.add_trace(go.Scatter(x=sr_dc,y=pay_dc+impl_cpn*100,name="RC",line=dict(color="#ef4444",width=2.5)))
            fig_dc.add_hline(y=100,line_dash="dash",line_color="#0f172a")

        else:
            d1_tmp,d2_tmp=bs_d1_d2(S_dc,S_dc,T_dc/2,sig_dc,r_dc,q_dc)
            dig=zc*norm.cdf(d2_tmp) if d2_tmp else 0
            put_di=bs_price(S_dc,S_dc*0.7,T_dc,sig_dc,r_dc,q_dc,"Put")*0.6
            pay_dc=100*np.where(sr_dc>=S_dc,1.05,np.where(sr_dc>=S_dc*0.7,1.0,sr_dc/S_dc))
            st.markdown(f"""**Autocall = Zero-Coupon Bond + Conditional Digitals + Short PDI Put**

| Component | Approx. Value |
|---|---|
| Zero-Coupon Bond | {zc:.4f} |
| Digital Calls (recall) | ~{dig:.4f} |
| Short Down-and-In Put  | ~{put_di:.4f} |

**Investor:** Coupon remunerates vol sold + barrier risk.
*Simplified — actual autocalls require dynamic delta hedging.*
""")
            fig_dc=go.Figure()
            fig_dc.add_trace(go.Scatter(x=sr_dc,y=pay_dc,name="Autocall",line=dict(color="#f97316",width=2.5)))
            fig_dc.add_vline(x=S_dc*0.7,line_dash="dash",line_color="#ef4444",annotation_text="Barrier 70%")

        fig_dc.add_vline(x=S_dc,line_dash="dot",line_color="#94a3b8",annotation_text="S₀")
        fig_dc.add_hline(y=100,line_dash="dash",line_color="#0f172a",line_width=1)
        fig_style(fig_dc,f"{prod} — Payoff at Maturity")
        fig_dc.update_xaxes(title="Spot at Maturity"); fig_dc.update_yaxes(title="Payoff (%)")
        st.plotly_chart(fig_dc,use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 7 — SENSITIVITIES
# ═════════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:24px;font-weight:800;color:#0f172a;">Market Sensitivities</div>
    <div style="font-size:14px;color:#64748b;margin-bottom:24px;">Live volatility surface, macro scenarios and option sensitivity charts</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border-radius:12px;padding:24px 28px;margin-bottom:24px;
    box-shadow:0 1px 4px rgba(0,0,0,0.06),0 6px 18px rgba(15,35,63,0.05);">
      <div style="font-size:18px;font-weight:800;color:#0f172a;margin-bottom:6px;">Live Volatility Surface</div>
      <div style="font-size:13px;color:#64748b;margin-bottom:16px;">
        Yahoo option-chain implied volatility by strike and maturity. Works best on liquid US tickers such as AAPL, MSFT, NVDA, SPY.
      </div>
    """, unsafe_allow_html=True)

    vs_c1, vs_c2, vs_c3 = st.columns([1.4, 1, 1])
    with vs_c1:
        surface_ticker = st.text_input(
            "Surface ticker",
            value=st.session_state.get("bs_ticker") or "AAPL",
            key="surface_ticker",
        )
    with vs_c2:
        max_exp = st.slider("Expiries", 3, 12, 8, 1, key="surface_expiries")
    with vs_c3:
        opt_side = st.selectbox("Chain side", ["Call", "Put", "Both"], key="surface_side")

    if st.button("Load Vol Surface", type="primary", use_container_width=True):
        st.session_state["surface_loaded"] = True

    if st.session_state.get("surface_loaded"):
        try:
            source_label = "Yahoo option chain"
            try:
                surface_df, surface_spot, expiries = fetch_option_surface(surface_ticker, max_exp)
            except Exception as yahoo_err:
                surface_df, surface_spot, expiries = fetch_estimated_surface(surface_ticker, max_exp)
                source_label = f"Estimated from historical volatility (Yahoo IV unavailable: {yahoo_err})"

            plot_df = surface_df if opt_side == "Both" else surface_df[surface_df["type"] == opt_side]
            if plot_df.empty:
                raise ValueError("No option rows for this side after cleaning.")

            atm_df = plot_df[(plot_df["moneyness"] >= 0.75) & (plot_df["moneyness"] <= 1.25)].copy()
            smile_df = atm_df.sort_values(["dte", "strike"])

            st.markdown(f"""
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;
            padding:12px 16px;margin:12px 0 16px 0;font-size:13px;color:#1e40af;">
              <strong>Surface source:</strong> {source_label}
            </div>
            """, unsafe_allow_html=True)

            sm1, sm2, sm3 = st.columns(3)
            with sm1:
                metric_card("Spot", f"{surface_spot:,.2f}", "#3b82f6")
            with sm2:
                metric_card("Expiries loaded", str(len(sorted(plot_df["expiry"].unique()))), "#22c55e")
            with sm3:
                metric_card("IV points", f"{len(plot_df):,}", "#f97316")

            fig_smile = go.Figure()
            for expiry in sorted(smile_df["expiry"].unique())[:8]:
                sub = smile_df[smile_df["expiry"] == expiry]
                fig_smile.add_trace(go.Scatter(
                    x=sub["moneyness"], y=sub["iv_pct"], mode="lines+markers",
                    name=f"{expiry}", line=dict(width=2)
                ))
            fig_style(fig_smile, f"{surface_ticker.upper()} IV Smile / Skew")
            fig_smile.update_xaxes(title="Moneyness K / Spot")
            fig_smile.update_yaxes(title="Implied Volatility (%)")
            fig_smile.add_vline(x=1.0, line_dash="dash", line_color="#0f172a", annotation_text="ATM")
            st.plotly_chart(fig_smile, use_container_width=True)

            grid = (
                plot_df.assign(moneyness_bucket=(plot_df["moneyness"] * 100).round() / 100)
                .groupby(["dte", "moneyness_bucket"], as_index=False)["iv_pct"].mean()
            )
            pivot = grid.pivot(index="dte", columns="moneyness_bucket", values="iv_pct").sort_index()

            surf_l, surf_r = st.columns(2)
            with surf_l:
                fig_surface = go.Figure(data=[go.Surface(
                    x=pivot.columns.values,
                    y=pivot.index.values,
                    z=pivot.values,
                    colorscale="Blues",
                    colorbar=dict(title="IV %"),
                )])
                fig_surface.update_layout(
                    **BASE_LAYOUT,
                    height=520,
                    title=dict(text="3D IV Surface", font=dict(size=14, color="#0f172a")),
                    scene=dict(
                        xaxis_title="Moneyness",
                        yaxis_title="Days to Expiry",
                        zaxis_title="IV %",
                    ),
                )
                st.plotly_chart(fig_surface, use_container_width=True)
            with surf_r:
                fig_heat = go.Figure(data=go.Heatmap(
                    x=pivot.columns.values,
                    y=pivot.index.values,
                    z=pivot.values,
                    colorscale="Blues",
                    colorbar=dict(title="IV %"),
                ))
                fig_style(fig_heat, "IV Heatmap")
                fig_heat.update_xaxes(title="Moneyness")
                fig_heat.update_yaxes(title="Days to Expiry")
                st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as err:
            st.error(f"Vol surface unavailable for {surface_ticker.upper()}: {err}")

    st.markdown("</div>", unsafe_allow_html=True)

    cm,cr=st.columns([1,3],gap="large")
    with cm:
        ot_m =st.selectbox("Option Type",["Call","Put"],key="ms_ot")
        S_m  =st.number_input("Spot",      value=float(st.session_state["bs_spot"]),  min_value=0.01, step=1.0, key="ms_s")
        K_m  =st.number_input("Strike",    value=float(st.session_state["bs_spot"]),  min_value=0.01, step=1.0, key="ms_k")
        T_m  =st.number_input("Maturity (yr)", value=1.0, min_value=0.01, max_value=10.0, step=0.05, key="ms_T")
        sig_m=st.number_input("Volatility (%)",value=float(st.session_state["bs_sigma"]), min_value=0.1, max_value=150.0, step=0.5, key="ms_sig") / 100
        r_m  =st.number_input("Rate (%)",  value=3.0, step=0.1, key="ms_r") / 100
        q_m  =st.number_input("Div. Yield (%)", value=0.0, min_value=0.0, step=0.1, key="ms_q") / 100

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
        st.markdown("**Macro Scenario Shocks**")
        bull_spot = st.number_input("Bull spot shock (%)", value=12.0, step=1.0, key="macro_bull_spot") / 100
        bear_spot = st.number_input("Bear spot shock (%)", value=-12.0, step=1.0, key="macro_bear_spot") / 100
        bull_vol = st.number_input("Bull vol shock (pts)", value=-3.0, step=0.5, key="macro_bull_vol") / 100
        bear_vol = st.number_input("Bear vol shock (pts)", value=8.0, step=0.5, key="macro_bear_vol") / 100
        rate_shock = st.number_input("Rate shock (pts)", value=0.5, step=0.1, key="macro_rate_shock") / 100

    with cr:
        scen_rows = []
        base_price = bs_price(S_m, K_m, T_m, sig_m, r_m, q_m, ot_m)
        scenarios = [
            ("Bull", S_m * (1 + bull_spot), max(sig_m + bull_vol, 0.001), r_m + rate_shock, "#22c55e"),
            ("Base", S_m, sig_m, r_m, "#3b82f6"),
            ("Bear", S_m * (1 + bear_spot), max(sig_m + bear_vol, 0.001), r_m - rate_shock, "#ef4444"),
        ]
        for name, s_sc, v_sc, r_sc, color in scenarios:
            price_sc = bs_price(s_sc, K_m, T_m, v_sc, r_sc, q_m, ot_m)
            greeks_sc = bs_greeks(s_sc, K_m, T_m, v_sc, r_sc, q_m, ot_m)
            scen_rows.append({
                "Scenario": name,
                "Spot": s_sc,
                "Vol (%)": v_sc * 100,
                "Rate (%)": r_sc * 100,
                "Price": price_sc,
                "vs Base": price_sc - base_price,
                "Delta": greeks_sc["Delta"],
                "Vega": greeks_sc["Vega"],
            })
        scen_df = pd.DataFrame(scen_rows)

        st.markdown("""
        <div style="font-size:18px;font-weight:800;color:#0f172a;margin-bottom:6px;">Bull / Base / Bear Macro Scenarios</div>
        <div style="font-size:13px;color:#64748b;margin-bottom:12px;">Compare the selected option price under spot, volatility and rate assumptions.</div>
        """, unsafe_allow_html=True)
        sleft, sright = st.columns([1.1, 1])
        with sleft:
            st.dataframe(
                scen_df.style.format({
                    "Spot": "{:,.2f}",
                    "Vol (%)": "{:.2f}",
                    "Rate (%)": "{:.2f}",
                    "Price": "{:,.4f}",
                    "vs Base": "{:+,.4f}",
                    "Delta": "{:+.4f}",
                    "Vega": "{:.4f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        with sright:
            fig_macro = go.Figure()
            fig_macro.add_trace(go.Bar(
                x=scen_df["Scenario"], y=scen_df["Price"], name="Option Price",
                marker_color=["#22c55e", "#3b82f6", "#ef4444"],
                text=[f"{v:.2f}" for v in scen_df["Price"]],
                textposition="outside",
            ))
            fig_style(fig_macro, f"{ot_m} Price by Macro Scenario")
            fig_macro.update_yaxes(title="Option Price")
            st.plotly_chart(fig_macro, use_container_width=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        tv,ts,tt,tr,tg=st.tabs(["Price vs Vol","Price vs Spot","Price vs Time","Price vs Rate","Greeks vs Spot"])
        with tv:
            vols=np.linspace(0.01,1.0,300)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=vols*100,y=[bs_price(S_m,K_m,T_m,v,r_m,q_m,ot_m) for v in vols],
                line=dict(color="#3b82f6",width=2.5),name=ot_m))
            fig.add_vline(x=sig_m*100,line_dash="dash",line_color="#f97316",annotation_text="σ")
            fig_style(fig,f"{ot_m} Price vs Volatility")
            fig.update_xaxes(title="Volatility (%)"); fig.update_yaxes(title="Price")
            st.plotly_chart(fig,use_container_width=True)
            info_box("Vega is highest for ATM options. Implied volatility is the market's price of uncertainty.")
        with ts:
            sps=np.linspace(max(0.1,S_m*0.4),S_m*1.8,300)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=sps,y=[bs_price(s,K_m,T_m,sig_m,r_m,q_m,ot_m) for s in sps],
                line=dict(color="#22c55e",width=2.5),name=ot_m))
            fig.add_vline(x=S_m,line_dash="dash",line_color="#94a3b8",annotation_text="Spot")
            fig.add_vline(x=K_m,line_dash="dot", line_color="#3b82f6",annotation_text="Strike")
            fig_style(fig,f"{ot_m} Price vs Spot")
            fig.update_xaxes(title="Spot"); fig.update_yaxes(title="Price")
            st.plotly_chart(fig,use_container_width=True)
            info_box("Delta is the slope of this curve. Gamma is the curvature — peaks near the strike.")
        with tt:
            tms=np.linspace(0.01,max(T_m*2,2.0),300)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=tms,y=[bs_price(S_m,K_m,t,sig_m,r_m,q_m,ot_m) for t in tms],
                line=dict(color="#ef4444",width=2.5),name=ot_m))
            fig.add_vline(x=T_m,line_dash="dash",line_color="#f97316",annotation_text="T")
            fig_style(fig,f"{ot_m} Price vs Time to Maturity")
            fig.update_xaxes(title="Time (years)"); fig.update_yaxes(title="Price")
            st.plotly_chart(fig,use_container_width=True)
            info_box("Theta erodes time value daily. Decay accelerates near expiry.")
        with tr:
            rts=np.linspace(-0.05,0.15,300)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=rts*100,y=[bs_price(S_m,K_m,T_m,sig_m,r_,q_m,ot_m) for r_ in rts],
                line=dict(color="#8b5cf6",width=2.5),name=ot_m))
            fig.add_vline(x=r_m*100,line_dash="dash",line_color="#f97316",annotation_text="r")
            fig_style(fig,f"{ot_m} Price vs Risk-Free Rate")
            fig.update_xaxes(title="Rate (%)"); fig.update_yaxes(title="Price")
            st.plotly_chart(fig,use_container_width=True)
            info_box("Rho matters most for long-dated products. Higher rates increase calls, decrease puts.")
        with tg:
            sps_g=np.linspace(max(0.1,S_m*0.5),S_m*1.6,300)
            gv={n:[bs_greeks(s,K_m,T_m,sig_m,r_m,q_m,ot_m)[n] for s in sps_g]
                for n in ["Delta","Gamma","Vega","Theta"]}
            fig_g=make_subplots(rows=2,cols=2,
                subplot_titles=["Delta vs Spot","Gamma vs Spot","Vega vs Spot","Theta vs Spot"])
            for name,color,row,col_ in [("Delta","#3b82f6",1,1),("Gamma","#8b5cf6",1,2),
                                         ("Vega","#f97316",2,1),("Theta","#ef4444",2,2)]:
                fig_g.add_trace(go.Scatter(x=sps_g,y=gv[name],name=name,
                    line=dict(color=color,width=2)),row=row,col=col_)
                fig_g.add_vline(x=S_m,line_dash="dash",line_color="#94a3b8",row=row,col=col_)
            fig_g.update_layout(**BASE_LAYOUT,height=560,
                title=dict(text="Greeks vs Spot",font=dict(size=14,color="#0f172a")))
            st.plotly_chart(fig_g,use_container_width=True)
            info_box("Delta = directional exposure. Gamma peaks ATM. Vega drives structured product pricing. Theta = daily time decay cost.")
    st.markdown("</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 8 — LEARNINGS
# ═════════════════════════════════════════════════════════════════════════════
with tabs[8]:
    st.markdown('<div style="padding:32px 40px;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:24px;font-weight:800;color:#0f172a;">Key Learnings</div>
    <div style="font-size:14px;color:#64748b;margin-bottom:28px;">From quantitative models to structured finance practice</div>
    """, unsafe_allow_html=True)

    cards_l=[
        ("#3b82f6","Replication & No-Arbitrage",[
            "BS replication links Itô calculus to a concrete self-financing portfolio (Δ·S − B)",
            "Drift cancels under ℚ — only realized vs implied vol drives Gamma P&L",
            "Model limits (constant vol, no jumps) motivate local vol and SABR in production"]),
        ("#22c55e","Structured Product Architecture",[
            "Autocall = conditional digitals + short PDI put. Coupon = premium from selling barrier",
            "BRC = zero-coupon bond + short KI put. KI put premium = client coupon",
            "Issuers hedge via OTC vanilla puts — their flow systematically impacts the skew"]),
        ("#8b5cf6","Greeks as Risk Language",[
            "Delta = hedge ratio, not a probability. N(d₂) is the risk-neutral ITM probability",
            "Gamma and Theta are linked: long convexity = paying daily time decay",
            "Vega in structured products: BRC/Autocall issuers are structurally short Vega across the surface"]),
        ("#f97316","Desk Relevance",[
            "Structuring: decompose payoffs into option building blocks, price from vol surface",
            "Sales Trading: translate Greeks and scenarios into client-facing language in real-time",
            "Pricing Support: Monte Carlo for path-dependent products, scenario analysis for risk reporting"]),
    ]
    c1,c2=st.columns(2,gap="large")
    for i,(color,title,bullets) in enumerate(cards_l):
        tgt=c1 if i%2==0 else c2
        with tgt:
            items="".join([f'<li style="margin-bottom:8px;color:#374151;line-height:1.55;">{b}</li>' for b in bullets])
            st.markdown(f"""
            <div style="background:white;border-radius:12px;padding:24px 28px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.04);
            border-top:4px solid {color};margin-bottom:20px;">
              <div style="font-size:15px;font-weight:700;color:#0f172a;margin-bottom:14px;">{title}</div>
              <ul style="margin:0;padding-left:18px;list-style-type:disc;">{items}</ul>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#0d1b2a;border-radius:16px;padding:40px 48px;text-align:center;margin-top:8px;">
      <div style="font-size:20px;font-weight:700;color:white;margin-bottom:12px;">
        Looking for a Market Finance Internship</div>
      <div style="font-size:14px;color:rgba(255,255,255,0.6);line-height:1.7;max-width:600px;margin:0 auto 20px auto;">
        This project is part of my preparation for a market finance internship in
        <strong style="color:white;">Sales, Structuring, Trading Support or Pricing</strong>.</div>
      <div style="display:flex;justify-content:center;gap:10px;flex-wrap:wrap;">
        <span style="background:rgba(255,255,255,0.12);color:rgba(255,255,255,0.85);border-radius:20px;padding:5px 16px;font-size:13px;font-weight:600;">Python</span>
        <span style="background:rgba(255,255,255,0.12);color:rgba(255,255,255,0.85);border-radius:20px;padding:5px 16px;font-size:13px;font-weight:600;">NumPy · Pandas · SciPy</span>
        <span style="background:rgba(255,255,255,0.12);color:rgba(255,255,255,0.85);border-radius:20px;padding:5px 16px;font-size:13px;font-weight:600;">Streamlit</span>
        <span style="background:rgba(255,255,255,0.12);color:rgba(255,255,255,0.85);border-radius:20px;padding:5px 16px;font-size:13px;font-weight:600;">Plotly</span>
        <span style="background:rgba(255,255,255,0.12);color:rgba(255,255,255,0.85);border-radius:20px;padding:5px 16px;font-size:13px;font-weight:600;">yfinance</span>
        <span style="background:rgba(255,255,255,0.12);color:rgba(255,255,255,0.85);border-radius:20px;padding:5px 16px;font-size:13px;font-weight:600;">Monte Carlo</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
