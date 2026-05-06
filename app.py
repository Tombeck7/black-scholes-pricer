import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from bs_core import (
    BSOption, call_price, put_price,
    delta, gamma, vega, theta, rho, vanna, volga, charm, speed,
)
from implied_vol import implied_vol, forward_vol
from vol_surface import realistic_smile
from payoff_analysis import (
    long_call, long_put, long_straddle, long_strangle,
    short_straddle, short_strangle,
    bull_call_spread, bear_put_spread, bull_put_spread, bear_call_spread,
    butterfly_calls, iron_condor, risk_reversal,
    collar, covered_call, protective_put,
    preexpiry_pnl_straddle, straddle_breakevens,
)
from live_data import fetch, risk_free_rate

# ── Design tokens ─────────────────────────────────────────────────────────────
C = dict(
    primary   = "#6366f1",
    success   = "#22c55e",
    danger    = "#ef4444",
    warning   = "#f59e0b",
    info      = "#38bdf8",
    purple    = "#a855f7",
    pink      = "#ec4899",
    text      = "#e2e8f0",
    muted     = "#94a3b8",
    grid      = "rgba(255,255,255,0.06)",
    line      = "rgba(255,255,255,0.12)",
    bg        = "rgba(0,0,0,0)",
)

def sf(fig, title="", height=420, legend_h=True):
    """Apply consistent style to a Plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=C["text"]), x=0.01),
        template="plotly_dark",
        height=height,
        paper_bgcolor=C["bg"],
        plot_bgcolor=C["bg"],
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=C["text"]),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor=C["line"], font=dict(size=11, color=C["text"])),
        margin=dict(t=45 if title else 20, b=36, l=52, r=16),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
            orientation="h" if legend_h else "v",
            y=1.06 if legend_h else 1,
            font=dict(size=10),
        ),
    )
    fig.update_xaxes(
        gridcolor=C["grid"], linecolor=C["line"],
        zeroline=False, showgrid=True,
    )
    fig.update_yaxes(
        gridcolor=C["grid"], linecolor=C["line"],
        zeroline=False, showgrid=True,
    )
    return fig

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black-Scholes Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }



div[data-testid="stMetricValue"]  { font-size: 1.5rem; font-weight: 700; }
div[data-testid="stMetricLabel"]  { font-size: 0.75rem; color: #94a3b8; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; }
div[data-testid="stMetricDelta"]  { font-size: 0.8rem; }

.live-banner {
    background: linear-gradient(90deg, #0f2027, #203a43, #2c5364);
    border: 1px solid #38bdf8;
    border-radius: 10px;
    padding: 12px 18px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.ticker-pill {
    background: #6366f1;
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 700;
    font-size: .85rem;
    letter-spacing: .05em;
}
.price-big { font-size: 1.6rem; font-weight: 700; color: #e2e8f0; }
.chg-pos   { color: #22c55e; font-weight: 600; }
.chg-neg   { color: #ef4444; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
for k, v in dict(S=100.0, K=100.0, r=5.0, sigma=20.0, q=2.0, live=None).items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Black-Scholes Pricer")
st.sidebar.divider()

# Live data
st.sidebar.subheader("🔴 Live Market Data")
col_t, col_b = st.sidebar.columns([3, 1])
ticker_input = col_t.text_input("Ticker", value="SPY", label_visibility="collapsed",
                                 placeholder="SPY, AAPL, MSFT...")
if col_b.button("Load", use_container_width=True):
    with st.spinner(f"Fetching {ticker_input.upper()}..."):
        data = fetch(ticker_input.strip().upper())
    if data:
        st.session_state.live  = data
        st.session_state.S     = round(data["price"], 2)
        st.session_state.K     = round(data["price"], 2)
        st.session_state.sigma = round(data["sigma_3m"] * 100, 1)
        try:
            st.session_state.r = round(risk_free_rate() * 100, 2)
        except Exception:
            pass
        st.sidebar.success(f"{data['ticker']}  {data['price']:.2f}  ({data['change_pct']:+.2f}%)")
    else:
        st.sidebar.error("Ticker not found.")

st.sidebar.divider()
st.sidebar.subheader("⚙️ Parameters")

S     = st.sidebar.number_input("Spot  S",  1.0, 10000.0, float(st.session_state.S), 1.0)
K     = st.sidebar.number_input("Strike K", 1.0, 10000.0, float(st.session_state.K), 1.0)
T     = st.sidebar.slider("Maturity T (years)", 0.01, 5.0, 1.0, 0.01)
r     = st.sidebar.slider("Risk-free r (%)",    0.0, 15.0, float(st.session_state.r),   0.1) / 100
sigma = st.sidebar.slider("Volatility σ (%)",   1.0, 120.0, float(st.session_state.sigma), 0.5) / 100
q     = st.sidebar.slider("Dividend q (%)",     0.0, 15.0,  2.0, 0.1) / 100

st.sidebar.subheader("Option type")
opt_type = st.sidebar.radio("Type", ["Call", "Put"], horizontal=True,
                             label_visibility="collapsed").lower()

# ── Live banner ───────────────────────────────────────────────────────────────
live = st.session_state.live
if live:
    chg_class = "chg-pos" if live["change"] >= 0 else "chg-neg"
    chg_sign  = "+" if live["change"] >= 0 else ""
    beta_str = f" &nbsp; Beta: <b style='color:#e2e8f0'>{live['beta']:.2f}</b>" if live.get('beta') else ""
    hv_str   = f"HV 1M <b style='color:#e2e8f0'>{live['sigma_1m']*100:.1f}%</b> &nbsp; HV 3M <b style='color:#e2e8f0'>{live['sigma_3m']*100:.1f}%</b>{beta_str}"
    st.markdown(
        f"<div class='live-banner'><span class='ticker-pill'>{live['ticker']}</span>&nbsp;"
        f"<span class='price-big'>{live['price']:.2f} {live['currency']}</span>&nbsp;"
        f"<span class='{chg_class}'>{chg_sign}{live['change_pct']:.2f}%</span>&nbsp;&nbsp;"
        f"<span style='color:#94a3b8;font-size:.85rem'>{hv_str}</span></div>",
        unsafe_allow_html=True,
    )

# ── Pre-compute core objects ──────────────────────────────────────────────────
call = BSOption(S, K, T, r, sigma, q, 'call')
put  = BSOption(S, K, T, r, sigma, q, 'put')
opt  = call if opt_type == 'call' else put

# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊 Pricing & Greeks",
    "📈 Greeks Profiles",
    "🔍 Implied Vol",
    "🌐 Vol Surface",
    "💼 Strategy Payoffs",
    "🏗️ Structured Products",
    "📐 Delta Hedging",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Pricing & Greeks
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.header("European Option Pricing & Analytical Greeks")

    # ── Top KPIs ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Call Price",   f"{call.price:.4f}")
    k2.metric("Put Price",    f"{put.price:.4f}")
    pcp = call.price - put.price
    k3.metric("C - P (parity check)", f"{pcp:.4f}",
              delta=f"theory: {S*np.exp(-q*T) - K*np.exp(-r*T):.4f}")
    k4.metric("Intrinsic Value", f"{opt.intrinsic_value:.4f}")
    k5.metric("Time Value",      f"{opt.time_value:.4f}")

    st.divider()

    # ── Greeks table ──
    col_c, col_p = st.columns(2)

    def greek_table(opt_obj, label):
        g = opt_obj.greeks_summary()
        rows = []
        meta = {
            "Price":  ("Option fair value", ""),
            "Delta":  ("dP / dS",           "spot sensitivity"),
            "Gamma":  ("d²P / dS²",         "curvature — same call/put"),
            "Vega":   ("dP / dσ",           "per 1 % vol move"),
            "Theta":  ("dP / dt",           "per calendar day"),
            "Rho":    ("dP / dr",           "per 1 % rate move"),
            "Vanna":  ("dΔ / dσ",           "cross sensitivity"),
            "Volga":  ("dVega / dσ",        "vol convexity"),
            "Charm":  ("dΔ / dt",           "delta decay / day"),
        }
        for name, (formula, note) in meta.items():
            v = g[name]
            arrow = "▲" if v > 0 else ("▼" if v < 0 else "—")
            rows.append({"Greek": name, "Formula": formula,
                         "Value": f"{v:+.6f}", "Note": note})
        import pandas as pd
        df = pd.DataFrame(rows).set_index("Greek")
        return df

    import pandas as pd

    with col_c:
        st.subheader("📗 Call Greeks")
        st.dataframe(greek_table(call, "Call"), width='stretch')

    with col_p:
        st.subheader("📕 Put Greeks")
        st.dataframe(greek_table(put, "Put"), width='stretch')

    st.divider()

    # ── d1 / d2 / moneyness details ──
    from bs_core import d1 as _d1, d2 as _d2
    d1v = _d1(S, K, T, r, sigma, q)
    d2v = _d2(S, K, T, r, sigma, q)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("d1",          f"{d1v:.6f}")
    c2.metric("d2",          f"{d2v:.6f}")
    c3.metric("Moneyness S/K", f"{S/K:.4f}")
    c4.metric("log(S/K)",    f"{np.log(S/K):.4f}")

    st.caption(
        f"Forward F = S·e^(r-q)T = **{S * np.exp((r-q)*T):.4f}** | "
        f"PV(K) = K·e^(-rT) = **{K * np.exp(-r*T):.4f}**"
    )


    # -- Market data panel ------------------------------------------------
    if live:
        st.divider()
        st.subheader(f"📡 {live['ticker']} — Live Market Data")

        # Row 1: key metrics
        r1 = st.columns(6)
        r1[0].metric('Price',      f"{live['price']:.2f} {live['currency']}",
                     f"{live['change_pct']:+.2f}%")
        r1[1].metric('Day range',  f"{live.get('low_day', live['price']):.2f} / {live.get('high_day', live['price']):.2f}")
        r1[2].metric('52W range',  f"{live.get('low_52w', 0):.2f} / {live.get('high_52w', 0):.2f}")
        r1[3].metric('HV 1M',      f"{live['sigma_1m']*100:.1f}%")
        r1[4].metric('HV 3M',      f"{live['sigma_3m']*100:.1f}%")
        r1[5].metric('HV 1Y',      f"{live['sigma_1y']*100:.1f}%")

        # Row 2: IV + fundamentals
        r2 = st.columns(6)
        if live.get('iv_atm_near'):
            r2[0].metric('IV ATM (near exp)', f"{live['iv_atm_near']*100:.1f}%",
                         help=f"Expiry: {live['iv_chain'].get('exp_near','')}")
        if live.get('iv_atm_far'):
            r2[1].metric('IV ATM (far exp)',  f"{live['iv_atm_far']*100:.1f}%",
                         help=f"Expiry: {live['iv_chain'].get('exp_far','')}")
        if live.get('pe_ratio'):
            r2[2].metric('P/E (trailing)', f"{live['pe_ratio']:.1f}x")
        if live.get('fwd_pe'):
            r2[3].metric('P/E (forward)',  f"{live['fwd_pe']:.1f}x")
        if live.get('beta'):
            r2[4].metric('Beta',           f"{live['beta']:.2f}")
        if live.get('div_yield'):
            r2[5].metric('Div yield',      f"{live['div_yield']*100:.2f}%")

        # Row 3: analyst + sector
        info_parts = []
        if live.get('name')   != live['ticker']: info_parts.append(f"**{live['name']}**")
        if live.get('sector'):   info_parts.append(live['sector'])
        if live.get('industry'): info_parts.append(live['industry'])
        if live.get('market_cap_str'): info_parts.append(f"Mkt cap: {live['market_cap_str']}")
        if live.get('target_price'):   info_parts.append(f"Target: {live['target_price']:.2f}")
        if live.get('analyst_rating'): info_parts.append(f"Analyst: {live['analyst_rating'].upper()}")
        if info_parts:
            st.caption('  ·  '.join(info_parts))

        # Real IV smile chart
        smile_near = live['iv_chain'].get('smile_near')
        smile_far  = live['iv_chain'].get('smile_far')
        if smile_near is not None and len(smile_near) > 3:
            st.subheader('📉 Real Implied Volatility Smile (from option chain)')
            fig_smile = go.Figure()
            fig_smile.add_trace(go.Scatter(
                x=smile_near['strike'], y=smile_near['mid_iv']*100,
                mode='lines+markers', name=f"IV {live['iv_chain'].get('exp_near','near')}",
                line=dict(color='#6366f1', width=2), marker=dict(size=5),
            ))
            if smile_far is not None and len(smile_far) > 3:
                fig_smile.add_trace(go.Scatter(
                    x=smile_far['strike'], y=smile_far['mid_iv']*100,
                    mode='lines+markers', name=f"IV {live['iv_chain'].get('exp_far','far')}",
                    line=dict(color='#f59e0b', width=2, dash='dot'), marker=dict(size=5),
                ))
            fig_smile.add_vline(x=live['price'], line_dash='dash', line_color='#ef4444',
                                annotation_text='Spot')
            fig_smile.update_layout(
                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)', height=320,
                font=dict(family='Inter, sans-serif', color='#e2e8f0'),
                xaxis_title='Strike', yaxis_title='IV (%)',
                margin=dict(t=20,b=40,l=52,r=16),
                legend=dict(orientation='h', y=1.05, bgcolor='rgba(0,0,0,0)'),
            )
            fig_smile.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
            fig_smile.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
            st.plotly_chart(fig_smile, width='stretch')

    # -- Live price chart -------------------------------------------------
    if live and 'hist' in live:
        st.divider()
        st.subheader(f"📊 {live['ticker']} — Price History (1Y)")
        hist_df = live['hist'].tail(252)
        fig_live = go.Figure()
        fig_live.add_trace(go.Candlestick(
            x=hist_df.index,
            open=hist_df['Open'], high=hist_df['High'],
            low=hist_df['Low'],   close=hist_df['Close'],
            increasing_line_color='#22c55e', decreasing_line_color='#ef4444',
            name='Price',
        ))
        ma20 = hist_df['Close'].rolling(20).mean()
        fig_live.add_trace(go.Scatter(x=hist_df.index, y=ma20, name='MA 20',
                                       line=dict(color='#f59e0b', width=1.5, dash='dot')))
        fig_live.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', height=380, xaxis_rangeslider_visible=False,
            font=dict(family='Inter, sans-serif', color='#e2e8f0'),
            margin=dict(t=16,b=30,l=52,r=16),
            legend=dict(orientation='h', y=1.04, bgcolor='rgba(0,0,0,0)'),
        )
        fig_live.update_xaxes(gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.12)')
        fig_live.update_yaxes(gridcolor='rgba(255,255,255,0.06)', linecolor='rgba(255,255,255,0.12)')
        st.plotly_chart(fig_live, width='stretch')

        rets_df = live['returns'].tail(252)
        from scipy.stats import norm as _norm_live
        fig_ret = go.Figure()
        fig_ret.add_trace(go.Histogram(x=rets_df*100, nbinsx=80,
            marker_color='#6366f1', opacity=0.8, histnorm='probability density', name='Daily returns'))
        mu_r  = float(rets_df.mean()*100)
        sig_r = float(rets_df.std()*100)
        xs    = np.linspace(mu_r-4*sig_r, mu_r+4*sig_r, 200)
        fig_ret.add_trace(go.Scatter(x=xs, y=_norm_live.pdf(xs, mu_r, sig_r),
            name='Normal', line=dict(color='#f59e0b', width=2)))
        fig_ret.add_vline(x=0, line_color='#94a3b8', line_dash='dot')
        fig_ret.update_layout(
            title=f"Daily Returns | HV1M {live['sigma_1m']*100:.1f}%  HV3M {live['sigma_3m']*100:.1f}%  HV1Y {live['sigma_1y']*100:.1f}%",
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=280, font=dict(family='Inter, sans-serif', color='#e2e8f0'),
            margin=dict(t=40,b=30,l=52,r=16),
            legend=dict(orientation='h', y=1.04, bgcolor='rgba(0,0,0,0)'),
        )
        fig_ret.update_xaxes(gridcolor='rgba(255,255,255,0.06)')
        fig_ret.update_yaxes(gridcolor='rgba(255,255,255,0.06)')
        st.plotly_chart(fig_ret, width='stretch')



# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Greeks Profiles
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.header("Greeks vs Spot / Time / Volatility")

    mode = st.radio("X-axis", ["Spot S", "Time to Expiry T", "Volatility σ"],
                    horizontal=True)

    greek_choice = st.multiselect(
        "Greeks to display",
        ["Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "Vanna", "Volga"],
        default=["Price", "Delta", "Gamma", "Vega", "Theta"],
    )

    N_pts = 300

    if mode == "Spot S":
        x_range = np.linspace(max(S * 0.4, 1), S * 1.6, N_pts)
        x_label = "Spot S"
        x_ref   = S
        ref_label = f"Current S={S}"
    elif mode == "Time to Expiry T":
        x_range = np.linspace(0.005, max(T * 2, 2.0), N_pts)
        x_label = "Time to Expiry (years)"
        x_ref   = T
        ref_label = f"Current T={T}"
    else:
        x_range = np.linspace(0.01, 1.20, N_pts)
        x_label = "Volatility σ"
        x_ref   = sigma
        ref_label = f"Current σ={sigma*100:.1f}%"

    def compute_greek(greek, x, mode):
        if mode == "Spot S":
            s_, k_, t_, r_, sg_ = x, K, T, r, sigma
        elif mode == "Time to Expiry T":
            s_, k_, t_, r_, sg_ = S, K, x, r, sigma
        else:
            s_, k_, t_, r_, sg_ = S, K, T, r, x
        t_ = max(t_, 1e-5)

        pricer = call_price if opt_type == 'call' else put_price
        if greek == "Price":  return pricer(s_, k_, t_, r_, sg_, q)
        if greek == "Delta":  return delta(s_, k_, t_, r_, sg_, q, opt_type)
        if greek == "Gamma":  return gamma(s_, k_, t_, r_, sg_, q)
        if greek == "Vega":   return vega(s_, k_, t_, r_, sg_, q)
        if greek == "Theta":  return theta(s_, k_, t_, r_, sg_, q, opt_type)
        if greek == "Rho":    return rho(s_, k_, t_, r_, sg_, q, opt_type)
        if greek == "Vanna":  return vanna(s_, k_, t_, r_, sg_, q)
        if greek == "Volga":  return volga(s_, k_, t_, r_, sg_, q)
        return 0.0

    colors_g = px.colors.qualitative.Plotly
    ncols = 2
    nrows = max(1, (len(greek_choice) + 1) // ncols)
    fig_g = make_subplots(rows=nrows, cols=ncols,
                          subplot_titles=greek_choice or ["—"])

    for idx, gk in enumerate(greek_choice):
        row = idx // ncols + 1
        col = idx % ncols + 1
        vals = np.array([compute_greek(gk, x, mode) for x in x_range])
        x_disp = x_range * 100 if mode == "Volatility σ" else x_range

        fig_g.add_trace(
            go.Scatter(x=x_disp, y=vals, name=gk,
                       line=dict(color=colors_g[idx % len(colors_g)], width=2)),
            row=row, col=col,
        )
        ref_disp = x_ref * 100 if mode == "Volatility σ" else x_ref
        fig_g.add_vline(x=ref_disp, line_dash="dash", line_color="rgba(255,100,100,0.6)",
                        row=row, col=col)
        fig_g.add_hline(y=0, line_width=0.5, line_color="gray", row=row, col=col)

    fig_g.update_layout(height=280 * nrows, showlegend=False,
                        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), margin=dict(t=40, b=20))
    fig_g.update_xaxes(title_text=x_label if mode != "Volatility σ" else "σ (%)")
    st.plotly_chart(fig_g, width='stretch')

    # ── Theta decay curve ──
    st.subheader("ATM Price vs Time to Expiry (Theta decay)")
    t_decay = np.linspace(0.003, max(T, 1.5), 300)
    pricer  = call_price if opt_type == 'call' else put_price
    p_decay = [pricer(S, K, t, r, sigma, q) for t in t_decay]
    intrinsic_line = [max(S - K, 0) if opt_type == 'call' else max(K - S, 0)] * len(t_decay)

    fig_td = go.Figure()
    fig_td.add_trace(go.Scatter(x=t_decay, y=p_decay, name="Option Price",
                                line=dict(color="#00d4aa", width=2),
                                fill='tozeroy', fillcolor="rgba(0,212,170,0.08)"))
    fig_td.add_trace(go.Scatter(x=t_decay, y=intrinsic_line, name="Intrinsic Value",
                                line=dict(color="tomato", width=1.5, dash="dash")))
    fig_td.add_vline(x=T, line_dash="dot", line_color="orange",
                     annotation_text=f"Current T={T:.2f}", annotation_position="top right")
    fig_td.update_layout(
        xaxis_title="Time to Expiry (years)",
        yaxis_title="Price",
        xaxis_autorange="reversed",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")),
        height=350,
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_td, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Implied Vol
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.header("Implied Volatility Extraction (Brent Method)")

    col_iv1, col_iv2 = st.columns([1, 2])

    with col_iv1:
        st.subheader("Single-strike extraction")
        mkt_price = st.number_input(
            "Market price",
            min_value=0.001, max_value=float(S),
            value=round(float(opt.price), 3), step=0.001, format="%.3f"
        )
        iv_result = implied_vol(mkt_price, S, K, T, r, q, opt_type)
        if np.isnan(iv_result):
            st.error("No solution found — check price vs. arbitrage bounds.")
        else:
            st.success(f"**Implied Vol = {iv_result*100:.4f}%**")
            col_a, col_b = st.columns(2)
            col_a.metric("IV", f"{iv_result*100:.4f}%")
            col_b.metric("Delta at IV", f"{delta(S, K, T, r, iv_result, q, opt_type):.4f}")

        st.divider()
        st.caption(
            f"Arbitrage bounds: "
            f"intrinsic = {opt.intrinsic_value:.4f} | "
            f"max = {S:.4f} (call) / {K*np.exp(-r*T):.4f} (put)"
        )

    with col_iv2:
        st.subheader("Smile — IV across strikes")

        K_lo_pct = st.slider("Strike range: lower (%S)", 60, 95,  75)
        K_hi_pct = st.slider("Strike range: upper (%S)", 105, 150, 130)

        strikes_iv = np.linspace(S * K_lo_pct / 100, S * K_hi_pct / 100, 40)
        log_m      = np.log(strikes_iv / S)

        # Build realistic smile from current sigma as ATM base
        true_vols = realistic_smile(strikes_iv, S, T, atm_vol=sigma,
                                    skew_coef=-0.08, conv_coef=0.10)
        iv_extracted = np.array([
            implied_vol(call_price(S, Ki, T, r, sv, q), S, Ki, T, r, q, 'call')
            for Ki, sv in zip(strikes_iv, true_vols)
        ])

        fig_iv = make_subplots(rows=1, cols=2,
                               subplot_titles=["Smile (strike space)",
                                               "Skew (log-moneyness)"])
        for xdata, xlabel, c in [
            (strikes_iv, "Strike K", 1),
            (log_m,      "log(K/S)", 2),
        ]:
            fig_iv.add_trace(
                go.Scatter(x=xdata, y=iv_extracted * 100,
                           mode="lines+markers", name="IV smile",
                           line=dict(color="#00d4aa", width=2),
                           marker=dict(size=5)),
                row=1, col=c,
            )
        fig_iv.add_vline(x=S,   line_dash="dash", line_color="tomato",
                         annotation_text="Spot", row=1, col=1)
        fig_iv.add_vline(x=0.0, line_dash="dash", line_color="tomato",
                         annotation_text="ATM", row=1, col=2)
        fig_iv.update_yaxes(title_text="IV (%)")
        fig_iv.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=360,
                             showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_iv, width='stretch')

    st.divider()

    # ── IV term structure ──
    st.subheader("ATM IV Term Structure + Forward Vol")

    maturities_ts = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
    atm_vols_ts   = np.array([
        sigma + 0.03 * np.exp(-2.5 * t)
        for t in maturities_ts
    ])
    fwd_vols_ts = [float('nan')]
    for i in range(1, len(maturities_ts)):
        fwd_vols_ts.append(
            forward_vol(atm_vols_ts[i-1], atm_vols_ts[i],
                        maturities_ts[i-1], maturities_ts[i]) * 100
        )

    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=maturities_ts * 12, y=atm_vols_ts * 100,
        name="ATM IV", mode="lines+markers",
        line=dict(color="#4C9BE8", width=2), marker=dict(size=7),
    ))
    fig_ts.add_trace(go.Scatter(
        x=maturities_ts * 12, y=fwd_vols_ts,
        name="Forward vol", mode="lines+markers",
        line=dict(color="#F4A261", width=2, dash="dot"), marker=dict(size=6),
    ))
    fig_ts.add_vline(x=T * 12, line_dash="dash", line_color="tomato",
                     annotation_text=f"T={T:.2f}y")
    fig_ts.update_layout(
        xaxis_title="Maturity (months)", yaxis_title="Vol (%)",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=320,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_ts, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Vol Surface
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.header("Volatility Surface")

    col_sv1, col_sv2 = st.columns([1, 3])

    with col_sv1:
        st.subheader("Surface params")
        skew_c  = st.slider("Skew coefficient",      -0.20, 0.05, -0.08, 0.01)
        conv_c  = st.slider("Convexity coefficient",  0.00, 0.30,  0.10, 0.01)
        K_lo_s  = st.slider("Min strike (%S)", 55, 90, 70)
        K_hi_s  = st.slider("Max strike (%S)", 110, 160, 135)
        n_k     = st.slider("Strike grid points", 10, 50, 27)
        n_t     = st.slider("Maturity grid points",  4, 12,  8)
        T_max   = st.slider("Max maturity (years)",  0.5, 5.0, 2.0, 0.1)

    with col_sv2:
        mats_sv    = np.linspace(1/12, T_max, n_t)
        strikes_sv = np.linspace(S * K_lo_s / 100, S * K_hi_s / 100, n_k)

        iv_sv = np.zeros((len(mats_sv), len(strikes_sv)))
        for i, Ti in enumerate(mats_sv):
            atm_v = sigma + 0.03 * np.exp(-2.5 * Ti)
            iv_sv[i] = realistic_smile(strikes_sv, S, Ti, atm_vol=atm_v,
                                       skew_coef=skew_c, conv_coef=conv_c)

        view = st.radio("View", ["3D Surface", "Heatmap", "Smile by Maturity"],
                        horizontal=True)

        if view == "3D Surface":
            K_g, T_g = np.meshgrid(strikes_sv, mats_sv)
            fig_3d = go.Figure(data=[go.Surface(
                x=K_g, y=T_g, z=iv_sv * 100,
                colorscale="RdYlGn_r",
                colorbar=dict(title="IV (%)"),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=False)
                ),
            )])
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title="Strike K",
                    yaxis_title="Maturity T",
                    zaxis_title="IV (%)",
                    camera=dict(eye=dict(x=1.6, y=-1.8, z=0.9)),
                ),
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")),
                height=580,
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_3d, width='stretch')

        elif view == "Heatmap":
            fig_hm = go.Figure(data=go.Heatmap(
                x=strikes_sv, y=mats_sv * 12, z=iv_sv * 100,
                colorscale="RdYlGn_r",
                colorbar=dict(title="IV (%)"),
            ))
            fig_hm.add_vline(x=S, line_dash="dash", line_color="red",
                             annotation_text=f"Spot={S}")
            fig_hm.update_layout(
                xaxis_title="Strike K",
                yaxis_title="Maturity (months)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=480,
            )
            st.plotly_chart(fig_hm, width='stretch')

        else:  # Smile by Maturity
            fig_sm = go.Figure()
            colors_sm = px.colors.sequential.Viridis
            for i, (Ti, iv_row) in enumerate(zip(mats_sv, iv_sv)):
                col_idx = int(i / len(mats_sv) * (len(colors_sm) - 1))
                lbl = f"T={Ti:.2f}y"
                fig_sm.add_trace(go.Scatter(
                    x=strikes_sv, y=iv_row * 100,
                    name=lbl, mode="lines+markers",
                    line=dict(color=colors_sm[col_idx], width=1.8),
                    marker=dict(size=4),
                ))
            fig_sm.add_vline(x=S, line_dash="dash", line_color="tomato",
                             annotation_text="Spot")
            fig_sm.update_layout(
                xaxis_title="Strike K", yaxis_title="IV (%)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=480,
                legend=dict(title="Maturity", orientation="v"),
            )
            st.plotly_chart(fig_sm, width='stretch')


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Strategy Payoffs
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.header("Option Strategy Payoffs")

    col_st1, col_st2 = st.columns([1, 3])

    with col_st1:
        strategy = st.selectbox("Strategy", [
            "Long Call",
            "Long Put",
            "Short Call",
            "Short Put",
            "Long Straddle",
            "Short Straddle",
            "Long Strangle",
            "Short Strangle",
            "Bull Call Spread",
            "Bear Put Spread",
            "Bull Put Spread (credit)",
            "Bear Call Spread (credit)",
            "Call Butterfly",
            "Iron Condor",
            "Risk Reversal",
            "Covered Call",
            "Protective Put",
            "Collar",
        ])

        S_lo_pct = st.slider("Spot range: lower (%S)", 40, 90, 60)
        S_hi_pct = st.slider("Spot range: upper (%S)", 110, 200, 145)
        show_prex = st.checkbox("Show pre-expiry P&L slices", value=True)

        # Secondary strike inputs
        K2_pct = st.slider("2nd strike K2 (% of K)", 100, 130, 110,
                           help="Used for spreads, condor, strangle")
        K3_pct = st.slider("3rd strike K3 (% of K)", 85, 100, 95,
                           help="Used for strangle put leg, iron condor")

    with col_st2:
        S_range_st = np.linspace(S * S_lo_pct / 100, S * S_hi_pct / 100, 400)
        K2 = K * K2_pct / 100
        K3 = K * K3_pct / 100

        c0  = call_price(S, K,  T, r, sigma, q)
        p0  = put_price(S,  K,  T, r, sigma, q)
        c2  = call_price(S, K2, T, r, sigma, q)
        p2  = put_price(S,  K2, T, r, sigma, q)
        c3  = call_price(S, K3, T, r, sigma, q)
        p3  = put_price(S,  K3, T, r, sigma, q)

        strats = {
            "Long Call":              long_call(S_range_st,  K,  c0),
            "Long Put":               long_put(S_range_st,   K,  p0),
            "Short Call":             -long_call(S_range_st, K,  c0) + 2*c0,
            "Short Put":              -long_put(S_range_st,  K,  p0) + 2*p0,
            "Long Straddle":          long_straddle(S_range_st, K, c0, p0),
            "Short Straddle":         short_straddle(S_range_st, K, c0, p0),
            "Long Strangle":          long_strangle(S_range_st, K2, K3, c2, p3),
            "Short Strangle":         short_strangle(S_range_st, K2, K3, c2, p3),
            "Bull Call Spread":       bull_call_spread(S_range_st, K, K2, c0, c2),
            "Bear Put Spread":        bear_put_spread(S_range_st,  K3, K, p3, p0),
            "Bull Put Spread (credit)": bull_put_spread(S_range_st, K3, K, p3, p0),
            "Bear Call Spread (credit)": bear_call_spread(S_range_st, K, K2, c0, c2),
            "Call Butterfly":         butterfly_calls(S_range_st, K3, K, K2, c3, c0, c2),
            "Iron Condor":            iron_condor(S_range_st, K3, K, K, K2, p3, p0, c0, c2),
            "Risk Reversal":          risk_reversal(S_range_st, K2, K3, c2, p3),
            "Covered Call":           covered_call(S_range_st, S, K2, c2),
            "Protective Put":         protective_put(S_range_st, S, K3, p3),
            "Collar":                 collar(S_range_st, S, K2, K3, c2, p3),
        }

        payoff = strats[strategy]

        fig_st = go.Figure()

        # Pre-expiry slices
        if show_prex and strategy in ("Long Straddle", "Short Straddle"):
            T_snaps = [T, T*0.75, T*0.5, T*0.25, T*0.1]
            colors_t = px.colors.sequential.Blues[2:]
            for ti, col_t in zip(T_snaps, colors_t):
                curves, lbls = preexpiry_pnl_straddle(S_range_st, K, T, [ti], r, sigma, q)
                fig_st.add_trace(go.Scatter(
                    x=S_range_st, y=curves[0], name=lbls[0],
                    line=dict(color=col_t, width=1.2, dash="dot"),
                    opacity=0.7,
                ))
        elif show_prex and strategy in ("Long Call", "Short Call",
                                         "Long Put", "Short Put"):
            pricer_  = call_price if "Call" in strategy else put_price
            sign     = 1 if "Long" in strategy else -1
            prem_    = c0 if "Call" in strategy else p0
            T_snaps  = [T, T*0.75, T*0.5, T*0.25, T*0.1]
            entry_p  = prem_
            colors_t = px.colors.sequential.Blues[2:]
            for ti, col_t in zip(T_snaps, colors_t):
                pnl_ti = np.array([
                    sign * (pricer_(s, K, max(ti, 1e-5), r, sigma, q) - entry_p)
                    for s in S_range_st
                ])
                fig_st.add_trace(go.Scatter(
                    x=S_range_st, y=pnl_ti,
                    name=f"T-t={ti:.2f}y",
                    line=dict(color=col_t, width=1.2, dash="dot"),
                    opacity=0.7,
                ))

        # Profit / loss fill
        fig_st.add_trace(go.Scatter(
            x=S_range_st, y=np.where(payoff >= 0, payoff, 0),
            fill='tozeroy', fillcolor='rgba(0,212,100,0.15)',
            line=dict(width=0), showlegend=False, name="Profit zone",
        ))
        fig_st.add_trace(go.Scatter(
            x=S_range_st, y=np.where(payoff < 0, payoff, 0),
            fill='tozeroy', fillcolor='rgba(255,80,80,0.15)',
            line=dict(width=0), showlegend=False, name="Loss zone",
        ))

        # Main payoff
        fig_st.add_trace(go.Scatter(
            x=S_range_st, y=payoff, name=f"{strategy} (expiry)",
            line=dict(color="#00d4aa", width=2.5),
        ))

        # Reference lines
        fig_st.add_hline(y=0, line_width=0.8, line_color="white")
        fig_st.add_vline(x=S, line_dash="dash", line_color="tomato",
                         annotation_text=f"Spot={S}", annotation_position="top right")
        fig_st.add_vline(x=K, line_dash="dot", line_color="yellow",
                         annotation_text=f"K={K}", annotation_position="top left")

        fig_st.update_layout(
            xaxis_title="Underlying at Expiry",
            yaxis_title="P&L",
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")),
            height=460,
            legend=dict(orientation="h", y=1.06),
        )
        st.plotly_chart(fig_st, width='stretch')

        # ── Analytics box ──
        max_loss   = float(payoff.min())
        max_gain   = float(payoff.max())
        be_count   = np.sum(np.diff(np.sign(payoff)) != 0)

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Max Gain",      f"{max_gain:+.3f}" if max_gain < 1e8 else "Unlimited")
        a2.metric("Max Loss",      f"{max_loss:+.3f}" if max_loss > -1e8 else "Unlimited")
        rr = abs(max_gain / max_loss) if max_loss < 0 else float('inf')
        a3.metric("Reward / Risk", f"{rr:.2f}x" if rr < 1e6 else "inf")
        a4.metric("Break-even pts", str(be_count))

        if strategy == "Long Straddle":
            be_lo, be_hi, prem_total = straddle_breakevens(K, c0, p0)
            st.info(
                f"Straddle cost = **{prem_total:.4f}** | "
                f"Lower B/E = **{be_lo:.2f}** | Upper B/E = **{be_hi:.2f}** | "
                f"Implied move = **+/-{prem_total/S*100:.1f}%**"
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Structured Products
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.header("Structured Products Pricer")
    st.caption("Analytical pricing of common structured notes using Black-Scholes.")

    from scipy.stats import norm as _norm

    # ── Barrier option helpers ────────────────────────────────────────────────
    def barrier_call_dao(S, K, H, T, r, sigma, q=0.0):
        """Down-and-Out Call (H < S, H < K analytical formula)."""
        if H >= S:
            return 0.0
        lam = (r - q + 0.5 * sigma**2) / sigma**2
        x1  = np.log(S / H) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        x2  = np.log(S * H / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)  # noqa
        y1  = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        y2  = np.log(H / S) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        c   = call_price(S, K, T, r, sigma, q)
        c_i = (S * np.exp(-q*T) * (H/S)**(2*lam) * _norm.cdf(y1)
               - K * np.exp(-r*T) * (H/S)**(2*lam-2) * _norm.cdf(y1 - sigma*np.sqrt(T)))
        return max(c - c_i, 0.0)

    def barrier_put_di(S, K, H, T, r, sigma, q=0.0):
        """Down-and-In Put (H < S, H < K)."""
        if H >= S:
            return put_price(S, K, T, r, sigma, q)
        lam  = (r - q + 0.5 * sigma**2) / sigma**2
        y    = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        y1   = np.log(H / S) / (sigma * np.sqrt(T)) + lam * sigma * np.sqrt(T)
        p_di = (-S * np.exp(-q*T) * (H/S)**(2*lam) * _norm.cdf(-y)
                + K * np.exp(-r*T) * (H/S)**(2*lam-2) * _norm.cdf(-y + sigma*np.sqrt(T)))
        return max(p_di, 0.0)

    def digital_call(S, K, T, r, sigma, q=0.0, payout=1.0):
        """Cash-or-nothing call: pays payout if S_T > K."""
        if T <= 0:
            return payout if S > K else 0.0
        from bs_core import d2 as _d2
        return payout * np.exp(-r * T) * _norm.cdf(_d2(S, K, T, r, sigma, q))

    def digital_put(S, K, T, r, sigma, q=0.0, payout=1.0):
        """Cash-or-nothing put: pays payout if S_T < K."""
        if T <= 0:
            return payout if S < K else 0.0
        from bs_core import d2 as _d2
        return payout * np.exp(-r * T) * _norm.cdf(-_d2(S, K, T, r, sigma, q))

    # ── Product selector ──────────────────────────────────────────────────────
    product = st.selectbox("Select Structure", [
        "Capital Protected Note (CPN)",
        "Reverse Convertible",
        "Participation Note (Booster)",
        "Shark Note (Up-and-Out Call)",
        "Down-and-In Put (PDI barrier)",
        "Digital / Binary Option",
        "Range Accrual (approx.)",
        "Bonus Certificate",
        "Standard Autocall (MC)",
        "Phoenix Autocall (MC)",
    ])

    S_range_sp = np.linspace(S * 0.40, S * 1.60, 400)

    # ─────────────────────────────────────────────────────────────────────────
    if product == "Capital Protected Note (CPN)":
        st.markdown("""
        **Structure:** ZCB (zero-coupon bond) + call option on the underlying.
        Investor gets 100% capital back at maturity + upside participation.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            part_rate = st.slider("Participation rate (%)", 50, 200, 100) / 100
            K_cpn = st.slider("Call strike (%S)", 80, 120, 100) / 100 * S
            notional_cpn = st.number_input("Notional", 100.0, 10000.0, 1000.0, 100.0)

            zcb   = notional_cpn * np.exp(-r * T)
            c_val = call_price(S, K_cpn, T, r, sigma, q)
            budget_call = notional_cpn - zcb
            max_part    = budget_call / c_val if c_val > 0 else 0
            issue_price = zcb + part_rate * c_val * (notional_cpn / 100)

            st.metric("ZCB (PV of 100%)",      f"{zcb:.2f}")
            st.metric("Call price (per unit)",  f"{c_val:.4f}")
            st.metric("Issue Price",            f"{issue_price:.2f}")
            st.metric("Max afford. part. rate", f"{max_part*100:.1f}%")
            st.caption(f"Break-even spot at maturity: {K_cpn + (issue_price - zcb)/part_rate/(notional_cpn/100):.2f}")

        with col2:
            pf_cpn = np.maximum(notional_cpn + part_rate * (notional_cpn/100) *
                                np.maximum(S_range_sp - K_cpn, 0), notional_cpn)
            pf_vanilla = notional_cpn + part_rate * (notional_cpn/100) * np.maximum(S_range_sp - K_cpn, 0)

            fig_cpn = go.Figure()
            fig_cpn.add_trace(go.Scatter(x=S_range_sp, y=pf_cpn,
                                          name="CPN payoff", line=dict(color="#00d4aa", width=2.5)))
            fig_cpn.add_trace(go.Scatter(x=S_range_sp, y=[notional_cpn]*len(S_range_sp),
                                          name="Capital protection", line=dict(color="gray", dash="dot")))
            fig_cpn.add_trace(go.Scatter(x=S_range_sp, y=pf_vanilla,
                                          name=f"Vanilla ({part_rate*100:.0f}% part.)",
                                          line=dict(color="steelblue", dash="dash")))
            fig_cpn.add_vline(x=S, line_dash="dash", line_color="tomato",
                               annotation_text=f"Spot={S}")
            fig_cpn.add_vline(x=K_cpn, line_dash="dot", line_color="gold",
                               annotation_text=f"K={K_cpn:.0f}")
            fig_cpn.update_layout(xaxis_title="S at maturity", yaxis_title="Payoff",
                                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=420,
                                   legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig_cpn, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Reverse Convertible":
        st.markdown("""
        **Structure:** Fixed coupon bond + short put. Investor gets enhanced coupon,
        but if S_T < K, receives shares instead of cash (full downside).
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            rc_coupon_pct = st.slider("Annual coupon (%)", 1.0, 30.0, 10.0, 0.5) / 100
            K_rc = st.slider("Conversion strike (%S)", 60, 100, 100) / 100 * S
            notional_rc = 100.0

            coupon_pv = notional_rc * rc_coupon_pct * T * np.exp(-r * T)
            put_val   = put_price(S, K_rc, T, r, sigma, q)
            fair_val  = notional_rc * np.exp(-r * T) + coupon_pv - put_val * (notional_rc / S)

            st.metric("Coupon PV",        f"{coupon_pv:.4f}")
            st.metric("Short put value",  f"{put_val * (notional_rc/S):.4f}")
            st.metric("Fair Value",       f"{fair_val:.4f}")
            be = K_rc - rc_coupon_pct * T * K_rc
            st.metric("Break-even (approx.)", f"{be:.2f}")
            st.caption("Coupon compensates put sale. Risky if vol spikes.")

        with col2:
            pf_rc = np.where(
                S_range_sp >= K_rc,
                notional_rc * (1 + rc_coupon_pct * T),
                notional_rc * S_range_sp / S + notional_rc * rc_coupon_pct * T
            )
            fig_rc = go.Figure()
            fig_rc.add_trace(go.Scatter(x=S_range_sp, y=pf_rc,
                                         name="Reverse Convertible", line=dict(color="#F4A261", width=2.5)))
            fig_rc.add_trace(go.Scatter(x=S_range_sp, y=[notional_rc]*len(S_range_sp),
                                         name="Par 100", line=dict(color="gray", dash="dot")))
            fig_rc.add_vline(x=K_rc, line_dash="dash", line_color="tomato",
                              annotation_text=f"Conversion K={K_rc:.0f}")
            fig_rc.add_vline(x=S, line_dash="dot", line_color="white",
                              annotation_text=f"Spot={S}")
            fig_rc.update_layout(xaxis_title="S at maturity", yaxis_title="Payoff (% notional)",
                                  template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=420,
                                  legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig_rc, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Participation Note (Booster)":
        st.markdown("""
        **Structure:** Long call (ATM) + long call spread (OTM) funded by short put.
        Leveraged upside up to cap, full downside if S falls below put strike.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            boost_part = st.slider("Participation rate (%)", 100, 300, 150) / 100
            K_cap      = st.slider("Cap level (%S)", 110, 200, 130) / 100 * S
            K_put_b    = st.slider("Put strike (%S)", 60, 100, 90) / 100 * S

            c_atm_b = call_price(S, S, T, r, sigma, q)
            c_cap_b = call_price(S, K_cap, T, r, sigma, q)
            p_put_b = put_price(S, K_put_b, T, r, sigma, q)
            structure_cost = boost_part * (c_atm_b - c_cap_b) - p_put_b

            st.metric("ATM Call",     f"{c_atm_b:.4f}")
            st.metric("OTM Call cap", f"{c_cap_b:.4f}")
            st.metric("Short Put",    f"{p_put_b:.4f}")
            st.metric("Net Cost",     f"{structure_cost:.4f}",
                      delta="credit" if structure_cost < 0 else "debit")

        with col2:
            pf_boost = np.where(
                S_range_sp < K_put_b,
                S_range_sp - S,
                np.where(
                    S_range_sp <= K_cap,
                    boost_part * (S_range_sp - S),
                    boost_part * (K_cap - S)
                )
            ) - structure_cost

            fig_boost = go.Figure()
            fig_boost.add_trace(go.Scatter(x=S_range_sp, y=pf_boost,
                                            name="Booster Note", line=dict(color="#9C27B0", width=2.5)))
            fig_boost.add_vline(x=S,       line_dash="dot",  line_color="white",  annotation_text="Spot")
            fig_boost.add_vline(x=K_cap,   line_dash="dash", line_color="gold",   annotation_text=f"Cap={K_cap:.0f}")
            fig_boost.add_vline(x=K_put_b, line_dash="dash", line_color="tomato", annotation_text=f"Put={K_put_b:.0f}")
            fig_boost.add_hline(y=0, line_color="gray", line_width=0.8)
            fig_boost.update_layout(xaxis_title="S at maturity", yaxis_title="P&L",
                                     template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=420)
            st.plotly_chart(fig_boost, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Shark Note (Up-and-Out Call)":
        st.markdown("""
        **Structure:** Up-and-out call (knock-out if S hits upper barrier).
        Cheaper than vanilla call, but profit capped if market rallies too hard.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            H_uo   = st.slider("Knock-out barrier (%S)", 105, 160, 130) / 100 * S
            K_uo   = st.slider("Strike (%S)", 80, 110, 100) / 100 * S
            rebate = st.slider("Rebate (if knocked out)", 0.0, 5.0, 0.0, 0.1)

            c_vanilla = call_price(S, K_uo, T, r, sigma, q)
            # Up-and-out = vanilla - up-and-in. Use put-call-symmetry approximation
            lam    = (r - q + 0.5*sigma**2) / sigma**2
            x1_uo  = np.log(S/H_uo)/(sigma*np.sqrt(T)) + lam*sigma*np.sqrt(T)
            y1_uo  = np.log(H_uo**2/(S*K_uo))/(sigma*np.sqrt(T)) + lam*sigma*np.sqrt(T)
            c_ui   = max(
                S*np.exp(-q*T)*(H_uo/S)**(2*lam)*_norm.cdf(y1_uo)
                - K_uo*np.exp(-r*T)*(H_uo/S)**(2*lam-2)*_norm.cdf(y1_uo - sigma*np.sqrt(T)), 0
            )
            c_uo   = max(c_vanilla - c_ui + rebate * np.exp(-r*T), 0)
            discount_pct = (1 - c_uo/c_vanilla)*100 if c_vanilla > 0 else 0

            st.metric("Vanilla call",      f"{c_vanilla:.4f}")
            st.metric("Up-and-Out call",   f"{c_uo:.4f}")
            st.metric("Discount vs vanilla", f"{discount_pct:.1f}%")
            st.caption(f"Knock-out at S = {H_uo:.1f} ({H_uo/S*100:.0f}% of spot)")

        with col2:
            # Payoff at expiry: 0 if path hit H_uo (shown as dotted), else call payoff
            pf_uo_alive  = np.maximum(S_range_sp - K_uo, 0) - c_uo
            pf_uo_ko     = np.where(S_range_sp >= H_uo, rebate - c_uo, pf_uo_alive)
            pf_van_line  = np.maximum(S_range_sp - K_uo, 0) - c_vanilla

            fig_uo = go.Figure()
            fig_uo.add_trace(go.Scatter(x=S_range_sp, y=pf_uo_ko,
                                         name="Up-and-Out Call P&L", line=dict(color="#FF9800", width=2.5)))
            fig_uo.add_trace(go.Scatter(x=S_range_sp, y=pf_van_line,
                                         name="Vanilla Call P&L", line=dict(color="steelblue", dash="dash")))
            fig_uo.add_vline(x=H_uo, line_dash="dash", line_color="red",
                              annotation_text=f"KO barrier={H_uo:.0f}")
            fig_uo.add_vline(x=K_uo, line_dash="dot",  line_color="gold",
                              annotation_text=f"K={K_uo:.0f}")
            fig_uo.add_hline(y=0, line_color="gray", line_width=0.7)
            fig_uo.update_layout(xaxis_title="S at maturity", yaxis_title="P&L",
                                  template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=420,
                                  legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig_uo, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Down-and-In Put (PDI barrier)":
        st.markdown("""
        **Structure:** Put that only activates if underlying crosses the barrier.
        Core building block of Autocall PDI. Cheaper than vanilla put.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            H_di   = st.slider("PDI barrier (%S)", 40, 95, 60) / 100 * S
            K_di   = st.slider("Strike (%S)", 80, 120, 100) / 100 * S

            p_vanilla = put_price(S, K_di, T, r, sigma, q)
            p_di_val  = barrier_put_di(S, K_di, H_di, T, r, sigma, q)
            p_do_val  = p_vanilla - p_di_val
            discount  = (1 - p_di_val/p_vanilla)*100 if p_vanilla > 0 else 0

            st.metric("Vanilla Put",      f"{p_vanilla:.4f}")
            st.metric("Down-and-In Put",  f"{p_di_val:.4f}")
            st.metric("Down-and-Out Put", f"{p_do_val:.4f}")
            st.metric("DI discount",      f"{discount:.1f}%")
            st.caption(f"PDI barrier: {H_di:.1f} ({H_di/S*100:.0f}% of spot)")
            st.caption("DI Put = Vanilla Put (if barrier breached during life)")

        with col2:
            # Payoff at expiry given that barrier was hit (approximation: show unconditional)
            pf_di    = np.maximum(K_di - S_range_sp, 0) - p_di_val
            pf_van_p = np.maximum(K_di - S_range_sp, 0) - p_vanilla

            fig_di = go.Figure()
            fig_di.add_trace(go.Scatter(x=S_range_sp, y=pf_van_p,
                                         name="Vanilla Put P&L", line=dict(color="steelblue", dash="dash")))
            fig_di.add_trace(go.Scatter(x=S_range_sp, y=pf_di,
                                         name="Down-and-In Put P&L", line=dict(color="#F44336", width=2.5)))
            fig_di.add_vrect(x0=S_range_sp[0], x1=H_di,
                              fillcolor="rgba(255,80,80,0.08)", line_width=0,
                              annotation_text="Barrier zone")
            fig_di.add_vline(x=H_di, line_dash="dash", line_color="red",
                              annotation_text=f"PDI={H_di:.0f}")
            fig_di.add_vline(x=K_di, line_dash="dot",  line_color="gold",
                              annotation_text=f"K={K_di:.0f}")
            fig_di.add_hline(y=0, line_color="gray", line_width=0.7)
            fig_di.update_layout(xaxis_title="S at maturity", yaxis_title="P&L",
                                  template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=420,
                                  legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig_di, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Digital / Binary Option":
        st.markdown("""
        **Structure:** Pays a fixed cash amount if S_T > K (call) or S_T < K (put).
        Used in structured products as conditional coupon triggers.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            K_dig    = st.slider("Strike K (%S)", 70, 130, 100) / 100 * S
            payout_d = st.number_input("Payout amount", 0.1, 100.0, 1.0, 0.5)
            dig_type = st.radio("Type", ["Call", "Put"], horizontal=True)

            d_call = digital_call(S, K_dig, T, r, sigma, q, payout_d)
            d_put  = digital_put(S, K_dig, T, r, sigma, q, payout_d)
            van_c  = call_price(S, K_dig, T, r, sigma, q)
            van_p  = put_price(S, K_dig, T, r, sigma, q)

            st.metric("Digital Call",  f"{d_call:.4f}")
            st.metric("Digital Put",   f"{d_put:.4f}")
            st.metric("Digital C+P",   f"{d_call+d_put:.4f}",
                      delta=f"vs e^(-rT)={np.exp(-r*T)*payout_d:.4f}")
            st.caption(f"Vs vanilla call: {van_c:.4f}  |  put: {van_p:.4f}")

        with col2:
            pf_dcall = np.where(S_range_sp > K_dig, payout_d, 0) - d_call
            pf_dput  = np.where(S_range_sp < K_dig, payout_d, 0) - d_put

            fig_dig = go.Figure()
            fig_dig.add_trace(go.Scatter(x=S_range_sp, y=pf_dcall,
                                          name="Digital Call P&L", line=dict(color="#4CAF50", width=2.5)))
            fig_dig.add_trace(go.Scatter(x=S_range_sp, y=pf_dput,
                                          name="Digital Put P&L",  line=dict(color="#F44336", width=2.5)))
            fig_dig.add_vline(x=K_dig, line_dash="dash", line_color="gold",
                               annotation_text=f"K={K_dig:.0f}")
            fig_dig.add_vline(x=S, line_dash="dot", line_color="white",
                               annotation_text=f"Spot={S}")
            fig_dig.add_hline(y=0, line_color="gray", line_width=0.7)
            fig_dig.update_layout(xaxis_title="S at maturity", yaxis_title="P&L",
                                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=420,
                                   legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig_dig, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Range Accrual (approx.)":
        st.markdown("""
        **Structure:** Pays coupon proportional to the number of days S stays
        within [L, U]. Approximated here as a digital call spread × days.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            L_ra = st.slider("Lower bound (%S)", 60, 98, 80)  / 100 * S
            U_ra = st.slider("Upper bound (%S)", 102, 150, 120) / 100 * S
            coupon_ra = st.slider("Annual coupon if in range (%)", 1.0, 20.0, 8.0, 0.5) / 100
            n_days_ra = int(T * 252)

            # Approx: for each future date, probability of being in range × daily coupon
            daily_cpn = coupon_ra / 252
            total_val = 0.0
            for i in range(1, n_days_ra + 1):
                ti = i / 252
                p_above_L = digital_call(S, L_ra, ti, r, sigma, q, 1.0) / np.exp(-r*ti)
                p_below_U = digital_put(S, U_ra, ti, r, sigma, q, 1.0) / np.exp(-r*ti)
                p_in = max(p_above_L + p_below_U - 1, 0)
                total_val += p_in * daily_cpn * np.exp(-r * ti)

            st.metric("Range Accrual PV",  f"{total_val:.4f}")
            st.metric("Max coupon (100%)",  f"{coupon_ra * T:.4f}")
            st.metric("Avg accrual ratio",  f"{total_val/(coupon_ra*T)*100:.1f}%",
                      help="Expected % of days in range")
            st.caption(f"Range: [{L_ra:.1f}, {U_ra:.1f}]  |  {n_days_ra} observation days")

        with col2:
            # Show range on spot distribution
            sig_range = np.linspace(S * 0.3, S * 2.0, 400)
            from scipy.stats import lognorm as _lg
            mu_ln  = np.log(S) + (r - q - 0.5*sigma**2) * T
            sig_ln = sigma * np.sqrt(T)
            pdf_T  = _lg.pdf(sig_range, s=sig_ln, scale=np.exp(mu_ln))

            fig_ra = go.Figure()
            fig_ra.add_trace(go.Scatter(x=sig_range, y=pdf_T,
                                         name="S_T distribution", fill='tozeroy',
                                         fillcolor="rgba(76,155,232,0.15)",
                                         line=dict(color="#4C9BE8", width=2)))
            fig_ra.add_vrect(x0=L_ra, x1=U_ra,
                              fillcolor="rgba(0,212,100,0.15)", line_width=0,
                              annotation_text="Accrual range", annotation_position="top left")
            fig_ra.add_vline(x=S,    line_dash="dot", line_color="white",
                              annotation_text=f"Spot={S}")
            fig_ra.add_vline(x=L_ra, line_dash="dash", line_color="tomato",
                              annotation_text=f"L={L_ra:.0f}")
            fig_ra.add_vline(x=U_ra, line_dash="dash", line_color="gold",
                              annotation_text=f"U={U_ra:.0f}")
            fig_ra.update_layout(
                xaxis_title="S at maturity", yaxis_title="Probability density",
                title=f"Risk-neutral distribution of S_T (T={T:.1f}y)",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=420,
            )
            st.plotly_chart(fig_ra, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Bonus Certificate":
        st.markdown("""
        **Structure:** Long underlying + Long Down-and-In Put.
        Delivers bonus level if barrier never breached; full upside otherwise.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            H_bon  = st.slider("PDI barrier (%S)", 40, 90, 65) / 100 * S
            bonus  = st.slider("Bonus level (%S)", 110, 160, 120) / 100 * S

            p_di_b  = barrier_put_di(S, bonus, H_bon, T, r, sigma, q)
            cost_b  = S * np.exp(-q*T) + p_di_b - S    # cost vs just holding stock
            fair_b  = S + p_di_b

            st.metric("Underlying (PV)",   f"{S*np.exp(-q*T):.2f}")
            st.metric("DI Put value",      f"{p_di_b:.4f}")
            st.metric("Bonus Certificate", f"{fair_b:.2f}")
            st.metric("Premium vs stock",  f"{cost_b:+.4f}")
            st.caption(f"Bonus at {bonus:.0f} if barrier {H_bon:.0f} never touched")

        with col2:
            pf_bonus = np.where(
                S_range_sp >= bonus, S_range_sp,
                np.where(S_range_sp >= H_bon, bonus, S_range_sp)
            ) - fair_b
            pf_stock = S_range_sp - S

            fig_bon = go.Figure()
            fig_bon.add_trace(go.Scatter(x=S_range_sp, y=pf_stock,
                                          name="Long stock P&L", line=dict(color="gray", dash="dash")))
            fig_bon.add_trace(go.Scatter(x=S_range_sp, y=pf_bonus,
                                          name="Bonus Certificate P&L", line=dict(color="#00BCD4", width=2.5)))
            fig_bon.add_vrect(x0=S_range_sp[0], x1=H_bon,
                               fillcolor="rgba(255,80,80,0.08)", line_width=0)
            fig_bon.add_vline(x=H_bon, line_dash="dash", line_color="tomato",
                               annotation_text=f"Barrier={H_bon:.0f}")
            fig_bon.add_vline(x=bonus, line_dash="dot", line_color="gold",
                               annotation_text=f"Bonus={bonus:.0f}")
            fig_bon.add_hline(y=0, line_color="gray", line_width=0.7)
            fig_bon.update_layout(xaxis_title="S at maturity", yaxis_title="P&L",
                                   template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=420,
                                   legend=dict(orientation="h", y=1.06))
            st.plotly_chart(fig_bon, width='stretch')

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Standard Autocall (MC)":
        st.markdown("""
        **Structure:** Annual observation autocall.
        Recalled with coupon if S(ti) >= B_ac × S0; otherwise PDI risk at maturity.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            ac_T      = st.slider("Maturity (years)",        1.0, 5.0,  3.0, 0.5, key="ac_T")
            ac_cpn    = st.slider("Annual coupon (%)",       1.0, 25.0, 8.0, 0.5, key="ac_cpn") / 100
            ac_bac    = st.slider("Recall barrier (%S0)",   80, 120, 100, key="ac_bac")   / 100
            ac_bpdi   = st.slider("PDI barrier (%S0)",      40,  90,  60, key="ac_bpdi")  / 100
            ac_paths  = st.select_slider("MC paths", [10_000, 25_000, 50_000], 25_000, key="ac_paths")
            run_ac    = st.button("Price Autocall", type="primary", key="btn_ac")

        with col2:
            if run_ac:
                with st.spinner("Running Monte Carlo..."):
                    @st.cache_data
                    def _price_autocall(s0, r_, sig_, q_, T_, cpn_, bac_, bpdi_, n_, seed_=42):
                        rng   = np.random.default_rng(seed_)
                        steps = int(252 * T_)
                        dt_   = T_ / steps
                        half  = n_ // 2
                        Z_h   = rng.standard_normal((half, steps))
                        Z_    = np.concatenate([Z_h, -Z_h], axis=0)
                        lp    = np.cumsum((r_ - q_ - 0.5*sig_**2)*dt_ + sig_*np.sqrt(dt_)*Z_, axis=1)
                        paths_ = s0 * np.exp(np.concatenate([np.zeros((n_, 1)), lp], axis=1))
                        obs_ts = np.arange(1, int(round(T_))+1, dtype=float)[:int(round(T_/1.0))]
                        payoffs_ = np.full(n_, np.nan)
                        called_  = np.zeros(n_, dtype=bool)
                        call_date_ = np.full(n_, np.nan)
                        call_prob_ = {}
                        for ot in obs_ts:
                            idx_ = min(round(ot/dt_), steps)
                            trig = ~called_ & (paths_[:, idx_] >= bac_ * s0)
                            disc_ = np.exp(-r_ * ot)
                            payoffs_ = np.where(trig & np.isnan(payoffs_),
                                                disc_ * s0 * (1 + cpn_ * ot), payoffs_)
                            call_date_ = np.where(trig & ~called_, ot, call_date_)
                            called_ = called_ | trig
                            call_prob_[ot] = float(trig.mean())
                        pmin_ = paths_[:, 1:].min(axis=1)
                        pdi_  = pmin_ < bpdi_ * s0
                        sT_   = paths_[:, -1]
                        disc_T_ = np.exp(-r_ * T_)
                        mat_pf_ = np.where(pdi_, s0 * sT_ / s0, s0 * 1.0)
                        payoffs_ = np.where(np.isnan(payoffs_), disc_T_ * mat_pf_, payoffs_)
                        price_   = float(payoffs_.mean())
                        se_      = float(payoffs_.std(ddof=1) / np.sqrt(n_))
                        return (price_, se_, float(called_.mean()), float(pdi_.mean()),
                                float((~called_&~pdi_).mean()), float(np.nanmean(call_date_)),
                                call_prob_, payoffs_)

                    res = _price_autocall(S, r, sigma, q, ac_T, ac_cpn,
                                          ac_bac, ac_bpdi, ac_paths)
                    pr_, se_, pc_, ppdi_, pok_, acd_, cpdict_, pfs_ = res

                m1, m2, m3 = st.columns(3)
                m1.metric("Fair Value",    f"{pr_:.4f}", f"{pr_-S:+.2f} vs S0")
                m2.metric("Std Error",     f"{se_:.4f}")
                m3.metric("95% CI",        f"[{pr_-1.96*se_:.3f}, {pr_+1.96*se_:.3f}]")
                p1, p2, p3 = st.columns(3)
                p1.metric("P(Called)",     f"{pc_*100:.1f}%")
                p2.metric("P(PDI)",        f"{ppdi_*100:.1f}%")
                p3.metric("P(Mat. OK)",    f"{pok_*100:.1f}%")
                if not np.isnan(acd_):
                    st.caption(f"Average call date: **{acd_:.2f}y**")

                fig_ac = go.Figure(go.Bar(
                    x=[f"T={t:.0f}y" for t in cpdict_],
                    y=[v*100 for v in cpdict_.values()],
                    marker_color="#4CAF50",
                ))
                fig_ac.update_layout(title="P(called) by observation date",
                                     xaxis_title="Date", yaxis_title="Probability (%)",
                                     template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=280, margin=dict(t=35))
                st.plotly_chart(fig_ac, width='stretch')

                fig_hist_ac = go.Figure()
                fig_hist_ac.add_trace(go.Histogram(x=pfs_, nbinsx=60,
                                                    marker_color="#4C9BE8", opacity=0.75,
                                                    histnorm="probability density", name="Payoffs"))
                fig_hist_ac.add_vline(x=pr_, line_dash="dash", line_color="white",
                                      annotation_text=f"FV={pr_:.2f}")
                fig_hist_ac.add_vline(x=S, line_dash="dot", line_color="gold",
                                      annotation_text="Par")
                fig_hist_ac.update_layout(xaxis_title="Discounted Payoff", yaxis_title="Density",
                                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=260, margin=dict(t=10))
                st.plotly_chart(fig_hist_ac, width='stretch')
            else:
                st.info("Set parameters and click **Price Autocall**.")

    # ─────────────────────────────────────────────────────────────────────────
    elif product == "Phoenix Autocall (MC)":
        st.markdown("""
        **Structure:** Quarterly observation Phoenix with coupon barrier and optional memory.
        Coupon paid on each date S(ti) >= B_cpn × S0, even if not recalled.
        """)
        col1, col2 = st.columns([1, 2])
        with col1:
            ph_T     = st.slider("Maturity (years)",         1.0, 5.0,  3.0, 0.5, key="ph_T")
            ph_cpn   = st.slider("Annual coupon (%)",        1.0, 30.0, 10.0, 0.5, key="ph_cpn") / 100
            ph_bac   = st.slider("Recall barrier (%S0)",    80, 120, 100, key="ph_bac")   / 100
            ph_bpdi  = st.slider("PDI barrier (%S0)",       40,  90,  60, key="ph_bpdi")  / 100
            ph_bcpn  = st.slider("Coupon barrier (%S0)",    50, 100,  70, key="ph_bcpn")  / 100
            ph_mem   = st.checkbox("Memory coupon", value=True, key="ph_mem")
            ph_paths = st.select_slider("MC paths", [10_000, 25_000, 50_000], 25_000, key="ph_paths")
            run_ph   = st.button("Price Phoenix", type="primary", key="btn_ph")

        with col2:
            if run_ph:
                with st.spinner("Running Monte Carlo..."):
                    @st.cache_data
                    def _price_phoenix(s0, r_, sig_, q_, T_, cpn_, bac_, bpdi_, bcpn_, mem_, n_, seed_=42):
                        rng   = np.random.default_rng(seed_)
                        steps = int(252 * T_)
                        dt_   = T_ / steps
                        half  = n_ // 2
                        Z_h   = rng.standard_normal((half, steps))
                        Z_    = np.concatenate([Z_h, -Z_h], axis=0)
                        lp    = np.cumsum((r_ - q_ - 0.5*sig_**2)*dt_ + sig_*np.sqrt(dt_)*Z_, axis=1)
                        paths_ = s0 * np.exp(np.concatenate([np.zeros((n_, 1)), lp], axis=1))
                        n_obs  = int(round(T_ / 0.25))
                        obs_ts = np.array([0.25*(i+1) for i in range(n_obs)])
                        payoffs_ = np.full(n_, np.nan)
                        called_  = np.zeros(n_, dtype=bool)
                        call_date_ = np.full(n_, np.nan)
                        pending  = np.zeros(n_)
                        cpn_per_obs = s0 * cpn_ * 0.25
                        call_prob_ = {}
                        for ot in obs_ts:
                            idx_ = min(round(ot/dt_), steps)
                            Sobs = paths_[:, idx_]
                            active = ~called_
                            cpn_elig = active & (Sobs >= bcpn_ * s0)
                            if mem_:
                                pending += np.where(active, cpn_per_obs, 0)
                            trig = active & (Sobs >= bac_ * s0)
                            recall_amt = s0 + pending if mem_ else s0 * (1 + cpn_ * ot)
                            disc_ = np.exp(-r_ * ot)
                            payoffs_ = np.where(trig & np.isnan(payoffs_),
                                                disc_ * recall_amt, payoffs_)
                            if not mem_:
                                cpn_pay = np.where(cpn_elig & ~trig & np.isnan(payoffs_),
                                                   disc_ * cpn_per_obs, 0)
                                payoffs_ = np.where(cpn_elig & ~trig,
                                                    np.where(np.isnan(payoffs_),
                                                             cpn_pay,
                                                             payoffs_ + disc_ * cpn_per_obs),
                                                    payoffs_)
                            call_date_ = np.where(trig & ~called_, ot, call_date_)
                            called_ = called_ | trig
                            call_prob_[ot] = float(trig.mean())
                        pmin_ = paths_[:, 1:].min(axis=1)
                        pdi_  = pmin_ < bpdi_ * s0
                        sT_   = paths_[:, -1]
                        disc_T_ = np.exp(-r_ * T_)
                        mat_base = np.where(pdi_, s0 * sT_ / s0, s0)
                        mat_pf_  = mat_base + (pending if mem_ else 0)
                        payoffs_ = np.where(np.isnan(payoffs_), disc_T_ * mat_pf_, payoffs_)
                        price_   = float(payoffs_.mean())
                        se_      = float(payoffs_.std(ddof=1) / np.sqrt(n_))
                        return (price_, se_, float(called_.mean()), float(pdi_.mean()),
                                float((~called_&~pdi_).mean()), float(np.nanmean(call_date_)),
                                call_prob_, payoffs_)

                    res_ph = _price_phoenix(S, r, sigma, q, ph_T, ph_cpn,
                                            ph_bac, ph_bpdi, ph_bcpn, ph_mem, ph_paths)
                    pr_p, se_p, pc_p, ppdi_p, pok_p, acd_p, cpd_p, pfs_p = res_ph

                m1, m2, m3 = st.columns(3)
                m1.metric("Fair Value",  f"{pr_p:.4f}", f"{pr_p-S:+.2f} vs S0")
                m2.metric("Std Error",   f"{se_p:.4f}")
                m3.metric("95% CI",      f"[{pr_p-1.96*se_p:.3f}, {pr_p+1.96*se_p:.3f}]")
                p1, p2, p3 = st.columns(3)
                p1.metric("P(Called)",   f"{pc_p*100:.1f}%")
                p2.metric("P(PDI)",      f"{ppdi_p*100:.1f}%")
                p3.metric("P(Mat. OK)",  f"{pok_p*100:.1f}%")
                if not np.isnan(acd_p):
                    st.caption(f"Average call date: **{acd_p:.2f}y**")

                # Quarterly call profile
                obs_labels = [f"T={t:.2f}y" for t in cpd_p]
                fig_ph = go.Figure(go.Bar(
                    x=obs_labels, y=[v*100 for v in cpd_p.values()],
                    marker_color="#E91E63",
                ))
                fig_ph.update_layout(title="P(called) by quarterly observation",
                                     xaxis_title="Date", yaxis_title="Probability (%)",
                                     template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=300, margin=dict(t=35))
                st.plotly_chart(fig_ph, width='stretch')

                fig_hist_ph = go.Figure()
                fig_hist_ph.add_trace(go.Histogram(x=pfs_p, nbinsx=60,
                                                    marker_color="#E91E63", opacity=0.75,
                                                    histnorm="probability density", name="Payoffs"))
                fig_hist_ph.add_vline(x=pr_p, line_dash="dash", line_color="white",
                                      annotation_text=f"FV={pr_p:.2f}")
                fig_hist_ph.add_vline(x=S, line_dash="dot", line_color="gold",
                                      annotation_text="Par")
                fig_hist_ph.update_layout(xaxis_title="Discounted Payoff", yaxis_title="Density",
                                          template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", color="#e2e8f0"), hoverlabel=dict(bgcolor="#1e293b", font=dict(color="#e2e8f0")), height=260, margin=dict(t=10))
                st.plotly_chart(fig_hist_ph, width='stretch')
            else:
                st.info("Set parameters and click **Price Phoenix**.")

    # ── Sensitivity table (price vs spot) ─────────────────────────────────────
    st.divider()
    st.subheader("Quick Greeks — this structure")

    g1, g2, g3, g4 = st.columns(4)
    dS_bump = S * 0.01
    dv_bump = 0.01

    def struct_price_at(s_, sig_):
        if product == "Capital Protected Note (CPN)":
            return (100 * np.exp(-r*T)
                    + part_rate * call_price(s_, K_cpn, T, r, sig_, q) * (100/100))
        elif product == "Reverse Convertible":
            return (100 * np.exp(-r*T) + 100*rc_coupon_pct*T*np.exp(-r*T)
                    - put_price(s_, K_rc, T, r, sig_, q) * (100/s_))
        elif product == "Down-and-In Put (PDI barrier)":
            return barrier_put_di(s_, K_di, H_di, T, r, sig_, q)
        elif product == "Digital / Binary Option":
            return digital_call(s_, K_dig, T, r, sig_, q, payout_d)
        elif product == "Shark Note (Up-and-Out Call)":
            return call_price(s_, K_uo, T, r, sig_, q) - max(
                s_*np.exp(-q*T)*(H_uo/s_)**(2*lam)*_norm.cdf(np.log(H_uo**2/(s_*K_uo))/(sig_*np.sqrt(T))+(r-q+0.5*sig_**2)/sig_**2*sig_*np.sqrt(T))
                - K_uo*np.exp(-r*T)*(H_uo/s_)**(2*lam-2)*_norm.cdf(np.log(H_uo**2/(s_*K_uo))/(sig_*np.sqrt(T))+(r-q-0.5*sig_**2)/sig_**2*sig_*np.sqrt(T)), 0)
        else:
            return call_price(s_, K, T, r, sig_, q)

    try:
        v0s  = struct_price_at(S, sigma)
        v_su = struct_price_at(S + dS_bump, sigma)
        v_sd = struct_price_at(S - dS_bump, sigma)
        v_vu = struct_price_at(S, sigma + dv_bump)
        v_vd = struct_price_at(S, sigma - dv_bump)
        s_delta = (v_su - v_sd) / (2 * dS_bump)
        s_gamma = (v_su - 2*v0s + v_sd) / dS_bump**2
        s_vega  = (v_vu - v_vd) / (2 * dv_bump)
        g1.metric("Price",  f"{v0s:.4f}")
        g2.metric("Delta",  f"{s_delta:.4f}")
        g3.metric("Gamma",  f"{s_gamma:.6f}")
        g4.metric("Vega",   f"{s_vega:.4f}")
    except Exception:
        st.caption("Greeks not available for this structure in current config.")


# ── Footer ────────────────────────────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Delta Hedging
# ══════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.header("📐 Dynamic Delta Hedging Simulation")
    st.caption(
        "Short 1 option, hedge with the underlying. "
        "P&L = premium compounded - hedge cost - payoff. "
        "In BS world: zero in continuous time, non-zero with discrete rebalancing."
    )

    from delta_hedge import simulate_hedge, rebal_frequency_analysis

    dh1, dh2 = st.columns([1, 2])

    with dh1:
        st.subheader("Parameters")
        dh_K      = st.number_input("Strike K",       1.0, 10000.0, float(K),     1.0,  key="dh_K")
        dh_T      = st.slider("Maturity T (years)",   0.1,  5.0,    float(T),     0.05, key="dh_T")
        dh_r      = st.slider("Rate r (%)",            0.0, 15.0,   float(r*100), 0.1,  key="dh_r") / 100
        dh_sigma  = st.slider("Volatility σ (%)",      1.0,100.0,  float(sigma*100), 0.5, key="dh_sig") / 100
        dh_q      = st.slider("Dividend q (%)",        0.0, 10.0,   float(q*100), 0.1,  key="dh_q")  / 100
        dh_opt    = st.radio("Option type", ["call", "put"], horizontal=True, key="dh_opt")
        dh_steps  = st.select_slider("Steps/year (n)", [52, 126, 252, 504], 252, key="dh_steps")
        dh_paths  = st.select_slider("Paths",     [500, 1_000, 2_000, 5_000], 2_000, key="dh_paths")
        dh_rebal  = st.select_slider(
            "Rebalance every N steps",
            options=[1, 2, 5, 10, 21, 63],
            value=1,
            format_func=lambda x: {1:"Daily",2:"2 days",5:"Weekly",
                                    10:"2 weeks",21:"Monthly",63:"Quarterly"}[x],
            key="dh_rebal",
        )
        run_dh = st.button("Run Hedging Simulation", type="primary", key="btn_dh")

    with dh2:
        if run_dh:
            total_steps_dh = round(dh_steps * dh_T)

            with st.spinner("Simulating delta hedging..."):
                @st.cache_data
                def _run_dh(s0_, K_, T_, r_, sig_, q_, opt_, n_p_, n_s_, rb_, seed_=42):
                    return simulate_hedge(s0_, K_, T_, r_, sig_, q_, opt_,
                                          n_paths=n_p_, n_steps=n_s_,
                                          rebal_every=rb_, antithetic=True, seed=seed_)
                res_dh = _run_dh(S, dh_K, dh_T, dh_r, dh_sigma, dh_q,
                                  dh_opt, dh_paths, total_steps_dh, dh_rebal)

            # KPIs
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Option price (V0)",  f"{res_dh['V0']:.4f}")
            m2.metric("Mean P&L",           f"{res_dh['mean_pnl']:+.4f}",
                      help="Should be ~0 in BS world")
            m3.metric("Std P&L",            f"{res_dh['std_pnl']:.4f}",
                      help="Hedging error — decays as 1/sqrt(N rebal)")
            m4.metric("RMSE",               f"{res_dh['rmse']:.4f}")

            m5, m6 = st.columns(2)
            m5.metric("5th pctile P&L",  f"{res_dh['pct5']:+.4f}")
            m6.metric("95th pctile P&L", f"{res_dh['pct95']:+.4f}")

            # P&L distribution
            fig_dh_hist = go.Figure()
            fig_dh_hist.add_trace(go.Histogram(
                x=res_dh["pnl"], nbinsx=80,
                marker_color="#6366f1", opacity=0.8,
                histnorm="probability density", name="Hedge P&L",
            ))
            fig_dh_hist.add_vline(x=0,                    line_dash="dot",  line_color="white")
            fig_dh_hist.add_vline(x=res_dh["mean_pnl"],   line_dash="dash", line_color="#f59e0b",
                                   annotation_text="Mean")
            fig_dh_hist.add_vline(x=res_dh["pct5"],       line_dash="dot",  line_color="#ef4444",
                                   annotation_text="5%")
            fig_dh_hist.add_vline(x=res_dh["pct95"],      line_dash="dot",  line_color="#22c55e",
                                   annotation_text="95%")
            fig_dh_hist.update_layout(
                title=f"Hedging P&L Distribution  (rebal every {dh_rebal} step(s))",
                xaxis_title="Terminal P&L", yaxis_title="Density",
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=340,
                font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                margin=dict(t=40, b=36, l=52, r=16),
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            fig_dh_hist.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            fig_dh_hist.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_dh_hist, use_container_width=True)

            # Sample paths + hedge P&L over time
            st.subheader("Sample Paths vs Hedge P&L")
            paths_dh = res_dh["paths"]
            times_dh = np.linspace(0, dh_T, paths_dh.shape[1])
            n_show_dh = min(30, dh_paths)
            idx_dh    = np.random.default_rng(1).choice(dh_paths, n_show_dh, replace=False)

            fig_dh_p = make_subplots(rows=1, cols=2,
                                      subplot_titles=["Simulated paths", "P&L distribution (CDF)"])
            colors_dh = ["#22c55e" if p > 0 else "#ef4444" for p in res_dh["pnl"][idx_dh]]
            for i, c in zip(idx_dh, colors_dh):
                fig_dh_p.add_trace(go.Scatter(x=times_dh, y=paths_dh[i],
                    line=dict(color=c, width=0.8), showlegend=False, hoverinfo="skip"),
                    row=1, col=1)
            fig_dh_p.add_hline(y=dh_K, line_dash="dash", line_color="#f59e0b",
                                annotation_text=f"K={dh_K:.0f}", row=1, col=1)

            sorted_pnl = np.sort(res_dh["pnl"])
            cdf_dh     = np.arange(1, len(sorted_pnl)+1) / len(sorted_pnl)
            fig_dh_p.add_trace(go.Scatter(x=sorted_pnl, y=cdf_dh*100,
                mode="lines", name="CDF", line=dict(color="#6366f1", width=2)),
                row=1, col=2)
            fig_dh_p.add_vline(x=0, line_dash="dot", line_color="white", row=1, col=2)
            fig_dh_p.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=380, showlegend=False,
                font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                margin=dict(t=40, b=36, l=52, r=16),
            )
            fig_dh_p.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            fig_dh_p.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_dh_p, use_container_width=True)

            # Rebalancing frequency analysis
            st.subheader("Hedging Error vs Rebalancing Frequency")
            st.caption("RMSE decays as 1/sqrt(N) — the more frequent, the better the hedge.")
            with st.spinner("Running frequency sweep..."):
                @st.cache_data
                def _freq_sweep(s0_, K_, T_, r_, sig_, q_, opt_, n_p_, n_s_):
                    return rebal_frequency_analysis(s0_, K_, T_, r_, sig_, q_, opt_,
                                                     freqs=[1,2,5,10,21,63],
                                                     n_paths=n_p_, n_steps=n_s_, seed=42)
                freq_res = _freq_sweep(S, dh_K, dh_T, dh_r, dh_sigma, dh_q,
                                        dh_opt, min(dh_paths, 2000), total_steps_dh)

            freq_labels = ["Daily","2d","Weekly","2w","Monthly","Quarterly"]
            fig_freq = make_subplots(rows=1, cols=2,
                                      subplot_titles=["RMSE vs Frequency", "Std P&L vs Frequency"])
            for col_f, metric, color in [(1, "rmse", "#6366f1"), (2, "std_pnl", "#f59e0b")]:
                fig_freq.add_trace(go.Scatter(
                    x=freq_labels, y=freq_res[metric],
                    mode="lines+markers", line=dict(color=color, width=2),
                    marker=dict(size=8), showlegend=False,
                ), row=1, col=col_f)
            fig_freq.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", height=300,
                font=dict(family="Inter, sans-serif", color="#e2e8f0"),
                margin=dict(t=40, b=36, l=52, r=16),
            )
            fig_freq.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
            fig_freq.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(fig_freq, use_container_width=True)

        else:
            st.info("Set parameters and click **Run Hedging Simulation**.")


st.divider()
st.caption("Black-Scholes Pricer | NumPy · SciPy · Streamlit · Plotly")