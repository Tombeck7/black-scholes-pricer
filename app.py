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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Black-Scholes Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 4px 0;
    }
    .greek-pos { color: #00d4aa; font-weight: bold; }
    .greek-neg { color: #ff6b6b; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar — Global Parameters ───────────────────────────────────────────────
st.sidebar.title("⚙️ Parameters")

st.sidebar.subheader("Market")
S     = st.sidebar.number_input("Spot  S",        min_value=1.0,   max_value=10000.0, value=100.0, step=1.0)
K     = st.sidebar.number_input("Strike  K",      min_value=1.0,   max_value=10000.0, value=100.0, step=1.0)
T     = st.sidebar.slider("Maturity  T (years)",  min_value=0.01,  max_value=5.0,     value=1.0,   step=0.01)
r     = st.sidebar.slider("Risk-free  r (%)",     min_value=0.0,   max_value=15.0,    value=5.0,   step=0.1) / 100
sigma = st.sidebar.slider("Volatility  σ (%)",    min_value=1.0,   max_value=120.0,   value=20.0,  step=0.5) / 100
q     = st.sidebar.slider("Dividend  q (%)",      min_value=0.0,   max_value=15.0,    value=2.0,   step=0.1) / 100

st.sidebar.subheader("Option type")
opt_type = st.sidebar.radio("", ["Call", "Put"], horizontal=True).lower()

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
        st.dataframe(greek_table(call, "Call"), use_container_width=True)

    with col_p:
        st.subheader("📕 Put Greeks")
        st.dataframe(greek_table(put, "Put"), use_container_width=True)

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
                        template="plotly_dark", margin=dict(t=40, b=20))
    fig_g.update_xaxes(title_text=x_label if mode != "Volatility σ" else "σ (%)")
    st.plotly_chart(fig_g, use_container_width=True)

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
        template="plotly_dark",
        height=350,
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_td, use_container_width=True)


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
        fig_iv.update_layout(template="plotly_dark", height=360,
                             showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig_iv, use_container_width=True)

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
        template="plotly_dark", height=320,
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig_ts, use_container_width=True)


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
                template="plotly_dark",
                height=580,
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_3d, use_container_width=True)

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
                template="plotly_dark", height=480,
            )
            st.plotly_chart(fig_hm, use_container_width=True)

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
                template="plotly_dark", height=480,
                legend=dict(title="Maturity", orientation="v"),
            )
            st.plotly_chart(fig_sm, use_container_width=True)


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
            template="plotly_dark",
            height=460,
            legend=dict(orientation="h", y=1.06),
        )
        st.plotly_chart(fig_st, use_container_width=True)

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

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Black-Scholes Pricer | NumPy · SciPy · Streamlit · Plotly")
