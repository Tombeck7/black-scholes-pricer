
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
