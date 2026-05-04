import sys; sys.stdout.reconfigure(encoding="utf-8")
"""
Black-Scholes Pricer -- Complete Demo
======================================
Covers: European pricing, analytical Greeks (Delta Gamma Vega Theta Rho + Vanna/Volga/Charm),
implied vol extraction via Brent inversion, vol smile/skew/surface,
strategy payoff diagrams, structuring analytics.

All plots are saved to  figures/  (non-interactive Agg backend).
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from bs_core import (
    BSOption, call_price, put_price,
    delta, gamma, vega, theta, rho, vanna, volga, charm,
)
from implied_vol import implied_vol, implied_vol_surface
from vol_surface import (
    realistic_smile, svi_total_variance,
    plot_vol_smile, plot_vol_skew,
    plot_vol_surface_3d, plot_iv_heatmap,
    plot_atm_term_structure,
    plot_delta_gamma_surface, plot_vega_theta_surface,
)
from payoff_analysis import (
    long_call, long_put, long_straddle, long_strangle,
    short_straddle, bull_call_spread, bear_put_spread,
    butterfly_calls, iron_condor, risk_reversal,
    collar, covered_call, protective_put,
    preexpiry_pnl_straddle,
    straddle_breakevens, risk_reward_ratio,
    plot_strategies_grid, plot_preexpiry_pnl, plot_greeks_vs_spot,
)

os.makedirs('figures', exist_ok=True)

# ============================================================================
# Market Parameters
# ============================================================================
S     = 100.0   # spot
K     = 100.0   # ATM strike
T     = 1.0     # 1 year
r     = 0.05    # 5 % risk-free rate
q     = 0.02    # 2 % continuous dividend yield
sigma = 0.20    # 20 % flat vol

sep = "=" * 64

# ============================================================================
# 1 . Pricing
# ============================================================================
print(sep)
print("  BLACK-SCHOLES PRICER -- COMPLETE DEMO")
print(sep)

call = BSOption(S, K, T, r, sigma, q, 'call')
put  = BSOption(S, K, T, r, sigma, q, 'put')

print(f"\n{'-'*64}")
print("  1 . EUROPEAN PRICING")
print(f"{'-'*64}")
print(f"  S={S}  K={K}  T={T}y  r={r*100:.1f}%  sigma={sigma*100:.0f}%  q={q*100:.1f}%\n")
print(f"  Call price       : {call.price:>9.4f}")
print(f"  Put  price       : {put.price:>9.4f}")
pcp = call.price - put.price
pcp_theory = S * np.exp(-q * T) - K * np.exp(-r * T)
print(f"  Put-Call parity  : C - P = {pcp:.4f}  |  S.e^(-qT) - K.e^(-rT) = {pcp_theory:.4f}"
      f"  OK" if abs(pcp - pcp_theory) < 1e-8 else "  ✗")
print(f"  Call time value  : {call.time_value:.4f}")
print(f"  Call moneyness   : {call.moneyness:.4f}")

# ============================================================================
# 2 . Greeks
# ============================================================================
print(f"\n{'-'*64}")
print("  2 . ANALYTICAL GREEKS")
print(f"{'-'*64}")
print(f"  {'Greek':<12} {'Call':>12} {'Put':>12}  Description")
print("  " + "-" * 56)
greek_meta = [
    ('Price',  'Option fair value'),
    ('Delta',  'dP/dS    -- spot sensitivity'),
    ('Gamma',  'd^2P/dS^2  -- curvature (same call/put)'),
    ('Vega',   'dP/dsigma    -- vol sensitivity per 1%'),
    ('Theta',  'dP/dt    -- time decay per day'),
    ('Rho',    'dP/dr    -- rate sensitivity per 1%'),
    ('Vanna',  'dDelta/dsigma    -- cross sensitivity'),
    ('Volga',  'dVega/dsigma -- vol convexity'),
    ('Charm',  'dDelta/dt    -- delta decay per day'),
    ('Speed',  'dGamma/dS    -- gamma convexity'),
]
cg = call.greeks_summary()
pg = put.greeks_summary()
for name, desc in greek_meta:
    cv = cg.get(name, call.speed if name == 'Speed' else float('nan'))
    pv = pg.get(name, put.speed  if name == 'Speed' else float('nan'))
    print(f"  {name:<12} {cv:>12.6f} {pv:>12.6f}  {desc}")

# ============================================================================
# 3 . Implied Volatility via Brent
# ============================================================================
print(f"\n{'-'*64}")
print("  3 . IMPLIED VOLATILITY (BRENT INVERSION)")
print(f"{'-'*64}")

iv_rt = implied_vol(call.price, S, K, T, r, q, 'call')
print(f"  Round-trip IV from call price {call.price:.4f}: sigma_impl = {iv_rt*100:.6f}%  "
      f"(input = {sigma*100:.4f}%)")

# smile grid: each strike has its own "true" vol (realistic skew)
strikes_smile = np.linspace(70, 135, 27)
log_m_smile   = np.log(strikes_smile / S)
true_vols     = realistic_smile(strikes_smile, S, T,
                                atm_vol=0.20, skew_coef=-0.08, conv_coef=0.10)

# price with the true vol, then recover IV
iv_smile = np.array([
    implied_vol(call_price(S, K_i, T, r, sv, q), S, K_i, T, r, q, 'call')
    for K_i, sv in zip(strikes_smile, true_vols)
])

print(f"\n  Strike-space smile (sample):")
print(f"  {'Strike':>8} {'True sigma':>8} {'IV_impl':>8}")
for i in range(0, len(strikes_smile), 4):
    print(f"  {strikes_smile[i]:>8.1f} {true_vols[i]*100:>7.2f}% "
          f"{iv_smile[i]*100:>7.2f}%")

# ============================================================================
# 4 . Volatility Surface
# ============================================================================
print(f"\n{'-'*64}")
print("  4 . VOL SURFACE CONSTRUCTION")
print(f"{'-'*64}")

maturities   = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
strikes_surf = np.linspace(70, 135, 27)
iv_surface   = np.zeros((len(maturities), len(strikes_surf)))

for i, T_i in enumerate(maturities):
    # ATM vol slightly inverted term structure (short end elevated)
    atm_v = 0.20 + 0.03 * np.exp(-2.5 * T_i)
    for j, K_j in enumerate(strikes_surf):
        iv_surface[i, j] = realistic_smile(
            [K_j], S, T_i, atm_vol=atm_v,
            skew_coef=-0.08, conv_coef=0.10
        )[0]

atm_idx  = np.argmin(np.abs(strikes_surf - S))
atm_vols = iv_surface[:, atm_idx]

print(f"  ATM vol term structure:")
print(f"  {'Maturity':>12} {'ATM IV':>10} {'Fwd vol (from prev)':>22}")
for i, (Ti, vi) in enumerate(zip(maturities, atm_vols)):
    if i == 0:
        fwd_str = "        --"
    else:
        from implied_vol import forward_vol
        fv = forward_vol(atm_vols[i-1], vi, maturities[i-1], Ti)
        fwd_str = f"{fv*100:>9.2f}%"
    print(f"  {Ti:>12.4f}y {vi*100:>9.2f}%  {fwd_str}")

# ============================================================================
# 5 . Structuring Analysis
# ============================================================================
print(f"\n{'-'*64}")
print("  5 . STRUCTURING ANALYSIS")
print(f"{'-'*64}")

c_atm  = call_price(S, K, T, r, sigma, q)
p_atm  = put_price(S, K, T, r, sigma, q)
K_hi   = K * 1.10
K_lo   = K * 0.95
K_c_otm = K * 1.05
K_p_otm = K * 0.95
c_hi   = call_price(S, K_hi,   T, r, sigma, q)
c_otm  = call_price(S, K_c_otm, T, r, sigma, q)
p_otm  = put_price(S,  K_p_otm, T, r, sigma, q)
c_lo   = call_price(S, K_lo,   T, r, sigma, q)
p_lo   = put_price(S,  K_lo,   T, r, sigma, q)

# -- Straddle --
be_lo, be_hi, total_prem = straddle_breakevens(K, c_atm, p_atm)
print(f"\n  * ATM Straddle")
print(f"    Total premium   : {total_prem:.4f}")
print(f"    Upper break-even: {be_hi:.2f}  (+{(be_hi/S - 1)*100:.1f}%)")
print(f"    Lower break-even: {be_lo:.2f}  ({(be_lo/S - 1)*100:.1f}%)")
print(f"    Implied +/-move   : {total_prem/S*100:.2f}%  (vs 1sigma = {sigma*100:.2f}%)")

# -- Bull call spread --
spread_cost = c_atm - c_hi
max_profit  = (K_hi - K) - spread_cost
be_spread   = K + spread_cost
print(f"\n  * Bull Call Spread (K={K} / K_hi={K_hi})")
print(f"    Net debit       : {spread_cost:.4f}")
print(f"    Max profit      : {max_profit:.4f}  at S >= {K_hi}")
print(f"    Break-even      : {be_spread:.2f}")
print(f"    Max R/R ratio   : {max_profit/spread_cost:.2f}x")

# -- Gamma / Theta tradeoff --
print(f"\n  * Gamma / Theta (convexity cost)")
print(f"    Call  Gamma/|Theta| = {call.gamma / abs(call.theta):.4f}  (gamma per unit daily decay)")
print(f"    Put   Gamma/|Theta| = {put.gamma  / abs(put.theta):.4f}")
straddle_g = 2 * call.gamma
straddle_t = call.theta + put.theta
print(f"    Straddle Gamma/|Theta| = {straddle_g / abs(straddle_t):.4f}")

# -- Skew implications --
print(f"\n  * Vol Skew implications")
skew_slope = (true_vols[0] - true_vols[-1]) / (log_m_smile[0] - log_m_smile[-1])
print(f"    Approx dsigma/d(log K) ~= {skew_slope*100:.1f}% per unit log-moneyness")
put_25d_K  = K * 0.95
call_25d_K = K * 1.05
iv_put25   = realistic_smile([put_25d_K],  S, T, 0.20)[0]
iv_call25  = realistic_smile([call_25d_K], S, T, 0.20)[0]
print(f"    25-Delta risk reversal: sigma(OTM put) - sigma(OTM call) ~= {(iv_put25-iv_call25)*100:.2f}%")
print(f"    -> OTM puts priced richer  -- market prices left-tail risk > right-tail gains")
print(f"    -> Structuring implication : puts overvalued for buyers; sell via spreads or RR")

# ============================================================================
# 6 . Generate All Figures
# ============================================================================
print(f"\n{'-'*64}")
print("  6 . GENERATING FIGURES  ->  figures/")
print(f"{'-'*64}")

S_range = np.linspace(60, 145, 350)

# -- Fig 01 : Greeks vs Spot --------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle(
    f'Greeks vs Spot -- K={K}, T={T}y, sigma={sigma*100:.0f}%, r={r*100:.1f}%, q={q*100:.1f}%',
    fontsize=12
)

prices_c = np.array([call_price(s, K, T, r, sigma, q) for s in S_range])
prices_p = np.array([put_price(s,  K, T, r, sigma, q) for s in S_range])
deltas_c = np.array([delta(s, K, T, r, sigma, q, 'call') for s in S_range])
deltas_p = np.array([delta(s, K, T, r, sigma, q, 'put')  for s in S_range])
gammas_v = np.array([gamma(s, K, T, r, sigma, q)          for s in S_range])
vegas_v  = np.array([vega(s,  K, T, r, sigma, q)          for s in S_range])
thetas_c = np.array([theta(s, K, T, r, sigma, q, 'call') for s in S_range])
thetas_p = np.array([theta(s, K, T, r, sigma, q, 'put')  for s in S_range])
rhos_c   = np.array([rho(s,   K, T, r, sigma, q, 'call') for s in S_range])
rhos_p   = np.array([rho(s,   K, T, r, sigma, q, 'put')  for s in S_range])

panel_data = [
    (axes[0,0], 'Price', [(prices_c,'royalblue','Call'), (prices_p,'tomato','Put')], True),
    (axes[0,1], 'Delta (Delta)', [(deltas_c,'royalblue','Call'), (deltas_p,'tomato','Put')], True),
    (axes[0,2], 'Gamma (Gamma) -- identical call/put', [(gammas_v,'seagreen','Gamma')], False),
    (axes[1,0], 'Vega (Vega) per 1% vol move', [(vegas_v,'darkorange','Vega')], False),
    (axes[1,1], 'Theta (Theta) per calendar day', [(thetas_c,'royalblue','Call'), (thetas_p,'tomato','Put')], True),
    (axes[1,2], 'Rho (Rho) per 1% rate move',  [(rhos_c,'royalblue','Call'), (rhos_p,'tomato','Put')], True),
]
for ax, title, curves, need_legend in panel_data:
    for vals, col, lbl in curves:
        ax.plot(S_range, vals, color=col, linewidth=1.9, label=lbl)
    ax.axvline(K, color='gray', linestyle='--', linewidth=0.9, alpha=0.5, label=f'K={K}')
    ax.axhline(0, color='black', linewidth=0.4)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel('Spot S', fontsize=8)
    ax.grid(True, alpha=0.25)
    if need_legend:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('figures/01_greeks_vs_spot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/01_greeks_vs_spot.png")

# -- Fig 02 : Vol Smile & Skew by Maturity ------------------------------------
T_list    = [1/12, 3/12, 6/12, 1.0, 2.0]
T_labels  = ['1M', '3M', '6M', '1Y', '2Y']
smile_curves = []
for Ti in T_list:
    atm_v = 0.20 + 0.03 * np.exp(-2.5 * Ti)
    smile_curves.append(realistic_smile(strikes_smile, S, Ti, atm_vol=atm_v))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Implied Volatility Smile & Skew by Maturity', fontsize=12)

colors_mat = plt.cm.viridis(np.linspace(0.1, 0.9, len(T_list)))
for iv_c, lbl, col in zip(smile_curves, T_labels, colors_mat):
    axes[0].plot(strikes_smile, iv_c * 100, 'o-', markersize=3,
                 linewidth=1.8, label=lbl, color=col)
    axes[1].plot(log_m_smile, iv_c * 100, 'o-', markersize=3,
                 linewidth=1.8, label=lbl, color=col)

axes[0].axvline(S, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Spot')
axes[0].set_xlabel('Strike K'); axes[0].set_ylabel('IV (%)')
axes[0].set_title('Smile (strike space)'); axes[0].legend(title='Maturity', fontsize=7)
axes[0].grid(True, alpha=0.25)

axes[1].axvline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='ATM')
axes[1].set_xlabel('log(K/S)'); axes[1].set_ylabel('IV (%)')
axes[1].set_title('Skew (log-moneyness space)'); axes[1].legend(title='Maturity', fontsize=7)
axes[1].grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('figures/02_vol_smile_skew.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/02_vol_smile_skew.png")

# -- Fig 03 : 3-D Vol Surface -------------------------------------------------
K_grid, T_grid = np.meshgrid(strikes_surf, maturities)
fig = plt.figure(figsize=(12, 7))
ax  = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(K_grid, T_grid, iv_surface * 100,
                       cmap='RdYlGn_r', alpha=0.88, edgecolor='none')
ax.set_xlabel('Strike K', labelpad=8)
ax.set_ylabel('Maturity T (years)', labelpad=8)
ax.set_zlabel('IV (%)', labelpad=8)
ax.set_title('Implied Volatility Surface', pad=14)
z_top = iv_surface.max() * 100 * 1.02
ax.plot([S, S], [maturities[0], maturities[-1]], [z_top, z_top],
        'r--', linewidth=2, label=f'Spot={S}')
ax.legend()
fig.colorbar(surf, ax=ax, shrink=0.45, aspect=12, pad=0.08, label='IV (%)')
plt.tight_layout()
plt.savefig('figures/03_vol_surface_3d.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/03_vol_surface_3d.png")

# -- Fig 04 : IV Heatmap (top-down) -------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
cf = ax.contourf(strikes_surf, maturities * 12, iv_surface * 100,
                 levels=25, cmap='RdYlGn_r')
ax.contour(strikes_surf, maturities * 12, iv_surface * 100,
           levels=12, colors='white', alpha=0.2, linewidths=0.5)
fig.colorbar(cf, ax=ax, label='IV (%)')
ax.axvline(S, color='red', linestyle='--', linewidth=1.5, label=f'Spot={S}')
ax.set_xlabel('Strike K'); ax.set_ylabel('Maturity (months)')
ax.set_title('Implied Volatility -- Top-down Heatmap')
ax.legend()
plt.tight_layout()
plt.savefig('figures/04_iv_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/04_iv_heatmap.png")

# -- Fig 05 : ATM Term Structure -----------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(maturities * 12, atm_vols * 100, 'o-', color='steelblue',
        linewidth=2, markersize=6, label='ATM IV')

# overlay forward vols
from implied_vol import forward_vol as fwd_v
fwd_vols = [float('nan')]
for i in range(1, len(maturities)):
    fv = fwd_v(atm_vols[i-1], atm_vols[i], maturities[i-1], maturities[i])
    fwd_vols.append(fv * 100)
ax.plot(maturities * 12, fwd_vols, 's--', color='darkorange',
        linewidth=1.5, markersize=5, label='Forward vol (bootstrapped)')

ax.set_xlabel('Maturity (months)'); ax.set_ylabel('Vol (%)')
ax.set_title('ATM Implied Vol & Forward Vol Term Structure')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('figures/05_atm_term_structure.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/05_atm_term_structure.png")

# -- Fig 06 : Strategy Payoffs Grid -------------------------------------------
S_range_strat = np.linspace(S * 0.60, S * 1.45, 400)
K_c5  = K * 1.05
K_p5  = K * 0.95
K_c10 = K * 1.10
K_lo2 = K * 0.92

c5  = call_price(S, K_c5,  T, r, sigma, q)
c10 = call_price(S, K_c10, T, r, sigma, q)
p5  = put_price(S,  K_p5,  T, r, sigma, q)
p10 = put_price(S,  K_lo2, T, r, sigma, q)
clo = call_price(S, K_lo2, T, r, sigma, q)

strategies = {
    'Long Call (ATM)':
        (long_call(S_range_strat, K, c_atm), 'royalblue'),
    'Long Put (ATM)':
        (long_put(S_range_strat, K, p_atm), 'tomato'),
    'Long Straddle':
        (long_straddle(S_range_strat, K, c_atm, p_atm), 'purple'),
    'Long Strangle (+5% / -5%)':
        (long_strangle(S_range_strat, K_c5, K_p5, c5, p5), 'darkorchid'),
    'Bull Call Spread':
        (bull_call_spread(S_range_strat, K, K_c10, c_atm, c10), 'seagreen'),
    'Bear Put Spread':
        (bear_put_spread(S_range_strat, K_p5, K, p5, p_atm), 'orangered'),
    'Call Butterfly':
        (butterfly_calls(S_range_strat, K_p5, K, K_c5, p5, c_atm, c5), 'goldenrod'),
    'Short Straddle (seller)':
        (short_straddle(S_range_strat, K, c_atm, p_atm), 'black'),
    'Risk Reversal (+5c / -5p)':
        (risk_reversal(S_range_strat, K_c5, K_p5, c5, p5), 'teal'),
    'Covered Call':
        (covered_call(S_range_strat, S, K_c5, c5), 'steelblue'),
    'Protective Put':
        (protective_put(S_range_strat, S, K_p5, p5), 'firebrick'),
    'Collar (-5p / +5c)':
        (collar(S_range_strat, S, K_c5, K_p5, c5, p5), 'darkcyan'),
}

fig, axes = plt.subplots(3, 4, figsize=(19, 13))
fig.suptitle(
    f'Option Strategies -- S={S}, K={K}, T={T}y, sigma={sigma*100:.0f}%, '
    f'r={r*100:.1f}%, q={q*100:.1f}%',
    fontsize=13, fontweight='bold'
)

for ax, (name, (payoff, color)) in zip(axes.flatten(), strategies.items()):
    ax.plot(S_range_strat, payoff, color=color, linewidth=2)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.axvline(S, color='tomato', linestyle='--', linewidth=0.9, alpha=0.75)
    ax.fill_between(S_range_strat, 0, np.maximum(payoff, 0),
                    alpha=0.13, color='limegreen')
    ax.fill_between(S_range_strat, np.minimum(payoff, 0), 0,
                    alpha=0.13, color='red')
    ax.set_title(name, fontsize=8.5, fontweight='bold')
    ax.set_xlabel('S at expiry', fontsize=7.5)
    ax.set_ylabel('P&L', fontsize=7.5)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig('figures/06_strategy_payoffs.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/06_strategy_payoffs.png")

# -- Fig 07 : Pre-expiry P&L + Theta Decay ------------------------------------
S_range_decay = np.linspace(S * 0.70, S * 1.30, 300)
T_snapshots   = [T, T * 0.75, T * 0.50, T * 0.25, T * 0.10, T * 0.02]
T_snap_labels = [f'T-t={t:.2f}y' for t in T_snapshots]

straddle_curves, straddle_labels = preexpiry_pnl_straddle(
    S_range_decay, K, T, T_snapshots, r, sigma, q
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Time Decay (Theta) Analysis', fontsize=12)

colors_snap = plt.cm.cool(np.linspace(0.05, 0.95, len(T_snapshots)))
for pnl, lbl, col in zip(straddle_curves, straddle_labels, colors_snap):
    axes[0].plot(S_range_decay, pnl, label=lbl, color=col, linewidth=1.8)
axes[0].axhline(0, color='black', linewidth=0.8)
axes[0].axvline(S, color='red', linestyle='--', linewidth=1, alpha=0.7)
axes[0].set_xlabel('Spot S'); axes[0].set_ylabel('P&L')
axes[0].set_title('Long Straddle -- P&L snapshots (time decay)')
axes[0].legend(fontsize=7, title='Time remaining'); axes[0].grid(True, alpha=0.28)

# ATM call price over time (theta decay curve)
times_decay = np.linspace(0.005, T, 250)
atm_call_prices = [call_price(S, K, t, r, sigma, q) for t in times_decay]
axes[1].plot(times_decay, atm_call_prices, color='royalblue', linewidth=2)
axes[1].fill_between(times_decay, 0, atm_call_prices, alpha=0.12, color='royalblue')
axes[1].set_xlabel('Time to Expiry (years)'); axes[1].set_ylabel('ATM Call Price')
axes[1].set_title('ATM Call Price vs Time to Expiry')
axes[1].invert_xaxis()
axes[1].grid(True, alpha=0.28)

plt.tight_layout()
plt.savefig('figures/07_time_decay.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/07_time_decay.png")

# -- Fig 08 : Vega / Vol Sensitivity ------------------------------------------
sigmas_range = np.linspace(0.04, 0.65, 250)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Volatility Sensitivity (Vega Profile)', fontsize=12)

c_prices_vol = [call_price(S, K, T, r, s, q) for s in sigmas_range]
p_prices_vol = [put_price(S,  K, T, r, s, q) for s in sigmas_range]
axes[0].plot(sigmas_range * 100, c_prices_vol, label='Call', color='royalblue', lw=2)
axes[0].plot(sigmas_range * 100, p_prices_vol, label='Put',  color='tomato',    lw=2)
axes[0].axvline(sigma * 100, color='gray', linestyle='--', lw=1, label=f'Current sigma={sigma*100:.0f}%')
axes[0].set_xlabel('sigma (%)'); axes[0].set_ylabel('Price'); axes[0].set_title('Price vs Vol')
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.28)

vegas_vol = [vega(S, K, T, r, s, q) for s in sigmas_range]
axes[1].plot(sigmas_range * 100, vegas_vol, color='darkorange', lw=2)
axes[1].axvline(sigma * 100, color='gray', linestyle='--', lw=1)
axes[1].set_xlabel('sigma (%)'); axes[1].set_ylabel('Vega (per 1% vol)'); axes[1].set_title('Vega vs Vol')
axes[1].grid(True, alpha=0.28)

volgas_vol = [volga(S, K, T, r, s, q) for s in sigmas_range]
axes[2].plot(sigmas_range * 100, volgas_vol, color='purple', lw=2, label='Volga dVega/dsigma')
axes[2].axvline(sigma * 100, color='gray', linestyle='--', lw=1)
axes[2].axhline(0, color='black', lw=0.5)
axes[2].set_xlabel('sigma (%)'); axes[2].set_ylabel('Volga'); axes[2].set_title('Volga (vol convexity)')
axes[2].grid(True, alpha=0.28)

plt.tight_layout()
plt.savefig('figures/08_vol_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/08_vol_sensitivity.png")

# -- Fig 09 : Delta & Gamma 3-D Surfaces --------------------------------------
S_surf_range   = np.linspace(70, 135, 60)
sig_surf_range = np.linspace(0.08, 0.50, 50)

S_g, sig_g = np.meshgrid(S_surf_range, sig_surf_range)
D_g  = np.array([[delta(s, K, T, r, sg, q, 'call') for s in S_surf_range]
                  for sg in sig_surf_range])
G_g  = np.array([[gamma(s, K, T, r, sg, q) for s in S_surf_range]
                  for sg in sig_surf_range])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                               subplot_kw={'projection': '3d'})
fig.suptitle(f'Delta & Gamma surfaces -- K={K}, T={T}y, Call', fontsize=12)

for ax, data, label, cmap_n in [(ax1, D_g, 'Delta (Delta)', 'coolwarm'),
                                  (ax2, G_g, 'Gamma (Gamma)', 'viridis')]:
    surf2 = ax.plot_surface(S_g, sig_g * 100, data,
                            cmap=cmap_n, alpha=0.85, edgecolor='none')
    ax.set_xlabel('Spot S', labelpad=6)
    ax.set_ylabel('Vol sigma (%)', labelpad=6)
    ax.set_zlabel(label, labelpad=6)
    ax.set_title(label)
    fig.colorbar(surf2, ax=ax, shrink=0.4, pad=0.08)

plt.tight_layout()
plt.savefig('figures/09_delta_gamma_surface.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/09_delta_gamma_surface.png")

# -- Fig 10 : Vega & Theta 3-D Surfaces ---------------------------------------
T_surf_range = np.linspace(0.02, 2.0, 50)
S_g2, T_g2 = np.meshgrid(S_surf_range, T_surf_range)
V_g2  = np.array([[vega(s,  K, t, r, sigma, q) for s in S_surf_range]
                   for t in T_surf_range])
TH_g2 = np.array([[theta(s, K, t, r, sigma, q, 'call') for s in S_surf_range]
                   for t in T_surf_range])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                               subplot_kw={'projection': '3d'})
fig.suptitle(f'Vega & Theta surfaces -- K={K}, sigma={sigma*100:.0f}%, Call', fontsize=12)

for ax, data, label, cmap_n in [(ax1, V_g2,  'Vega / 1%', 'YlOrRd'),
                                  (ax2, TH_g2, 'Theta /day', 'PuRd')]:
    surf3 = ax.plot_surface(S_g2, T_g2, data,
                            cmap=cmap_n, alpha=0.85, edgecolor='none')
    ax.set_xlabel('Spot S', labelpad=6)
    ax.set_ylabel('Time to Expiry T', labelpad=6)
    ax.set_zlabel(label, labelpad=6)
    ax.set_title(label)
    fig.colorbar(surf3, ax=ax, shrink=0.4, pad=0.08)

plt.tight_layout()
plt.savefig('figures/10_vega_theta_surface.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/10_vega_theta_surface.png")

# -- Fig 11 : Vanna & Volga Profile -------------------------------------------
vannas  = np.array([vanna(s, K, T, r, sigma, q) for s in S_range])
volgas_ = np.array([volga(s, K, T, r, sigma, q) for s in S_range])
charms  = np.array([charm(s, K, T, r, sigma, q, 'call') for s in S_range])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f'Second-order Greeks -- K={K}, T={T}y, sigma={sigma*100:.0f}%', fontsize=12)

for ax, vals, label, color in [
        (axes[0], vannas,  'Vanna (dDelta/dsigma)', 'mediumvioletred'),
        (axes[1], volgas_, 'Volga (dVega/dsigma)', 'darkorange'),
        (axes[2], charms,  'Charm (dDelta/dt /day)', 'steelblue'),
]:
    ax.plot(S_range, vals, color=color, linewidth=2)
    ax.axvline(K, color='gray', linestyle='--', lw=0.9, alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(label, fontsize=10)
    ax.set_xlabel('Spot S', fontsize=8)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig('figures/11_second_order_greeks.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/11_second_order_greeks.png")

# -- Fig 12 : IV Extraction Round-trip Accuracy -------------------------------
strikes_rt = np.linspace(75, 130, 40)
true_vols_rt = realistic_smile(strikes_rt, S, T, atm_vol=0.20)
iv_recovered = np.array([
    implied_vol(call_price(S, Ki, T, r, sv, q), S, Ki, T, r, q, 'call')
    for Ki, sv in zip(strikes_rt, true_vols_rt)
])
error_bp = (iv_recovered - true_vols_rt) * 10_000   # basis points

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Implied Vol Extraction -- Brent Round-trip Accuracy', fontsize=12)

axes[0].plot(strikes_rt, true_vols_rt * 100,   'o-', color='steelblue', lw=2, label='True sigma(K)')
axes[0].plot(strikes_rt, iv_recovered * 100, 'x--', color='tomato',    lw=1.5, label='IV extracted')
axes[0].set_xlabel('Strike K'); axes[0].set_ylabel('IV (%)')
axes[0].set_title('True vs Extracted IV Smile'); axes[0].legend(); axes[0].grid(True, alpha=0.28)

axes[1].bar(strikes_rt, error_bp, width=1.2, color='coral', alpha=0.8)
axes[1].axhline(0, color='black', lw=0.7)
axes[1].set_xlabel('Strike K'); axes[1].set_ylabel('Error (basis points)')
axes[1].set_title('Extraction Error (|sigma_true - sigma_impl|) in bps')
axes[1].grid(True, alpha=0.28)

plt.tight_layout()
plt.savefig('figures/12_iv_extraction_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  OK  figures/12_iv_extraction_accuracy.png")

# ============================================================================
print(f"\n{'='*64}")
print(f"  12 figures saved to  figures/")
print(f"{'='*64}\n")
