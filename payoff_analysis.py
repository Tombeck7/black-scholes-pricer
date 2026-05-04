"""
Option strategy payoffs, P&L profiles, and structuring analytics.

All payoff functions return P&L arrays over the given S_range.
Premiums are always passed as positive numbers; sign convention is handled
internally (long = pay premium, short = receive premium).
"""

import numpy as np
import matplotlib.pyplot as plt

from bs_core import call_price, put_price, BSOption


# ── Elementary payoffs ────────────────────────────────────────────────────────

def long_call(S_range, K, premium):
    return np.maximum(S_range - K, 0) - premium


def short_call(S_range, K, premium):
    return premium - np.maximum(S_range - K, 0)


def long_put(S_range, K, premium):
    return np.maximum(K - S_range, 0) - premium


def short_put(S_range, K, premium):
    return premium - np.maximum(K - S_range, 0)


# ── Volatility strategies ─────────────────────────────────────────────────────

def long_straddle(S_range, K, call_prem, put_prem):
    """Long ATM call + Long ATM put."""
    return long_call(S_range, K, call_prem) + long_put(S_range, K, put_prem)


def short_straddle(S_range, K, call_prem, put_prem):
    return short_call(S_range, K, call_prem) + short_put(S_range, K, put_prem)


def long_strangle(S_range, K_call, K_put, call_prem, put_prem):
    """Long OTM call (K_call > spot) + Long OTM put (K_put < spot)."""
    return long_call(S_range, K_call, call_prem) + long_put(S_range, K_put, put_prem)


def short_strangle(S_range, K_call, K_put, call_prem, put_prem):
    return short_call(S_range, K_call, call_prem) + short_put(S_range, K_put, put_prem)


# ── Directional spreads ───────────────────────────────────────────────────────

def bull_call_spread(S_range, K_lo, K_hi, prem_lo, prem_hi):
    """Long K_lo call, short K_hi call (K_hi > K_lo)."""
    return long_call(S_range, K_lo, prem_lo) + short_call(S_range, K_hi, prem_hi)


def bear_put_spread(S_range, K_lo, K_hi, prem_lo, prem_hi):
    """Long K_hi put, short K_lo put (K_hi > K_lo)."""
    return long_put(S_range, K_hi, prem_hi) + short_put(S_range, K_lo, prem_lo)


def bull_put_spread(S_range, K_lo, K_hi, prem_lo, prem_hi):
    """Short K_hi put, long K_lo put — credit spread."""
    return short_put(S_range, K_hi, prem_hi) + long_put(S_range, K_lo, prem_lo)


def bear_call_spread(S_range, K_lo, K_hi, prem_lo, prem_hi):
    """Short K_lo call, long K_hi call — credit spread."""
    return short_call(S_range, K_lo, prem_lo) + long_call(S_range, K_hi, prem_hi)


# ── Butterfly & condor ────────────────────────────────────────────────────────

def butterfly_calls(S_range, K_lo, K_mid, K_hi, p_lo, p_mid, p_hi):
    """Long K_lo + Short 2×K_mid + Long K_hi (all calls, equal spacing)."""
    return (long_call(S_range, K_lo, p_lo)
            + short_call(S_range, K_mid, p_mid) * 2
            + long_call(S_range, K_hi, p_hi))


def butterfly_puts(S_range, K_lo, K_mid, K_hi, p_lo, p_mid, p_hi):
    """Same payoff using puts."""
    return (long_put(S_range, K_lo, p_lo)
            + short_put(S_range, K_mid, p_mid) * 2
            + long_put(S_range, K_hi, p_hi))


def iron_condor(S_range,
                K_put_lo, K_put_hi, K_call_lo, K_call_hi,
                p_put_lo, p_put_hi, p_call_lo, p_call_hi):
    """
    Short put spread + Short call spread.
    Net credit = (p_put_hi - p_put_lo) + (p_call_lo - p_call_hi).
    """
    return (bull_put_spread(S_range, K_put_lo, K_put_hi, p_put_lo, p_put_hi)
            + bear_call_spread(S_range, K_call_lo, K_call_hi, p_call_lo, p_call_hi))


# ── Exotic-light strategies ───────────────────────────────────────────────────

def risk_reversal(S_range, K_call, K_put, call_prem, put_prem):
    """Long OTM call, short OTM put — bullish view, zero-cost possible."""
    return long_call(S_range, K_call, call_prem) + short_put(S_range, K_put, put_prem)


def collar(S_range, S0, K_call, K_put, call_prem, put_prem):
    """
    Long stock + Long OTM put + Short OTM call.
    Caps upside, floors downside.
    """
    stock = S_range - S0
    return stock + long_put(S_range, K_put, put_prem) + short_call(S_range, K_call, call_prem)


def covered_call(S_range, S0, K_call, call_prem):
    """Long stock + Short ATM/OTM call — income strategy."""
    return (S_range - S0) + short_call(S_range, K_call, call_prem)


def protective_put(S_range, S0, K_put, put_prem):
    """Long stock + Long OTM put — downside insurance."""
    return (S_range - S0) + long_put(S_range, K_put, put_prem)


# ── Pre-expiry P&L ────────────────────────────────────────────────────────────

def preexpiry_pnl_call(S_range, K, T_entry, T_remaining_list, r, sigma,
                        q=0.0):
    """
    P&L of a long call position at multiple time snapshots.

    Parameters
    ----------
    T_remaining_list : list of remaining times to expiry (years)

    Returns
    -------
    curves : list of P&L arrays (one per time snapshot)
    labels : corresponding labels
    """
    entry_price = call_price(S_range[len(S_range) // 2], K, T_entry, r, sigma, q)
    curves, labels = [], []
    for t in T_remaining_list:
        pnl = np.array([call_price(s, K, max(t, 1e-6), r, sigma, q)
                        for s in S_range]) - entry_price
        curves.append(pnl)
        labels.append(f'T-t = {t:.2f}y')
    return curves, labels


def preexpiry_pnl_straddle(S_range, K, T_entry, T_remaining_list, r, sigma,
                            q=0.0):
    """P&L of a long straddle at multiple time snapshots."""
    c0 = call_price(S_range[len(S_range) // 2], K, T_entry, r, sigma, q)
    p0 = put_price(S_range[len(S_range) // 2], K, T_entry, r, sigma, q)
    entry = c0 + p0
    curves, labels = [], []
    for t in T_remaining_list:
        pnl = np.array([
            call_price(s, K, max(t, 1e-6), r, sigma, q)
            + put_price(s, K, max(t, 1e-6), r, sigma, q)
            - entry
            for s in S_range
        ])
        curves.append(pnl)
        labels.append(f'T-t = {t:.2f}y')
    return curves, labels


# ── Structuring analytics ─────────────────────────────────────────────────────

def breakevens(K, total_premium, option='call'):
    """Return breakeven(s) at expiry for a single option."""
    if option == 'call':
        return K + total_premium
    return K - total_premium


def straddle_breakevens(K, call_prem, put_prem):
    cost = call_prem + put_prem
    return K - cost, K + cost, cost


def max_loss_reward(payoff):
    """Max loss (most negative) and max reward (most positive) in P&L array."""
    return float(payoff.min()), float(payoff.max())


def risk_reward_ratio(payoff):
    loss, reward = max_loss_reward(payoff)
    if loss >= 0:
        return np.inf
    return abs(reward / loss)


def probability_of_profit(payoff, S_range):
    """
    Rough (uniform) probability that the payoff is positive over S_range.
    (Not risk-neutral; purely geometric.)
    """
    return float(np.sum(payoff > 0)) / len(payoff)


def greeks_summary_table(S, K, T, r, sigma, q=0.0):
    """Return a dict of call and put greeks for display."""
    c = BSOption(S, K, T, r, sigma, q, 'call')
    p = BSOption(S, K, T, r, sigma, q, 'put')
    return c.greeks_summary(), p.greeks_summary()


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _payoff_axes(ax, S_range, payoff, color, label, S0=None):
    ax.plot(S_range, payoff, color=color, linewidth=2, label=label)
    ax.axhline(0, color='black', linewidth=0.7)
    if S0 is not None:
        ax.axvline(S0, color='tomato', linestyle='--', linewidth=0.9,
                   alpha=0.8, label=f'Spot={S0}')
    ax.fill_between(S_range, 0, np.maximum(payoff, 0), alpha=0.14, color='limegreen')
    ax.fill_between(S_range, np.minimum(payoff, 0), 0, alpha=0.14, color='red')
    ax.set_xlabel('S at expiry', fontsize=8)
    ax.set_ylabel('P&L', fontsize=8)
    ax.grid(True, alpha=0.28)


def plot_single_payoff(S_range, payoff, title, S0=None,
                       color='steelblue', save_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    _payoff_axes(ax, S_range, payoff, color, title, S0)
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_strategies_grid(strategies_dict, S_range, S0=None,
                          suptitle='Option Strategy Payoffs',
                          ncols=4, save_path=None):
    """
    Plot multiple strategies in a grid.

    Parameters
    ----------
    strategies_dict : OrderedDict {name: (payoff_array, color)}
    """
    n = len(strategies_dict)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4))
    fig.suptitle(suptitle, fontsize=13, fontweight='bold', y=1.01)

    for ax, (name, (payoff, color)) in zip(axes.flatten(), strategies_dict.items()):
        _payoff_axes(ax, S_range, payoff, color, '', S0)
        ax.set_title(name, fontsize=9, fontweight='bold')

    # hide unused axes
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_preexpiry_pnl(S_range, curves, labels, title,
                        S0=None, save_path=None):
    """Plot pre-expiry P&L curves (time decay slices)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(curves)))

    for pnl, lbl, col in zip(curves, labels, colors):
        ax.plot(S_range, pnl, label=lbl, color=col, linewidth=1.8)

    ax.axhline(0, color='black', linewidth=0.8)
    if S0 is not None:
        ax.axvline(S0, color='tomato', linestyle='--', linewidth=1, alpha=0.8)
    ax.set_xlabel('Spot S')
    ax.set_ylabel('P&L')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8, title='Time to expiry')
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_greeks_vs_spot(S_range, K, T, r, sigma, q=0.0, option='call',
                         save_path=None):
    """6-panel figure: price + Δ, Γ, ν, Θ, ρ vs spot."""
    from bs_core import delta, gamma, vega, theta, rho

    pricer = call_price if option == 'call' else put_price

    prices  = [pricer(s, K, T, r, sigma, q) for s in S_range]
    deltas  = [delta(s, K, T, r, sigma, q, option) for s in S_range]
    gammas  = [gamma(s, K, T, r, sigma, q) for s in S_range]
    vegas   = [vega(s, K, T, r, sigma, q) for s in S_range]
    thetas  = [theta(s, K, T, r, sigma, q, option) for s in S_range]
    rhos    = [rho(s, K, T, r, sigma, q, option) for s in S_range]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f'Greeks vs Spot — K={K}, T={T}y, σ={sigma*100:.0f}%, '
        f'r={r*100:.1f}%, {option.upper()}',
        fontsize=12
    )

    data_rows = [
        ('Price',             prices,  'black'),
        ('Delta (Δ)',         deltas,  'royalblue'),
        ('Gamma (Γ)',         gammas,  'seagreen'),
        ('Vega (ν) /1% vol',  vegas,   'darkorange'),
        ('Theta (Θ) /day',    thetas,  'purple'),
        ('Rho (ρ) /1% rate',  rhos,    'crimson'),
    ]
    for ax, (label, vals, color) in zip(axes.flatten(), data_rows):
        ax.plot(S_range, vals, color=color, linewidth=2)
        ax.axvline(K, color='gray', linestyle='--', alpha=0.55, linewidth=1,
                   label=f'K={K}')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel('Spot S', fontsize=8)
        ax.grid(True, alpha=0.28)
        ax.legend(fontsize=7)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
