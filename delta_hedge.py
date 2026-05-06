"""
Dynamic delta hedging simulation.

Strategy: short 1 option, delta-hedge with the underlying.

At each rebalancing step:
  1. Compute BS delta at current (S, t)
  2. Trade the underlying to match the new delta
  3. Finance the trade by borrowing / lending at rate r

P&L at expiry = (option premium invested at r) + (delta hedge gain) - (option payoff)

In a perfect continuous-time BS world this equals zero.
With discrete rebalancing the residual P&L (hedging error) is non-zero
and its variance decays as O(1/sqrt(N_rebal)).
"""

from __future__ import annotations
import numpy as np
from bs_core import call_price, put_price, delta as bs_delta


def simulate_hedge(
    S0: float,
    K:  float,
    T:  float,
    r:  float,
    sigma: float,
    q:  float = 0.0,
    option: str = "call",
    n_paths: int = 2_000,
    n_steps: int = 252,
    rebal_every: int = 1,       # rebalance every N steps (1=daily, 5=weekly…)
    antithetic: bool = True,
    seed: int | None = 42,
) -> dict:
    """
    Simulate delta-hedging of a short option position.

    Parameters
    ----------
    rebal_every : rebalance every this many time steps

    Returns
    -------
    dict with keys:
        pnl          – ndarray (n_paths,)  terminal P&L of the hedge
        paths        – ndarray (n_paths, n_steps+1)
        V0           – float, option price at inception
        mean_pnl     – float
        std_pnl      – float
        rmse         – float  (root-mean-square hedging error)
        sharpe_hedge – float  (mean/std of normalised P&L)
    """
    rng    = np.random.default_rng(seed)
    dt     = T / n_steps
    pricer = call_price if option == "call" else put_price

    # Option value at inception
    V0 = pricer(S0, K, T, r, sigma, q)

    # Simulate paths
    if antithetic:
        half   = n_paths // 2
        Z_half = rng.standard_normal((half, n_steps))
        Z      = np.concatenate([Z_half, -Z_half], axis=0)
    else:
        Z = rng.standard_normal((n_paths, n_steps))

    log_inc   = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1
    )
    paths = S0 * np.exp(log_paths)           # (n_paths, n_steps+1)

    # Vectorised delta-hedge over all paths simultaneously
    # cash[i]  = cash account for path i  (positive = lent, negative = borrowed)
    # delta[i] = current stock holding for path i
    cash      = np.full(n_paths, V0)         # receive option premium
    delta_pos = np.zeros(n_paths)
    disc_step = np.exp(r * dt)

    for step in range(n_steps):
        t_rem = T - step * dt

        # Rebalance on schedule
        if step % rebal_every == 0 and t_rem > 1e-6:
            S_now     = paths[:, step]
            new_delta = np.array([
                bs_delta(s, K, max(t_rem, 1e-6), r, sigma, q, option)
                for s in S_now
            ])
            trade     = new_delta - delta_pos        # shares to buy (+) / sell (-)
            cash     -= trade * S_now                # pay for shares
            delta_pos = new_delta

        # Accrue interest on cash balance
        cash *= disc_step

    # At expiry: liquidate stock position, pay option payoff
    S_T    = paths[:, -1]
    payoff = (np.maximum(S_T - K, 0) if option == "call"
              else np.maximum(K - S_T, 0))

    cash  += delta_pos * S_T       # sell remaining stock
    cash  -= payoff                # pay option holder
    pnl    = cash                  # terminal hedging P&L

    return {
        "pnl":          pnl,
        "paths":        paths,
        "V0":           V0,
        "mean_pnl":     float(pnl.mean()),
        "std_pnl":      float(pnl.std(ddof=1)),
        "rmse":         float(np.sqrt(np.mean(pnl ** 2))),
        "sharpe_hedge": float(pnl.mean() / pnl.std(ddof=1)) if pnl.std() > 0 else 0.0,
        "pct5":         float(np.percentile(pnl, 5)),
        "pct95":        float(np.percentile(pnl, 95)),
    }


def rebal_frequency_analysis(
    S0, K, T, r, sigma, q=0.0, option="call",
    freqs: list[int] | None = None,
    n_paths: int = 5_000,
    n_steps: int = 252,
    seed: int = 42,
) -> dict:
    """
    Run the hedge simulation at multiple rebalancing frequencies.
    Returns arrays: freqs, rmse, std_pnl, mean_pnl.
    """
    if freqs is None:
        freqs = [1, 2, 5, 10, 21, 63]    # daily, 2d, weekly, 2w, monthly, quarterly

    rmses, stds, means = [], [], []
    for f in freqs:
        res = simulate_hedge(S0, K, T, r, sigma, q, option,
                             n_paths=n_paths, n_steps=n_steps,
                             rebal_every=f, seed=seed)
        rmses.append(res["rmse"])
        stds.append(res["std_pnl"])
        means.append(res["mean_pnl"])

    return {
        "freqs":    np.array(freqs),
        "rmse":     np.array(rmses),
        "std_pnl":  np.array(stds),
        "mean_pnl": np.array(means),
    }
