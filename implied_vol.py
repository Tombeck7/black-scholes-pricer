"""
Implied volatility extraction via Brent's method.

All functions accept scalar inputs; use implied_vol_surface for grid computation.
"""

import numpy as np
from scipy.optimize import brentq

from bs_core import call_price, put_price, delta as bs_delta

_SIGMA_LO = 1e-6
_SIGMA_HI = 10.0


# ── Single-point IV ───────────────────────────────────────────────────────────

def implied_vol(market_price, S, K, T, r, q=0.0, option='call',
                tol=1e-9, max_iter=1_000):
    """
    Extract implied volatility from a market price via Brent inversion.

    Returns
    -------
    float
        IV if converged, np.nan otherwise.
    """
    if T <= 0:
        return np.nan

    pricer = call_price if option.lower() == 'call' else put_price

    # Arbitrage / boundary checks
    fwd    = S * np.exp(-q * T)
    pv_K   = K * np.exp(-r * T)
    intrinsic = max(fwd - pv_K, 0) if option == 'call' else max(pv_K - fwd, 0)
    if market_price < intrinsic - 1e-8 or market_price <= 0:
        return np.nan

    def obj(sigma):
        return pricer(S, K, T, r, sigma, q) - market_price

    try:
        f_lo = obj(_SIGMA_LO)
        f_hi = obj(_SIGMA_HI)
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(obj, _SIGMA_LO, _SIGMA_HI, xtol=tol, maxiter=max_iter)
    except (ValueError, RuntimeError):
        return np.nan


# ── Surface (vectorised) ──────────────────────────────────────────────────────

def implied_vol_surface(market_prices, S, strikes, maturities, r, q=0.0,
                        option='call'):
    """
    Compute the IV surface from a matrix of market prices.

    Parameters
    ----------
    market_prices : ndarray, shape (len(maturities), len(strikes))
    strikes       : 1-D array
    maturities    : 1-D array

    Returns
    -------
    ndarray, same shape as market_prices.
    """
    iv = np.empty_like(market_prices, dtype=float)
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            iv[i, j] = implied_vol(market_prices[i, j], S, K, T, r, q, option)
    return iv


# ── Delta → strike mapping ────────────────────────────────────────────────────

def strike_from_delta(target_delta, S, T, r, sigma, q=0.0, option='call',
                      tol=1e-9):
    """
    Find the strike K such that BS delta equals target_delta.

    Useful for quoting in delta space (e.g. 25-delta call / put).
    """
    if T <= 0:
        return np.nan

    def obj(K):
        return bs_delta(S, K, T, r, sigma, q, option) - target_delta

    try:
        return brentq(obj, S * 0.01, S * 20, xtol=tol)
    except ValueError:
        return np.nan


# ── Vol-of-vol / local vol helpers ────────────────────────────────────────────

def iv_term_structure(market_prices_atm, S, maturities, r, q=0.0, option='call'):
    """
    Extract the ATM IV term structure from a list of ATM prices.

    Parameters
    ----------
    market_prices_atm : list / array of ATM market prices, one per maturity.

    Returns
    -------
    ndarray of IV values.
    """
    return np.array([
        implied_vol(px, S, S, T, r, q, option)
        for px, T in zip(market_prices_atm, maturities)
    ])


def forward_vol(iv1, iv2, T1, T2):
    """
    Bootstrap the forward (instantaneous) vol between T1 and T2
    from two ATM IVs.

    σ_fwd = sqrt( (σ2²·T2 - σ1²·T1) / (T2 - T1) )
    """
    if T2 <= T1:
        raise ValueError("T2 must be strictly greater than T1.")
    var_fwd = (iv2 ** 2 * T2 - iv1 ** 2 * T1) / (T2 - T1)
    return np.sqrt(max(var_fwd, 0.0))


def iv_surface_by_delta(S, T_list, sigma_atm, skew25=0.02, wing_conv=0.01,
                        r=0.0, q=0.0):
    """
    Build a parametric IV surface quoted in delta space.

    At each maturity:
        σ(Δ) = σ_atm + skew25 * (Δ - 0.5) * 2 + wing_conv * (Δ - 0.5)^2 * 4

    Returns
    -------
    deltas : 1-D array (0.05 .. 0.95)
    iv_mat : ndarray shape (len(T_list), len(deltas))
    strikes: ndarray, same shape as iv_mat
    """
    deltas = np.linspace(0.05, 0.95, 19)
    iv_mat  = np.zeros((len(T_list), len(deltas)))
    strikes = np.zeros_like(iv_mat)

    for i, T in enumerate(T_list):
        atm = sigma_atm[i] if hasattr(sigma_atm, '__len__') else sigma_atm
        sk  = skew25 * np.exp(-0.4 * T)
        cv  = wing_conv * np.exp(-0.3 * T)
        for j, d in enumerate(deltas):
            iv_val = atm + sk * (d - 0.5) * 2 + cv * (d - 0.5) ** 2 * 4
            iv_mat[i, j] = max(iv_val, 0.02)
            K = strike_from_delta(d, S, T, r, iv_val, q, option='call')
            strikes[i, j] = K if not np.isnan(K) else S

    return deltas, iv_mat, strikes
