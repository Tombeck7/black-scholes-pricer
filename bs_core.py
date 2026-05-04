"""
Black-Scholes core engine.

Conventions
-----------
S     : current spot price
K     : strike price
T     : time to expiry in years
r     : continuously-compounded risk-free rate
sigma : annualised implied / input volatility
q     : continuous dividend / repo yield (default 0)
"""

import numpy as np
from scipy.stats import norm

N = norm.cdf   # standard-normal CDF
n = norm.pdf   # standard-normal PDF


# ── d1 / d2 ──────────────────────────────────────────────────────────────────

def d1(S, K, T, r, sigma, q=0.0):
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S, K, T, r, sigma, q=0.0):
    return d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


# ── Price ─────────────────────────────────────────────────────────────────────

def call_price(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return float(max(S * np.exp(-q * T) - K, 0))
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * N(_d1) - K * np.exp(-r * T) * N(_d2)


def put_price(S, K, T, r, sigma, q=0.0):
    if T <= 0:
        return float(max(K - S * np.exp(-q * T), 0))
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * N(-_d2) - S * np.exp(-q * T) * N(-_d1)


# ── First-order Greeks ────────────────────────────────────────────────────────

def delta(S, K, T, r, sigma, q=0.0, option='call'):
    if T <= 0:
        if option == 'call':
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    if option == 'call':
        return np.exp(-q * T) * N(_d1)
    return np.exp(-q * T) * (N(_d1) - 1)


def gamma(S, K, T, r, sigma, q=0.0):
    """Same for call and put."""
    if T <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * n(_d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """Per 1 % move in volatility."""
    if T <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * n(_d1) * np.sqrt(T) / 100


def theta(S, K, T, r, sigma, q=0.0, option='call'):
    """Per calendar day."""
    if T <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)
    decay = -(S * np.exp(-q * T) * n(_d1) * sigma) / (2 * np.sqrt(T))
    if option == 'call':
        rate  = -r * K * np.exp(-r * T) * N(_d2)
        div   =  q * S * np.exp(-q * T) * N(_d1)
    else:
        rate  =  r * K * np.exp(-r * T) * N(-_d2)
        div   = -q * S * np.exp(-q * T) * N(-_d1)
    return (decay + rate + div) / 365


def rho(S, K, T, r, sigma, q=0.0, option='call'):
    """Per 1 % move in the risk-free rate."""
    if T <= 0:
        return 0.0
    _d2 = d2(S, K, T, r, sigma, q)
    if option == 'call':
        return  K * T * np.exp(-r * T) * N(_d2)  / 100
    return    -K * T * np.exp(-r * T) * N(-_d2) / 100


# ── Second-order Greeks ───────────────────────────────────────────────────────

def vanna(S, K, T, r, sigma, q=0.0):
    """dΔ/dσ = dVega/dS  (sensitivity of delta to vol and vice-versa)."""
    if T <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)
    return -np.exp(-q * T) * n(_d1) * _d2 / sigma


def volga(S, K, T, r, sigma, q=0.0):
    """dVega/dσ  (Vomma) — curvature of price with respect to vol."""
    if T <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)
    raw_vega = S * np.exp(-q * T) * n(_d1) * np.sqrt(T)
    return raw_vega * _d1 * _d2 / sigma


def charm(S, K, T, r, sigma, q=0.0, option='call'):
    """dΔ/dt  — rate of change of delta with respect to time (per day)."""
    if T <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    _d2 = _d1 - sigma * np.sqrt(T)
    core = n(_d1) * (2 * (r - q) * T - _d2 * sigma * np.sqrt(T)) / (
        2 * T * sigma * np.sqrt(T)
    )
    if option == 'call':
        return -np.exp(-q * T) * (core - q * N(_d1))  / 365
    return  -np.exp(-q * T) * (core + q * N(-_d1)) / 365


def speed(S, K, T, r, sigma, q=0.0):
    """dΓ/dS — rate of change of gamma with respect to spot."""
    if T <= 0:
        return 0.0
    _d1 = d1(S, K, T, r, sigma, q)
    G = gamma(S, K, T, r, sigma, q)
    return -G / S * (1 + _d1 / (sigma * np.sqrt(T)))


# ── Convenience class ─────────────────────────────────────────────────────────

class BSOption:
    """European Black-Scholes option with lazy-evaluated Greeks."""

    def __init__(self, S, K, T, r, sigma, q=0.0, option='call'):
        self.S     = float(S)
        self.K     = float(K)
        self.T     = float(T)
        self.r     = float(r)
        self.sigma = float(sigma)
        self.q     = float(q)
        self.option = option.lower()

    # ── Price & intrinsics ──

    @property
    def price(self):
        return call_price(self.S, self.K, self.T, self.r, self.sigma, self.q) \
               if self.option == 'call' \
               else put_price(self.S, self.K, self.T, self.r, self.sigma, self.q)

    @property
    def intrinsic_value(self):
        if self.option == 'call':
            return max(self.S - self.K, 0.0)
        return max(self.K - self.S, 0.0)

    @property
    def time_value(self):
        return self.price - self.intrinsic_value

    @property
    def moneyness(self):
        return self.S / self.K

    @property
    def log_moneyness(self):
        return np.log(self.S / self.K)

    # ── Greeks ──

    @property
    def delta(self):
        return delta(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option)

    @property
    def gamma(self):
        return gamma(self.S, self.K, self.T, self.r, self.sigma, self.q)

    @property
    def vega(self):
        return vega(self.S, self.K, self.T, self.r, self.sigma, self.q)

    @property
    def theta(self):
        return theta(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option)

    @property
    def rho(self):
        return rho(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option)

    @property
    def vanna(self):
        return vanna(self.S, self.K, self.T, self.r, self.sigma, self.q)

    @property
    def volga(self):
        return volga(self.S, self.K, self.T, self.r, self.sigma, self.q)

    @property
    def charm(self):
        return charm(self.S, self.K, self.T, self.r, self.sigma, self.q, self.option)

    @property
    def speed(self):
        return speed(self.S, self.K, self.T, self.r, self.sigma, self.q)

    # ── Summary ──

    def greeks_summary(self):
        return {
            'Price':    self.price,
            'Delta':    self.delta,
            'Gamma':    self.gamma,
            'Vega':     self.vega,
            'Theta':    self.theta,
            'Rho':      self.rho,
            'Vanna':    self.vanna,
            'Volga':    self.volga,
            'Charm':    self.charm,
            'Speed':    self.speed,
        }

    def __repr__(self):
        return (
            f"BSOption(S={self.S}, K={self.K}, T={self.T:.4f}, "
            f"r={self.r:.4f}, σ={self.sigma:.4f}, q={self.q:.4f}, "
            f"type={self.option})\n"
            f"  Price={self.price:.4f} | Δ={self.delta:.4f} | "
            f"Γ={self.gamma:.6f} | ν={self.vega:.4f} | "
            f"Θ={self.theta:.4f} | ρ={self.rho:.4f}"
        )
