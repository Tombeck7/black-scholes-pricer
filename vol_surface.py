"""
Volatility surface utilities: parametric models, smile / skew / surface plots,
and Greek surfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from bs_core import call_price, delta, gamma, vega, theta


# ── Parametric smile models ───────────────────────────────────────────────────

def quadratic_smile(log_moneyness, atm_vol, skew, convexity):
    """
    σ(k) = σ_atm + skew·k + convexity·k²
    k = log(K/F)
    """
    k = np.asarray(log_moneyness)
    return atm_vol + skew * k + convexity * k ** 2


def svi_total_variance(k, a, b, rho, m, sigma):
    """
    SVI (Gatheral 2004) total-variance parametrisation.
        w(k) = a + b · [ ρ·(k-m) + sqrt((k-m)² + σ²) ]

    Returns total variance w = σ_impl² · T.
    """
    k = np.asarray(k, dtype=float)
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def svi_vol(k, T, a, b, rho, m, sigma):
    """SVI implied volatility (not total var)."""
    w = svi_total_variance(k, a, b, rho, m, sigma)
    return np.sqrt(np.maximum(w / T, 0.0))


def heston_smile_approx(log_moneyness, v0, xi, rho_h, T):
    """
    Li (2005) first-order Heston approximation valid near ATM.

    σ(k) ≈ σ_0 + (ρ_H·ξ/2)·k/σ_0 + (ξ²/8·v0)·(1-ρ_H²)·k²
    """
    k     = np.asarray(log_moneyness)
    sig0  = np.sqrt(v0)
    term1 = sig0
    term2 = (rho_h * xi / 2) * k / sig0
    term3 = (xi ** 2 / (8 * v0)) * (1 - rho_h ** 2) * k ** 2
    return term1 + term2 + term3


def realistic_smile(strikes, S, T, atm_vol, skew_coef=-0.08, conv_coef=0.10):
    """
    Build a realistic equity-style smile: downward skew + convexity.
    Skew decays with sqrt(T) (as observed empirically).
    """
    strikes = np.asarray(strikes, dtype=float)
    k = np.log(strikes / S)
    sk = skew_coef / np.sqrt(max(T, 1e-4))
    cv = conv_coef  / max(T, 1e-4) ** 0.3
    return np.maximum(atm_vol + sk * k + cv * k ** 2, 0.02)


# ── Smile & skew plots ────────────────────────────────────────────────────────

def plot_vol_smile(strikes, iv_curves, S=None, labels=None,
                   title='Implied Volatility Smile', ax=None,
                   save_path=None):
    """
    Plot one or several IV smiles on the same axes.

    Parameters
    ----------
    iv_curves : single 1-D array  OR  list of 1-D arrays
    labels    : list of strings (one per curve)
    """
    show = ax is None and save_path is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    if isinstance(iv_curves, np.ndarray) and iv_curves.ndim == 1:
        iv_curves = [iv_curves]
    if labels is None:
        labels = [f'Curve {i+1}' for i in range(len(iv_curves))]

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(iv_curves)))
    for iv, lbl, col in zip(iv_curves, labels, colors):
        ax.plot(strikes, np.array(iv) * 100, marker='o', markersize=3,
                linewidth=1.8, label=lbl, color=col)

    if S is not None:
        ax.axvline(S, color='tomato', linestyle='--', linewidth=1.2,
                   alpha=0.8, label=f'Spot = {S}')
    ax.set_xlabel('Strike K')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    elif show:
        plt.tight_layout()
        plt.show()


def plot_vol_skew(log_moneyness, iv_curves, labels=None,
                  title='Implied Volatility Skew (log-moneyness)',
                  ax=None, save_path=None):
    """Plot IV vs log(K/S) to visualise the skew."""
    show = ax is None and save_path is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))

    if isinstance(iv_curves, np.ndarray) and iv_curves.ndim == 1:
        iv_curves = [iv_curves]
    if labels is None:
        labels = [f'Curve {i+1}' for i in range(len(iv_curves))]

    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(iv_curves)))
    for iv, lbl, col in zip(iv_curves, labels, colors):
        ax.plot(log_moneyness, np.array(iv) * 100, marker='o', markersize=3,
                linewidth=1.8, label=lbl, color=col)

    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.6,
               label='ATM (log-m = 0)')
    ax.set_xlabel('log(K/S)')
    ax.set_ylabel('Implied Volatility (%)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    elif show:
        plt.tight_layout()
        plt.show()


# ── 3-D surface ───────────────────────────────────────────────────────────────

def plot_vol_surface_3d(strikes, maturities, iv_grid, S=None,
                        title='Implied Volatility Surface',
                        save_path=None):
    """
    3-D surface plot of the IV surface.

    iv_grid : shape (len(maturities), len(strikes))
    """
    K_grid, T_grid = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(
        K_grid, T_grid, iv_grid * 100,
        cmap=cm.RdYlGn_r, alpha=0.88, edgecolor='none', linewidth=0
    )
    ax.set_xlabel('Strike K', labelpad=8)
    ax.set_ylabel('Maturity T (years)', labelpad=8)
    ax.set_zlabel('IV (%)', labelpad=8)
    ax.set_title(title, pad=14)

    if S is not None:
        z_top = iv_grid.max() * 100 * 1.02
        ax.plot([S, S], [maturities[0], maturities[-1]], [z_top, z_top],
                'r--', linewidth=2, label=f'Spot = {S}')
        ax.legend()

    fig.colorbar(surf, ax=ax, shrink=0.45, aspect=12, pad=0.08, label='IV (%)')

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_iv_heatmap(strikes, maturities, iv_grid, S=None,
                    title='IV Surface — Top-down Heatmap', save_path=None):
    """
    Top-down heatmap (contour fill) of the IV surface.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    cf = ax.contourf(strikes, maturities * 12, iv_grid * 100,
                     levels=25, cmap='RdYlGn_r')
    ax.contour(strikes, maturities * 12, iv_grid * 100,
               levels=12, colors='white', alpha=0.25, linewidths=0.6)
    cbar = fig.colorbar(cf, ax=ax, label='IV (%)')

    if S is not None:
        ax.axvline(S, color='red', linestyle='--', linewidth=1.5,
                   label=f'Spot = {S}')
        ax.legend()

    ax.set_xlabel('Strike K')
    ax.set_ylabel('Maturity (months)')
    ax.set_title(title)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# ── Term structure ────────────────────────────────────────────────────────────

def plot_atm_term_structure(maturities, atm_vols,
                            title='ATM Implied Vol Term Structure',
                            save_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.array(maturities) * 12, np.array(atm_vols) * 100,
            'o-', color='steelblue', linewidth=2, markersize=6)
    ax.set_xlabel('Maturity (months)')
    ax.set_ylabel('ATM IV (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


# ── Greek surfaces ────────────────────────────────────────────────────────────

def plot_delta_gamma_surface(S_range, sigma_range, K, T, r, q=0.0,
                              option='call', save_path=None):
    """Delta and Gamma as 3-D surfaces over (spot, vol) space."""
    S_g, sig_g = np.meshgrid(S_range, sigma_range)
    D_g = np.zeros_like(S_g)
    G_g = np.zeros_like(S_g)

    for i, sig in enumerate(sigma_range):
        for j, s in enumerate(S_range):
            D_g[i, j] = delta(s, K, T, r, sig, q, option)
            G_g[i, j] = gamma(s, K, T, r, sig, q)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={'projection': '3d'})
    fig.suptitle(f'Greek surfaces — K={K}, T={T}y, {option.upper()}', fontsize=12)

    for ax, data, label, cmap_name in zip(
            axes, [D_g, G_g], ['Delta (Δ)', 'Gamma (Γ)'], ['coolwarm', 'viridis']):
        surf = ax.plot_surface(S_g, sig_g * 100, data,
                               cmap=cmap_name, alpha=0.85, edgecolor='none')
        ax.set_xlabel('Spot S', labelpad=6)
        ax.set_ylabel('Vol σ (%)', labelpad=6)
        ax.set_zlabel(label, labelpad=6)
        ax.set_title(label)
        fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.08)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()


def plot_vega_theta_surface(S_range, T_range, K, r, sigma, q=0.0,
                             option='call', save_path=None):
    """Vega and Theta as 3-D surfaces over (spot, time-to-expiry) space."""
    S_g, T_g = np.meshgrid(S_range, T_range)
    V_g = np.zeros_like(S_g)
    TH_g = np.zeros_like(S_g)

    for i, t in enumerate(T_range):
        for j, s in enumerate(S_range):
            V_g[i, j]  = vega(s, K, t, r, sigma, q)
            TH_g[i, j] = theta(s, K, t, r, sigma, q, option)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             subplot_kw={'projection': '3d'})
    fig.suptitle(f'Vega & Theta surfaces — K={K}, σ={sigma*100:.0f}%, {option.upper()}',
                 fontsize=12)

    for ax, data, label, cmap_name in zip(
            axes, [V_g, TH_g],
            ['Vega (ν) / 1% vol', 'Theta (Θ) / day'],
            ['YlOrRd', 'PuRd']):
        surf = ax.plot_surface(S_g, T_g, data,
                               cmap=cmap_name, alpha=0.85, edgecolor='none')
        ax.set_xlabel('Spot S', labelpad=6)
        ax.set_ylabel('Time to Expiry (y)', labelpad=6)
        ax.set_zlabel(label, labelpad=6)
        ax.set_title(label)
        fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.08)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()
