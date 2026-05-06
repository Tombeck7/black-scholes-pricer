"""
PDF Term Sheet generator for the Black-Scholes Pricer.
Uses fpdf2 for layout and matplotlib (Agg) for embedded charts.
"""

import io
import numpy as np
from datetime import datetime
from fpdf import FPDF, XPos, YPos

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ?? Colour palette ????????????????????????????????????????????????????????????
NAVY   = (15,  23,  42)
INDIGO = (99, 102, 241)
GREEN  = (34, 197,  94)
RED    = (239,  68,  68)
AMBER  = (245, 158,  11)
LGRAY  = (241, 245, 249)
WHITE  = (255, 255, 255)
DARK   = ( 30,  41,  59)
MUTED  = (100, 116, 139)


# ?? Helpers ???????????????????????????????????????????????????????????????????

def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


def _section(pdf, title: str):
    pdf.set_fill_color(*INDIGO)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 7, f"  {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.ln(2)
    pdf.set_text_color(*DARK)


def _row(pdf, label: str, value: str, fill: bool = False):
    pdf.set_font("Helvetica", "", 9)
    pdf.set_fill_color(*LGRAY)
    pdf.cell(70, 6, f"  {label}", border=0, fill=fill)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, f"  {value}", border=0, fill=fill,
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _greek_table(pdf, greeks_c: dict, greeks_p: dict):
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(*INDIGO)
    pdf.set_text_color(*WHITE)
    for h, w in [("Greek", 35), ("Call", 40), ("Put", 40), ("Formula", 0)]:
        pdf.cell(w, 7, f"  {h}", border=0, fill=True)
    pdf.ln()
    pdf.set_text_color(*DARK)

    meta = [
        ("Delta",  greeks_c["Delta"],  greeks_p["Delta"],  "dV/dS"),
        ("Gamma",  greeks_c["Gamma"],  greeks_p["Gamma"],  "d2V/dS2"),
        ("Vega",   greeks_c["Vega"],   greeks_p["Vega"],   "dV/d_sigma per 1%"),
        ("Theta",  greeks_c["Theta"],  greeks_p["Theta"],  "dV/dt per day"),
        ("Rho",    greeks_c["Rho"],    greeks_p["Rho"],    "dV/dr per 1bp"),
        ("Vanna",  greeks_c["Vanna"],  greeks_p["Vanna"],  "dDelta/d_sigma"),
        ("Volga",  greeks_c["Volga"],  greeks_p["Volga"],  "dVega/d_sigma"),
    ]
    for i, (name, cv, pv, formula) in enumerate(meta):
        fill = i % 2 == 0
        pdf.set_fill_color(*LGRAY)
        pdf.set_font("Helvetica", "B", 8.5)
        pdf.cell(35, 6, f"  {name}", fill=fill)
        pdf.set_font("Helvetica", "", 8.5)
        pdf.cell(40, 6, f"  {cv:+.6f}", fill=fill)
        pdf.cell(40, 6, f"  {pv:+.6f}", fill=fill)
        pdf.cell(0,  6, f"  {formula}", fill=fill,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _chart_greeks(S, K, T, r, sigma, q) -> bytes:
    from bs_core import (call_price, put_price,
                          delta, gamma, vega, theta)
    S_range = np.linspace(max(S * 0.5, 1), S * 1.5, 300)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.patch.set_facecolor("#0f172a")

    data = [
        ("Price",   [call_price(s,K,T,r,sigma,q) for s in S_range],
                    [put_price(s,K,T,r,sigma,q)  for s in S_range],  True),
        ("Delta",   [delta(s,K,T,r,sigma,q,'call') for s in S_range],
                    [delta(s,K,T,r,sigma,q,'put')  for s in S_range], True),
        ("Gamma",   [gamma(s,K,T,r,sigma,q) for s in S_range], None, False),
        ("Vega",    [vega(s,K,T,r,sigma,q)  for s in S_range], None, False),
        ("Theta C", [theta(s,K,T,r,sigma,q,'call') for s in S_range],
                    [theta(s,K,T,r,sigma,q,'put')  for s in S_range], True),
    ]
    colors = ["#6366f1", "#ef4444"]
    for ax, (lbl, vc, vp, two) in zip(axes.flatten(), data):
        ax.set_facecolor("#1e293b")
        ax.plot(S_range, vc, color=colors[0], lw=1.8, label="Call")
        if two and vp is not None:
            ax.plot(S_range, vp, color=colors[1], lw=1.8, label="Put")
        ax.axvline(K, color="#f59e0b", lw=0.9, ls="--", alpha=0.7)
        ax.axhline(0, color="white",   lw=0.4, alpha=0.3)
        ax.set_title(lbl, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#334155")
        if two: ax.legend(fontsize=7, facecolor="#1e293b", labelcolor="white")

    axes.flatten()[-1].set_visible(False)
    plt.tight_layout()
    b = _fig_to_bytes(fig)
    plt.close(fig)
    return b


def _chart_payoff(S, K, T, r, sigma, q) -> bytes:
    from bs_core import call_price, put_price
    from payoff_analysis import long_call, long_put, long_straddle

    S_range = np.linspace(S * 0.6, S * 1.4, 300)
    c0 = call_price(S, K, T, r, sigma, q)
    p0 = put_price(S,  K, T, r, sigma, q)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0f172a")

    for ax, (pf, lbl, col) in zip(axes, [
        (long_call(S_range, K, c0),            "Long Call",    "#6366f1"),
        (long_put(S_range,  K, p0),            "Long Put",     "#ef4444"),
        (long_straddle(S_range, K, c0, p0),    "Long Straddle","#22c55e"),
    ]):
        ax.set_facecolor("#1e293b")
        ax.plot(S_range, pf, color=col, lw=2)
        ax.fill_between(S_range, 0, np.maximum(pf, 0), alpha=0.15, color="limegreen")
        ax.fill_between(S_range, np.minimum(pf, 0), 0, alpha=0.15, color="red")
        ax.axhline(0, color="white", lw=0.5, alpha=0.4)
        ax.axvline(S, color="#f59e0b", lw=0.9, ls="--", alpha=0.7)
        ax.set_title(lbl, color="white", fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ax.spines.values(): sp.set_color("#334155")

    plt.tight_layout()
    b = _fig_to_bytes(fig)
    plt.close(fig)
    return b


# ?? Main export function ??????????????????????????????????????????????????????

def generate_bs_pdf(S, K, T, r, sigma, q,
                     greeks_call: dict, greeks_put: dict,
                     ticker: str = "") -> bytes:
    """
    Generate a professional BS option term sheet PDF.
    Returns raw PDF bytes for st.download_button.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ?? Header ????????????????????????????????????????????????????????????????
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, 210, 28, "F")
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_y(6)
    pdf.cell(0, 8, "BLACK-SCHOLES OPTION TERM SHEET",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.set_font("Helvetica", "", 9)
    ticker_str = f" | {ticker}" if ticker else ""
    pdf.cell(0, 6,
             f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}{ticker_str}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(8)
    pdf.set_text_color(*DARK)

    # ?? Market Parameters ?????????????????????????????????????????????????????
    _section(pdf, "MARKET PARAMETERS")
    for i, (lbl, val) in enumerate([
        ("Spot  S",       f"{S:.4f}"),
        ("Strike  K",     f"{K:.4f}"),
        ("Maturity  T",   f"{T:.4f} years"),
        ("Risk-free  r",  f"{r*100:.2f}%"),
        ("Volatility  sigma", f"{sigma*100:.2f}%"),
        ("Dividend  q",   f"{q*100:.2f}%"),
    ]):
        _row(pdf, lbl, val, fill=i % 2 == 0)
    pdf.ln(4)

    # ?? Pricing Results ???????????????????????????????????????????????????????
    _section(pdf, "PRICING RESULTS")
    c_price = greeks_call["Price"]
    p_price = greeks_put["Price"]
    fwd     = S * np.exp((r - q) * T)
    pv_k    = K * np.exp(-r * T)
    for i, (lbl, val) in enumerate([
        ("Call Price",        f"{c_price:.6f}"),
        ("Put Price",         f"{p_price:.6f}"),
        ("C - P  (PCP check)",f"{c_price - p_price:.6f}  (theory: {fwd - pv_k:.6f})"),
        ("Forward  F",        f"{fwd:.4f}"),
        ("PV(K)",             f"{pv_k:.4f}"),
        ("Call Moneyness",    f"{S/K:.4f}  ({'ITM' if S>K else 'OTM' if S<K else 'ATM'})"),
        ("Call Intrinsic",    f"{max(S-K, 0):.4f}"),
        ("Call Time Value",   f"{c_price - max(S-K, 0):.4f}"),
    ]):
        _row(pdf, lbl, val, fill=i % 2 == 0)
    pdf.ln(4)

    # ?? Greeks Table ??????????????????????????????????????????????????????????
    _section(pdf, "ANALYTICAL GREEKS")
    _greek_table(pdf, greeks_call, greeks_put)
    pdf.ln(4)

    # ?? Greeks Chart ?????????????????????????????????????????????????????????
    pdf.add_page()
    _section(pdf, "GREEKS vs SPOT")
    chart_g = _chart_greeks(S, K, T, r, sigma, q)
    pdf.image(io.BytesIO(chart_g), x=10, w=190)
    pdf.ln(4)

    # ?? Payoff Chart ??????????????????????????????????????????????????????????
    _section(pdf, "PAYOFF DIAGRAMS AT EXPIRY")
    chart_p = _chart_payoff(S, K, T, r, sigma, q)
    pdf.image(io.BytesIO(chart_p), x=10, w=190)

    # ?? Footer ????????????????????????????????????????????????????????????????
    pdf.set_y(-15)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(*MUTED)
    pdf.cell(0, 5,
             "Black-Scholes Pricer | For informational purposes only -- not financial advice.",
             align="C")

    return bytes(pdf.output())
