"""
AMZN / MSFT Relative Value Dashboard
GJR-GARCH → DCC → Copula Pipeline | Bhawini Singh
"""

from __future__ import annotations

import warnings
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from arch import arch_model
from scipy import optimize, stats
from scipy.special import gammaln

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
# TYPES & CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CopulaFamily = Literal["gaussian", "student_t", "clayton", "gumbel", "frank", "bb1"]
COPULA_FAMILIES = ["gaussian", "student_t", "clayton", "gumbel", "frank", "bb1"]

DARK = "#0d1117"
CARD = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
BLUE = "#58a6ff"
GREEN = "#3fb950"
RED = "#f85149"
PURPLE = "#d2a8ff"
ORANGE = "#ffa657"
TEAL = "#39d353"

st.set_page_config(
    page_title="AMZN / MSFT Relative Value",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  html, body, [data-testid="stAppViewContainer"] {{
      background-color: {DARK};
      color: {TEXT};
  }}
  [data-testid="stSidebar"] {{
      background-color: {CARD};
      border-right: 1px solid {BORDER};
  }}
  [data-testid="metric-container"] {{
      background-color: {CARD};
      border: 1px solid {BORDER};
      border-radius: 8px;
      padding: 12px 16px;
  }}
  .section-header {{
      font-family: 'Space Mono', monospace;
      font-size: 0.75rem;
      color: {BLUE};
      letter-spacing: 0.1em;
      text-transform: uppercase;
      padding: 8px 0 4px 0;
      border-bottom: 1px solid {BORDER};
      margin-bottom: 12px;
  }}
  div[data-testid="stMetricValue"] {{ color: {TEXT}; font-size: 1.25rem; }}
  div[data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: 0.75rem; }}
  div[data-testid="stMetricDelta"] {{ font-size: 0.75rem; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# DATACLASSES  (mirrors the notebook exactly)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GARCHResult:
    ticker: str
    params: dict
    nu: float
    eta: float
    conditional_vol: pd.Series
    standardized_residuals: pd.Series
    log_returns: pd.Series

@dataclass
class DCCResult:
    rho: pd.Series
    H: np.ndarray
    hedge_ratio: pd.Series
    dcc_params: dict
    Q_bar: np.ndarray

@dataclass
class CopulaFitResult:
    family: CopulaFamily
    params: dict
    log_likelihood: float
    n_params: int
    aic: float

# ──────────────────────────────────────────────────────────────────────────────
# MODEL FUNCTIONS  (exact copies from notebook)
# ──────────────────────────────────────────────────────────────────────────────

def fit_gjr_garch(log_returns: pd.Series, ticker: str) -> GARCHResult:
    model = arch_model(
        log_returns * 100,
        mean="Constant", vol="GARCH", p=1, o=1, q=1,
        dist="skewstudent",
    )
    result = model.fit(disp="off", show_warning=False)
    sigma = result.conditional_volatility / 100
    resid = result.resid / 100
    z = resid / sigma
    params = {
        "mu":    result.params["mu"] / 100,
        "omega": result.params["omega"],
        "alpha": result.params["alpha[1]"],
        "gamma": result.params["gamma[1]"],
        "beta":  result.params["beta[1]"],
    }
    nu  = result.params["eta"]
    eta = result.params["lambda"]
    return GARCHResult(
        ticker=ticker, params=params, nu=nu, eta=eta,
        conditional_vol=pd.Series(sigma, index=log_returns.index, name=f"sigma_{ticker}"),
        standardized_residuals=pd.Series(z, index=log_returns.index, name=f"z_{ticker}"),
        log_returns=log_returns,
    )


def _dcc_loglik(params, Z, Q_bar):
    a, b = params
    if a <= 0 or b <= 0 or a + b >= 1:
        return 1e10
    T = Z.shape[0]
    Q = Q_bar.copy()
    ll = 0.0
    for t in range(1, T):
        z_lag = Z[t - 1].reshape(-1, 1)
        Q = (1 - a - b) * Q_bar + a * (z_lag @ z_lag.T) + b * Q
        d_inv = np.diag(1.0 / np.sqrt(np.diag(Q)))
        R = d_inv @ Q @ d_inv
        R = np.clip(R, -0.9999, 0.9999)
        np.fill_diagonal(R, 1.0)
        sign, log_det = np.linalg.slogdet(R)
        if sign <= 0:
            return 1e10
        z_t = Z[t]
        ll += -0.5 * (log_det + z_t @ np.linalg.solve(R, z_t) - z_t @ z_t)
    return -ll


def fit_dcc(garch_a: GARCHResult, garch_b: GARCHResult) -> DCCResult:
    Z = np.column_stack([garch_a.standardized_residuals.values,
                         garch_b.standardized_residuals.values])
    Q_bar = np.cov(Z.T)
    result = optimize.minimize(
        _dcc_loglik, x0=[0.02, 0.95], args=(Z, Q_bar),
        method="L-BFGS-B",
        bounds=[(1e-6, 0.5), (1e-6, 0.9999)],
        options={"ftol": 1e-9, "maxiter": 500},
    )
    a, b = result.x
    T = Z.shape[0]
    rho_vals = np.zeros(T)
    H_vals = np.zeros((T, 2, 2))
    Q = Q_bar.copy()
    sigma_a = garch_a.conditional_vol.values
    sigma_b = garch_b.conditional_vol.values
    for t in range(T):
        if t > 0:
            z_lag = Z[t - 1].reshape(-1, 1)
            Q = (1 - a - b) * Q_bar + a * (z_lag @ z_lag.T) + b * Q
        d_inv = np.diag(1.0 / np.sqrt(np.diag(Q)))
        R = d_inv @ Q @ d_inv
        R = np.clip(R, -0.9999, 0.9999)
        np.fill_diagonal(R, 1.0)
        rho_vals[t] = R[0, 1]
        D = np.diag([sigma_a[t], sigma_b[t]])
        H_vals[t] = D @ R @ D
    index = garch_a.conditional_vol.index
    rho = pd.Series(rho_vals, index=index, name="rho")
    hedge_ratio = rho * (garch_a.conditional_vol / garch_b.conditional_vol)
    hedge_ratio.name = "hedge_ratio"
    return DCCResult(rho=rho, H=H_vals, hedge_ratio=hedge_ratio,
                     dcc_params={"a": a, "b": b}, Q_bar=Q_bar)


def pit_transform(garch: GARCHResult) -> pd.Series:
    """PIT using SkewStudent CDF — matches notebook exactly."""
    from arch.univariate.distribution import SkewStudent
    dist = SkewStudent()
    u = dist.cdf(
        garch.standardized_residuals.values,
        parameters=np.array([garch.nu, garch.eta]),
    )
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return pd.Series(u, index=garch.standardized_residuals.index, name=f"u_{garch.ticker}")


def validate_pit_uniformity(u: pd.Series, alpha: float = 0.05):
    stat, p_value = stats.kstest(u.values, "uniform")
    passed = p_value >= alpha
    return passed, stat, p_value


# ── Copula log-likelihoods ────────────────────────────────────────────────────

def _gaussian_copula_loglik(rho, u, v):
    if abs(rho) >= 1: return -np.inf
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    ll = (-0.5 * np.log(1 - rho**2)
          - (rho**2 * (x**2 + y**2) - 2 * rho * x * y) / (2 * (1 - rho**2)))
    return np.sum(ll)

def _student_t_copula_loglik(params, u, v):
    rho, nu = params
    if abs(rho) >= 1 or nu <= 2: return -np.inf
    x, y = stats.t.ppf(u, df=nu), stats.t.ppf(v, df=nu)
    det = 1 - rho**2
    A = (x**2 + y**2 - 2 * rho * x * y) / det
    ll = (gammaln((nu + 2) / 2) + gammaln(nu / 2)
          - 2 * gammaln((nu + 1) / 2)
          - 0.5 * np.log(det)
          + (nu + 1) / 2 * (np.log(1 + x**2 / nu) + np.log(1 + y**2 / nu))
          - (nu + 2) / 2 * np.log(1 + A / nu))
    return np.sum(ll)

def _clayton_loglik(theta, u, v):
    if theta <= 0: return -np.inf
    ll = (np.log(theta + 1)
          + (-theta - 1) * (np.log(u) + np.log(v))
          + (-1 / theta - 2) * np.log(u**(-theta) + v**(-theta) - 1))
    return np.sum(ll)

def _gumbel_loglik(theta, u, v):
    if theta < 1: return -np.inf
    lu, lv = -np.log(u), -np.log(v)
    S = lu**theta + lv**theta
    C = np.exp(-(S ** (1 / theta)))
    ll = (np.log(C)
          + (1 / theta - 2) * np.log(S)
          + (theta - 1) * (np.log(lu) + np.log(lv))
          - np.log(u) - np.log(v)
          + np.log(S ** (1 / theta - 1) + (theta - 1) * S ** (1 / theta - 2)))
    return np.sum(ll)

def _frank_loglik(theta, u, v):
    if theta == 0: return -np.inf
    et = np.exp(-theta)
    etu = np.exp(-theta * u)
    etv = np.exp(-theta * v)
    denom = (1 - et) - (1 - etu) * (1 - etv)
    if np.any(denom == 0): return -np.inf
    ll = np.log(theta) + np.log(1 - et) - theta - 2 * np.log(np.abs(denom))
    return np.sum(ll)

def _bb1_loglik(params, u, v):
    theta, delta = params
    if theta <= 0 or delta < 1: return -np.inf
    try:
        x = (u**(-theta) - 1) ** delta
        y = (v**(-theta) - 1) ** delta
        S = x + y
        ll = (np.log(theta * delta + 1)
              + (theta + 1) / theta * (np.log(u) + np.log(v))
              + (delta - 1) * (np.log(x / u) + np.log(y / v))
              + (1 / delta - 2) * np.log(S)
              - S ** (1 / delta))
        return np.sum(ll)
    except Exception:
        return -np.inf

def _aic(log_lik, k):
    return 2 * k - 2 * log_lik

def fit_copulas(u_a: np.ndarray, u_b: np.ndarray) -> list[CopulaFitResult]:
    results = []
    # Gaussian
    opt = optimize.minimize_scalar(
        lambda r: -_gaussian_copula_loglik(r, u_a, u_b),
        bounds=(-0.999, 0.999), method="bounded")
    ll = _gaussian_copula_loglik(opt.x, u_a, u_b)
    results.append(CopulaFitResult("gaussian", {"rho": opt.x}, ll, 1, _aic(ll, 1)))
    # Student-t
    opt = optimize.minimize(
        lambda p: -_student_t_copula_loglik(p, u_a, u_b),
        x0=[0.5, 5.0], bounds=[(-0.999, 0.999), (2.01, 50.0)], method="L-BFGS-B")
    ll = _student_t_copula_loglik(opt.x, u_a, u_b)
    results.append(CopulaFitResult("student_t", {"rho": opt.x[0], "nu": opt.x[1]}, ll, 2, _aic(ll, 2)))
    # Clayton
    opt = optimize.minimize_scalar(
        lambda t: -_clayton_loglik(t, u_a, u_b),
        bounds=(1e-4, 20.0), method="bounded")
    ll = _clayton_loglik(opt.x, u_a, u_b)
    results.append(CopulaFitResult("clayton", {"theta": opt.x}, ll, 1, _aic(ll, 1)))
    # Gumbel
    opt = optimize.minimize_scalar(
        lambda t: -_gumbel_loglik(t, u_a, u_b),
        bounds=(1.0, 20.0), method="bounded")
    ll = _gumbel_loglik(opt.x, u_a, u_b)
    results.append(CopulaFitResult("gumbel", {"theta": opt.x}, ll, 1, _aic(ll, 1)))
    # Frank
    try:
        opt = optimize.minimize_scalar(
            lambda t: -_frank_loglik(t, u_a, u_b),
            bounds=(-20.0, 20.0), method="bounded")
        ll = _frank_loglik(opt.x, u_a, u_b)
        aic_val = _aic(ll, 1) if np.isfinite(ll) else np.nan
        results.append(CopulaFitResult("frank", {"theta": opt.x}, ll, 1, aic_val))
    except Exception:
        results.append(CopulaFitResult("frank", {"theta": 0.0}, -np.inf, 1, np.nan))
    # BB1
    try:
        opt = optimize.minimize(
            lambda p: -_bb1_loglik(p, u_a, u_b),
            x0=[0.5, 1.5], bounds=[(1e-4, 10.0), (1.0, 10.0)], method="L-BFGS-B")
        ll = _bb1_loglik(opt.x, u_a, u_b)
        aic_val = _aic(ll, 2) if np.isfinite(ll) else np.nan
        results.append(CopulaFitResult("bb1", {"theta": opt.x[0], "delta": opt.x[1]}, ll, 2, aic_val))
    except Exception:
        results.append(CopulaFitResult("bb1", {"theta": 0.5, "delta": 1.5}, -np.inf, 2, np.nan))
    return sorted([r for r in results if np.isfinite(r.aic)],
                  key=lambda r: r.aic) + [r for r in results if not np.isfinite(r.aic)]


def select_copula(fits: list[CopulaFitResult], tie_threshold: float = 2.0) -> CopulaFitResult:
    valid = [f for f in fits if np.isfinite(f.aic)]
    if not valid:
        return fits[0]
    best = valid[0]
    t_fit = next((f for f in valid if f.family == "student_t"), None)
    if t_fit is not None and (t_fit.aic - best.aic) < tie_threshold:
        return t_fit
    return best


def compute_copula_signal(u_a, u_b, copula: CopulaFitResult):
    u_a = np.clip(np.asarray(u_a, dtype=float), 1e-6, 1 - 1e-6)
    u_b = np.clip(np.asarray(u_b, dtype=float), 1e-6, 1 - 1e-6)
    if copula.family == "gaussian":
        rho = copula.params["rho"]
        x_a, x_b = stats.norm.ppf(u_a), stats.norm.ppf(u_b)
        return stats.norm.cdf((x_b - rho * x_a) / np.sqrt(1 - rho**2))
    elif copula.family == "student_t":
        rho, nu = copula.params["rho"], copula.params["nu"]
        t_a, t_b = stats.t.ppf(u_a, df=nu), stats.t.ppf(u_b, df=nu)
        num = t_b - rho * t_a
        den = np.sqrt((1 - rho**2) * (nu + t_a**2) / (nu + 1))
        return stats.t.cdf(num / den, df=nu + 1)
    elif copula.family == "clayton":
        theta = copula.params["theta"]
        return (u_a ** (-(theta + 1))
                * (u_a ** (-theta) + u_b ** (-theta) - 1) ** (-(1 + 1 / theta)))
    elif copula.family == "gumbel":
        theta = copula.params["theta"]
        lu_a, lu_b = -np.log(u_a), -np.log(u_b)
        S = lu_a**theta + lu_b**theta
        C = np.exp(-(S ** (1 / theta)))
        return C / u_a * (S ** (1 / theta - 1)) * lu_a ** (theta - 1)
    elif copula.family == "frank":
        theta = copula.params["theta"]
        et = np.exp(-theta)
        etu = np.exp(-theta * u_a)
        etv = np.exp(-theta * u_b)
        return etu * (1 - etv) / ((1 - et) - (1 - etu) * (1 - etv))
    elif copula.family == "bb1":
        theta, delta = copula.params["theta"], copula.params["delta"]
        x = (u_a**(-theta) - 1) ** delta
        y = (u_b**(-theta) - 1) ** delta
        S = x + y
        C = np.exp(-S ** (1 / delta))
        dC_du_a = (C * S ** (1 / delta - 1) * delta
                   * x / u_a * (u_a**(-theta) - 1) ** (delta - 1) * theta * u_a**(-theta - 1))
        return np.clip(dC_du_a, 1e-6, 1 - 1e-6)
    else:
        return np.full_like(u_a, 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING & CACHING
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_prices():
    p = yf.download(["AMZN", "MSFT"], period="6y",
                    auto_adjust=True, progress=False)["Close"]
    if isinstance(p.columns, pd.MultiIndex):
        p = p.droplevel(0, axis=1)
    return p[["AMZN", "MSFT"]].dropna()


@st.cache_resource(show_spinner=False)
def load_model():
    prices = load_prices()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    garch_amzn = fit_gjr_garch(log_ret["AMZN"], "AMZN")
    garch_msft = fit_gjr_garch(log_ret["MSFT"], "MSFT")
    dcc = fit_dcc(garch_amzn, garch_msft)

    u_amzn = pit_transform(garch_amzn)
    u_msft = pit_transform(garch_msft)

    pit_amzn_valid, ks_stat_a, ks_p_a = validate_pit_uniformity(u_amzn)
    pit_msft_valid, ks_stat_b, ks_p_b = validate_pit_uniformity(u_msft)

    copula_fits = fit_copulas(u_amzn.values, u_msft.values)
    selected = select_copula(copula_fits)

    signal = pd.Series(
        compute_copula_signal(u_amzn.values, u_msft.values, selected),
        index=u_amzn.index, name="signal",
    )

    # Beta-adjusted spread Z-score
    spread_raw = np.log(prices["AMZN"] / prices["MSFT"]).dropna()
    hedge = dcc.hedge_ratio.reindex(spread_raw.index).ffill()
    spread_adj = np.log(prices["AMZN"]) - hedge * np.log(prices["MSFT"])
    spread_adj = spread_adj.dropna()
    roll = spread_adj.rolling(60)
    z_score = (spread_adj - roll.mean()) / roll.std()

    return {
        "garch_amzn": garch_amzn,
        "garch_msft": garch_msft,
        "dcc": dcc,
        "u_amzn": u_amzn,
        "u_msft": u_msft,
        "pit_amzn_valid": pit_amzn_valid,
        "pit_msft_valid": pit_msft_valid,
        "ks_stat_a": ks_stat_a, "ks_p_a": ks_p_a,
        "ks_stat_b": ks_stat_b, "ks_p_b": ks_p_b,
        "copula_fits": copula_fits,
        "selected": selected,
        "signal": signal,
        "z_score": z_score,
        "prices": prices,
        "log_ret": np.log(prices / prices.shift(1)).dropna(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY LAYOUT DEFAULTS
# ──────────────────────────────────────────────────────────────────────────────

def _base_layout(**kwargs):
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=DARK,
        font=dict(color=TEXT, family="Space Mono, monospace", size=11),
        margin=dict(l=40, r=20, t=36, b=36),
        xaxis=dict(gridcolor=BORDER, zeroline=False, showgrid=True),
        yaxis=dict(gridcolor=BORDER, zeroline=False, showgrid=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER, borderwidth=1),
        **kwargs,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:1.1rem;color:{BLUE};font-weight:700;'>AMZN / MSFT<br>Relative Value</div>", unsafe_allow_html=True)
    st.caption("GJR-GARCH + DCC + Copula Pipeline")
    st.markdown(f"<small style='color:{MUTED}'>Beta-Neutral · DCC · {datetime.today().strftime('%Y-%m-%d')}</small>", unsafe_allow_html=True)
    st.divider()

    nav_input = st.number_input("Portfolio NAV ($)", min_value=100_000, max_value=10_000_000,
                                 value=200_000, step=50_000)
    entry_thresh = st.slider("Entry threshold", 0.05, 0.20, 0.10, 0.01)
    corr_floor   = st.slider("Min correlation filter", 0.20, 0.70, 0.40, 0.05)
    time_stop    = st.number_input("Time stop (days)", 5, 30, 15)
    risk_budget  = st.slider("Risk budget (% NAV)", 0.01, 0.10, 0.05, 0.01)
    max_leg      = st.slider("Max leg size (% NAV)", 0.10, 0.40, 0.20, 0.05)
    st.divider()
    st.caption("Data: Yahoo Finance · Copula: 6-family AIC selection · Recalib: weekly (Fridays)")


# ──────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────────────────────

st.markdown(
    f"<h2 style='font-family:Space Mono,monospace;font-size:1.4rem;color:{TEXT};margin-bottom:4px;'>"
    "AMZN / MSFT Relative Value</h2>"
    f"<p style='color:{MUTED};font-size:0.8rem;margin-top:0;'>"
    f"GJR-GARCH + DCC + Copula Pipeline &nbsp;|&nbsp; Beta-Neutral Finance &nbsp;|&nbsp; "
    f"Last updated: {datetime.today().strftime('%Y-%m-%d %H:%M:%S')}</p>",
    unsafe_allow_html=True,
)

with st.spinner("Fitting GJR-GARCH → DCC → Copula pipeline…"):
    try:
        m = load_model()
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.stop()

ga   = m["garch_amzn"]
gm   = m["garch_msft"]
dcc  = m["dcc"]
sel  = m["selected"]
sig  = m["signal"]
zs   = m["z_score"]
fits = m["copula_fits"]
prices = m["prices"]

rho_latest    = float(dcc.rho.iloc[-1])
sig_latest    = float(sig.iloc[-1])
hedge_latest  = float(dcc.hedge_ratio.iloc[-1])
vol_amzn      = float(ga.conditional_vol.iloc[-1]) * np.sqrt(252)
vol_msft      = float(gm.conditional_vol.iloc[-1]) * np.sqrt(252)

# Position sizing
sigma_spread = np.sqrt(
    vol_amzn**2 + hedge_latest**2 * vol_msft**2
    - 2 * hedge_latest * rho_latest * vol_amzn * vol_msft
)
notional_long  = (risk_budget * nav_input) / sigma_spread if sigma_spread > 0 else 0
notional_short = hedge_latest * notional_long
notional_long  = min(notional_long,  max_leg * nav_input)
notional_short = min(notional_short, max_leg * nav_input)

# Trade direction
if sig_latest < entry_thresh and sig_latest == sig_latest:
    direction_str = "Long AMZN / Short MSFT"
    dir_color = RED
elif sig_latest > (1 - entry_thresh):
    direction_str = "Long MSFT / Short AMZN"
    dir_color = GREEN
else:
    direction_str = "No signal"
    dir_color = MUTED

# ── TOP METRICS ROW ──────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Copula Signal h_t", f"{sig_latest:.4f}",
          delta="Long AMZN" if sig_latest < entry_thresh else ("Long MSFT" if sig_latest > 1 - entry_thresh else "Neutral"))
c2.metric("DCC ρ_t", f"{rho_latest:.4f}",
          delta="Above floor ✓" if rho_latest > corr_floor else "Below floor ✗")
c3.metric("Hedge Ratio β_t", f"{hedge_latest:.4f}",
          delta="Beta-neutral")
c4.metric("AMZN σ (ann.)", f"{vol_amzn:.1%}",
          delta=f"GJR-GARCH α={ga.params['alpha']:.3f}")
c5.metric("MSFT σ (ann.)", f"{vol_msft:.1%}",
          delta=f"GJR-GARCH α={gm.params['alpha']:.3f}")
c6.metric("Active Copula", sel.family.replace("_", "-").title(),
          delta=f"AIC {sel.aic:.1f}")

st.markdown("<br>", unsafe_allow_html=True)

# ── SECTION 01: DCC CORRELATION & COPULA SIGNAL ──────────────────────────────
st.markdown("<div class='section-header'>01 — DCC CORRELATION & COPULA SIGNAL</div>", unsafe_allow_html=True)

col_dcc, col_sig = st.columns(2)

with col_dcc:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dcc.rho.index, y=dcc.rho.values,
        fill="tozeroy", fillcolor=f"rgba(88,166,255,0.1)",
        line=dict(color=BLUE, width=1.5),
        name="DCC ρ_t",
    ))
    fig.add_hline(y=corr_floor, line_dash="dash", line_color=RED,
                  annotation_text=f"Min corr floor ({corr_floor})",
                  annotation_font_color=RED)
    fig.update_layout(
        title=dict(text="DCC Time-Varying Correlation (ρ_t)", font=dict(color=TEXT, size=12)),
        yaxis_title="ρ_t", yaxis=dict(range=[0, 1], gridcolor=BORDER),
        **_base_layout(height=280),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_sig:
    colors_sig = [GREEN if s > (1 - entry_thresh) else RED if s < entry_thresh else PURPLE
                  for s in sig.values]
    fig = go.Figure()
    fig.add_hrect(y0=(1 - entry_thresh), y1=1.0, fillcolor=f"rgba(63,185,80,0.08)", line_width=0)
    fig.add_hrect(y0=0.0, y1=entry_thresh, fillcolor=f"rgba(241,81,73,0.08)", line_width=0)
    fig.add_trace(go.Bar(x=sig.index, y=sig.values,
                         marker_color=colors_sig, name="Signal h_t"))
    fig.add_hline(y=0.5,               line_dash="dot",  line_color=MUTED, line_width=0.8)
    fig.add_hline(y=entry_thresh,      line_dash="dash", line_color=RED,   line_width=1)
    fig.add_hline(y=1 - entry_thresh,  line_dash="dash", line_color=GREEN, line_width=1)
    fig.update_layout(
        title=dict(text=f"Copula Signal — h-Function  (entry: <{entry_thresh} or >{1-entry_thresh})",
                   font=dict(color=TEXT, size=12)),
        yaxis_title="h_t", yaxis=dict(range=[0, 1], gridcolor=BORDER),
        **_base_layout(height=280),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 02: COPULA DENSITY SURFACE & REGIME INDICATOR ────────────────────
st.markdown("<div class='section-header'>02 — COPULA DENSITY SURFACE & REGIME INDICATOR</div>", unsafe_allow_html=True)

col_surf, col_regime = st.columns([3, 2])

with col_surf:
    g = np.linspace(0.02, 0.98, 40)
    gg_u, gg_v = np.meshgrid(g, g)
    u_flat = gg_u.ravel()
    v_flat = gg_v.ravel()
    try:
        dens_flat = compute_copula_signal(u_flat, v_flat, sel)
        dens = dens_flat.reshape(gg_u.shape)
        dens = np.clip(dens, 0, np.percentile(dens, 97))  # trim extreme spikes
    except Exception:
        dens = np.ones_like(gg_u)

    fig = go.Figure(data=[go.Surface(
        x=g, y=g, z=dens,
        colorscale="Plasma", opacity=0.9,
        colorbar=dict(
            title=dict(text="c(u,v)", font=dict(color=MUTED)),
            tickfont=dict(color=MUTED), len=0.6,
        ),
    )])
    fig.add_trace(go.Scatter3d(
        x=[float(m["u_amzn"].iloc[-1])],
        y=[float(m["u_msft"].iloc[-1])],
        z=[float(sig_latest)],
        mode="markers",
        marker=dict(size=8, color=RED, symbol="circle"),
        name="Current",
    ))
    fig.update_layout(
        title=dict(text=f"Copula Density Surface — {sel.family.replace('_','-').title()}",
                   font=dict(color=TEXT, size=12)),
        scene=dict(
            xaxis=dict(title="u_AMZN", backgroundcolor=DARK, gridcolor=BORDER, color=TEXT),
            yaxis=dict(title="u_MSFT", backgroundcolor=DARK, gridcolor=BORDER, color=TEXT),
            zaxis=dict(title="h(u,v)", backgroundcolor=DARK, gridcolor=BORDER, color=TEXT),
            bgcolor=DARK,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        height=380,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_regime:
    st.markdown(f"<div style='background:{CARD};border:1px solid {BORDER};border-radius:8px;padding:16px;'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:{BLUE};letter-spacing:0.1em;margin-bottom:12px;'>REGIME INDICATOR</div>", unsafe_allow_html=True)

    pit_a_label = ("✅ Uniform" if m["pit_amzn_valid"] else "⚠️ Non-uniform (expected)")
    pit_b_label = ("✅ Uniform" if m["pit_msft_valid"] else "⚠️ Non-uniform (expected)")

    regime_data = [
        ("Active Copula",   sel.family.replace("_","-").title()),
        ("Parameters",      ", ".join(f"{k}={v:.4f}" for k, v in sel.params.items())),
        ("Log-Likelihood",  f"{sel.log_likelihood:.2f}"),
        ("AIC",             f"{sel.aic:.2f}"),
        ("PIT KS — AMZN",  pit_a_label),
        ("PIT KS — MSFT",  pit_b_label),
        ("Latest ρ_t",      f"{rho_latest:.4f}"),
        ("Latest h_t",      f"{sig_latest:.4f}"),
    ]
    for label, val in regime_data:
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;padding:4px 0;"
            f"border-bottom:1px solid {BORDER};font-size:0.8rem;'>"
            f"<span style='color:{MUTED};'>{label}</span>"
            f"<span style='color:{TEXT};'>{val}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown(f"<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:{BLUE};letter-spacing:0.1em;margin:14px 0 8px 0;'>COPULA AIC RANKINGS</div>", unsafe_allow_html=True)

    rows_html = ""
    for f in fits:
        is_sel = (f.family == sel.family)
        aic_str = f"{f.aic:.1f}" if np.isfinite(f.aic) else "nan"
        rank_str = f"<span style='color:{GREEN};'>★ selected</span>" if is_sel else ""
        rows_html += (
            f"<tr>"
            f"<td style='padding:3px 6px;color:{TEXT if is_sel else MUTED};'>{f.family.replace('_','-').title()}</td>"
            f"<td style='padding:3px 6px;color:{TEXT};text-align:right;'>{aic_str}</td>"
            f"<td style='padding:3px 6px;text-align:right;'>{rank_str}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table style='width:100%;font-size:0.78rem;border-collapse:collapse;'>"
        f"<tr><th style='color:{MUTED};text-align:left;padding:3px 6px;'>Family</th>"
        f"<th style='color:{MUTED};text-align:right;padding:3px 6px;'>AIC</th>"
        f"<th style='color:{MUTED};text-align:right;padding:3px 6px;'>Rank</th></tr>"
        f"{rows_html}</table>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ── SECTION 03: SPREAD Z-SCORE & SIGNAL DISTRIBUTION ────────────────────────
st.markdown("<div class='section-header'>03 — SPREAD Z-SCORE & SIGNAL DISTRIBUTION</div>", unsafe_allow_html=True)

col_z, col_dist = st.columns(2)

with col_z:
    z_latest = float(zs.iloc[-1]) if not zs.empty else 0.0
    fig = go.Figure()
    fig.add_hrect(y0=-1, y1=1, fillcolor=f"rgba(139,148,158,0.07)", line_width=0)
    fig.add_trace(go.Scatter(
        x=zs.index, y=zs.values,
        line=dict(color=ORANGE, width=1.2), name="Z-score",
    ))
    fig.add_hline(y=0,  line_dash="dot",  line_color=MUTED)
    fig.add_hline(y=2,  line_dash="dash", line_color=RED,   annotation_text="+2σ")
    fig.add_hline(y=-2, line_dash="dash", line_color=GREEN, annotation_text="−2σ")
    fig.update_layout(
        title=dict(text=f"Beta-Adjusted Spread Z-Score  |  Latest: {z_latest:.2f}",
                   font=dict(color=TEXT, size=12)),
        yaxis_title="Z-score",
        **_base_layout(height=280),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_dist:
    bins = [-np.inf, 0.05, 0.10, 0.15, 0.85, 0.90, 0.95, np.inf]
    labels = ["<5%", "5–10%", "10–15%", "15–85%", "85–90%", "90–95%", ">95%"]
    bin_colors = [RED, "#f85149cc", "#ffa65788", MUTED, "#3fb95088", "#3fb950cc", GREEN]
    counts = pd.cut(sig.dropna(), bins=bins, labels=labels).value_counts().reindex(labels).fillna(0)
    fig = go.Figure(go.Bar(
        x=labels, y=counts.values,
        marker_color=bin_colors, text=counts.values.astype(int),
        textposition="outside", textfont=dict(color=TEXT, size=10),
    ))
    fig.update_layout(
        title=dict(text="Signal Zone Distribution (Waterfall)", font=dict(color=TEXT, size=12)),
        yaxis_title="Count", xaxis_title="Signal bucket",
        **_base_layout(height=280),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 04: CONDITIONAL VOLATILITY & HEDGE RATIO ────────────────────────
st.markdown("<div class='section-header'>04 — CONDITIONAL VOLATILITY & HEDGE RATIO</div>", unsafe_allow_html=True)

col_v, col_h = st.columns(2)

with col_v:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ga.conditional_vol.index,
        y=(ga.conditional_vol * np.sqrt(252)).values,
        line=dict(color=BLUE, width=1.2), name="AMZN σ",
    ))
    fig.add_trace(go.Scatter(
        x=gm.conditional_vol.index,
        y=(gm.conditional_vol * np.sqrt(252)).values,
        line=dict(color=GREEN, width=1.2), name="MSFT σ",
    ))
    fig.update_layout(
        title=dict(text=f"Annualized Conditional Volatility (GJR-GARCH)", font=dict(color=TEXT, size=12)),
        yaxis_title="Ann. Vol", yaxis_tickformat=".0%",
        **_base_layout(height=280),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_h:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dcc.hedge_ratio.index, y=dcc.hedge_ratio.values,
        fill="tozeroy", fillcolor=f"rgba(210,168,255,0.1)",
        line=dict(color=PURPLE, width=1.2), name="β_t",
    ))
    fig.add_hline(y=1.0, line_dash="dot", line_color=MUTED, annotation_text="β=1")
    fig.update_layout(
        title=dict(text=f"Time-Varying Hedge Ratio β_t  |  Latest: {hedge_latest:.4f}",
                   font=dict(color=TEXT, size=12)),
        yaxis_title="Hedge ratio β_t",
        **_base_layout(height=280),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 05: LIVE POSITION SIZING ─────────────────────────────────────────
st.markdown("<div class='section-header'>05 — LIVE POSITION SIZING</div>", unsafe_allow_html=True)

p1, p2, p3, p4, p5, p6 = st.columns(6)
p1.metric("Long NAV ($)", f"${notional_long:,.0f}", delta="Long leg")
p2.metric("Short NAV ($)", f"${notional_short:,.0f}", delta="Short leg")
p3.metric("Spread Vol (ann.)", f"{sigma_spread:.2%}", delta="Risk budget input")
p4.metric("Risk Budget Used", f"{min((notional_long + notional_short) / nav_input, 1):.1%}",
          delta=f"of {risk_budget:.0%} target")
p5.metric("Hedge Ratio β_t", f"{hedge_latest:.4f}", delta="DCC-derived")
p6.metric("Signal Direction", direction_str[:16], delta="Active signal")

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f"<div style='text-align:center;color:{MUTED};font-size:0.7rem;font-family:Space Mono,monospace;'>"
    f"AMZN/MSFT RELATIVE VALUE &nbsp;·&nbsp; GJR-GARCH + DCC + COPULA &nbsp;·&nbsp; "
    f"AIC 6-FAMILY SELECTION &nbsp;·&nbsp; BETA-NEUTRAL FRAMEWORK</div>",
    unsafe_allow_html=True,
)
