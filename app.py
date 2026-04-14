import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize
from scipy.special import gammaln
import yfinance as yf
from arch import arch_model
import warnings
from dataclasses import dataclass, field
from typing import Literal
import logging
from datetime import datetime

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AMZN/MSFT Relative Value",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #58a6ff;
    --green:     #3fb950;
    --red:       #f85149;
    --yellow:    #d29922;
    --text:      #e6edf3;
    --muted:     #8b949e;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1c2128 100%);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--green), var(--accent));
}

.main-header h1 {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 4px 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: var(--muted);
    font-size: 0.85rem;
    margin: 0;
    letter-spacing: 0.3px;
}

.kpi-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 18px;
    position: relative;
    overflow: hidden;
}

.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
}

.kpi-card.blue::after   { background: var(--accent); }
.kpi-card.green::after  { background: var(--green); }
.kpi-card.red::after    { background: var(--red); }
.kpi-card.yellow::after { background: var(--yellow); }

.kpi-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 6px;
}

.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
    margin-bottom: 4px;
}

.kpi-sub {
    font-size: 0.75rem;
    color: var(--muted);
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: var(--accent);
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 24px 0 16px 0;
}

.regime-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}

.regime-table td {
    padding: 6px 10px;
    border-bottom: 1px solid var(--border);
}

.regime-table td:first-child { color: var(--muted); }
.regime-table td:last-child  { color: var(--text); font-family: 'Space Mono', monospace; }

.badge-green  { color: var(--green); }
.badge-red    { color: var(--red); }
.badge-yellow { color: var(--yellow); }
.badge-blue   { color: var(--accent); }

.stMetric { display: none; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Types and helpers (self-contained — no imports from Colab)
# ─────────────────────────────────────────────────────────────────────────────

CopulaFamily = Literal["gaussian", "student_t", "clayton", "gumbel", "frank", "bb1"]

@dataclass
class GARCHResult:
    ticker: str
    params: dict
    nu: float
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
    family: str
    params: dict
    log_likelihood: float
    n_params: int
    aic: float

@dataclass
class TechResult:
    garch_amzn: GARCHResult
    garch_msft: GARCHResult
    dcc: DCCResult
    u_amzn: pd.Series
    u_msft: pd.Series
    pit_amzn_valid: bool
    pit_msft_valid: bool
    copula_fits: list
    selected_copula: CopulaFitResult
    signal: pd.Series


def _aic(ll, k): return 2*k - 2*ll


def fit_gjr_garch(log_returns, ticker):
    model  = arch_model(log_returns * 100, mean="Constant", vol="GARCH",
                        p=1, o=1, q=1, dist="studentst")
    result = model.fit(disp="off", show_warning=False)
    sigma  = result.conditional_volatility / 100
    resid  = result.resid / 100
    z      = resid / sigma
    params = {
        "mu":    result.params["mu"] / 100,
        "omega": result.params["omega"],
        "alpha": result.params["alpha[1]"],
        "gamma": result.params["gamma[1]"],
        "beta":  result.params["beta[1]"],
    }
    return GARCHResult(
        ticker=ticker, params=params, nu=result.params["nu"],
        conditional_vol=pd.Series(sigma, index=log_returns.index),
        standardized_residuals=pd.Series(z, index=log_returns.index),
        log_returns=log_returns,
    )


def fit_dcc(ga, gb):
    Z     = np.column_stack([ga.standardized_residuals.values,
                              gb.standardized_residuals.values])
    Q_bar = np.cov(Z.T)

    def neg_ll(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 1: return 1e10
        T  = Z.shape[0]
        Q  = Q_bar.copy()
        ll = 0.0
        for t in range(1, T):
            zl = Z[t-1].reshape(-1,1)
            Q  = (1-a-b)*Q_bar + a*(zl@zl.T) + b*Q
            di = np.diag(1/np.sqrt(np.diag(Q)))
            R  = di@Q@di
            np.fill_diagonal(R, 1.0)
            s, ld = np.linalg.slogdet(R)
            if s <= 0: return 1e10
            zt = Z[t]
            ll += -0.5*(ld + zt@np.linalg.solve(R, zt) - zt@zt)
        return -ll

    res  = optimize.minimize(neg_ll, [0.02, 0.95], method="L-BFGS-B",
                              bounds=[(1e-6,0.5),(1e-6,0.9999)])
    a, b = res.x
    T    = Z.shape[0]
    rho_vals = np.zeros(T)
    H_vals   = np.zeros((T,2,2))
    Q        = Q_bar.copy()
    sa, sb   = ga.conditional_vol.values, gb.conditional_vol.values

    for t in range(T):
        if t > 0:
            zl = Z[t-1].reshape(-1,1)
            Q  = (1-a-b)*Q_bar + a*(zl@zl.T) + b*Q
        di = np.diag(1/np.sqrt(np.diag(Q)))
        R  = di@Q@di
        np.fill_diagonal(R, 1.0)
        rho_vals[t] = np.clip(R[0,1], -0.9999, 0.9999)
        D = np.diag([sa[t], sb[t]])
        H_vals[t] = D@R@D

    idx   = ga.conditional_vol.index
    rho   = pd.Series(rho_vals, index=idx, name="rho")
    hedge = rho * (ga.conditional_vol / gb.conditional_vol)
    return DCCResult(rho=rho, H=H_vals, hedge_ratio=hedge,
                     dcc_params={"a":a,"b":b}, Q_bar=Q_bar)


def pit_transform(garch):
    u = stats.t.cdf(garch.standardized_residuals.values, df=garch.nu)
    u = np.clip(u, 1e-6, 1-1e-6)
    return pd.Series(u, index=garch.standardized_residuals.index)


def _gauss_ll(r, u, v):
    if abs(r) >= 1: return -np.inf
    x, y = stats.norm.ppf(u), stats.norm.ppf(v)
    return np.sum(-0.5*np.log(1-r**2) - (r**2*(x**2+y**2)-2*r*x*y)/(2*(1-r**2)))

def _t_ll(p, u, v):
    r, nu = p
    if abs(r) >= 1 or nu <= 2: return -np.inf
    x, y = stats.t.ppf(u, df=nu), stats.t.ppf(v, df=nu)
    det  = 1-r**2
    A    = (x**2+y**2-2*r*x*y)/det
    return np.sum(gammaln((nu+2)/2)+gammaln(nu/2)-2*gammaln((nu+1)/2)
                  -0.5*np.log(det)+(nu+1)/2*(np.log(1+x**2/nu)+np.log(1+y**2/nu))
                  -(nu+2)/2*np.log(1+A/nu))

def _clay_ll(t, u, v):
    if t <= 0: return -np.inf
    return np.sum(np.log(t+1)+(-t-1)*(np.log(u)+np.log(v))
                  +(-1/t-2)*np.log(u**(-t)+v**(-t)-1))

def _gumb_ll(t, u, v):
    if t < 1: return -np.inf
    la, lb = -np.log(u), -np.log(v)
    S  = la**t+lb**t
    C  = np.exp(-(S**(1/t)))
    return np.sum(np.log(C)+(1/t-2)*np.log(S)+(t-1)*(np.log(la)+np.log(lb))
                  -np.log(u)-np.log(v)+np.log(S**(1/t-1)+(t-1)*S**(1/t-2)))

def _frank_ll(t, u, v):
    if t == 0: return -np.inf
    et  = np.exp(-t)
    etu = np.exp(-t*u)
    etv = np.exp(-t*v)
    d   = (1-et)-(1-etu)*(1-etv)
    return np.sum(np.log(t)+np.log(1-et)-t-2*np.log(np.abs(d)))

def _bb1_ll(p, u, v):
    t, d = p
    if t <= 0 or d < 1: return -np.inf
    x = (u**(-t)-1)**d
    y = (v**(-t)-1)**d
    S = x+y
    try:
        ll = (np.log(t*d+1)+(t+1)/t*(np.log(u)+np.log(v))
              +(d-1)*(np.log(x/u)+np.log(y/v))+(1/d-2)*np.log(S)-S**(1/d))
        return np.sum(ll)
    except: return -np.inf


def fit_copulas(ua, ub):
    res = []
    o = optimize.minimize_scalar(lambda r: -_gauss_ll(r,ua,ub), bounds=(-0.999,0.999), method="bounded")
    ll = _gauss_ll(o.x,ua,ub); res.append(CopulaFitResult("gaussian",{"rho":o.x},ll,1,_aic(ll,1)))

    o = optimize.minimize(lambda p: -_t_ll(p,ua,ub), x0=[0.5,5.0],
                           bounds=[(-0.999,0.999),(2.01,50.0)], method="L-BFGS-B")
    ll = _t_ll(o.x,ua,ub); res.append(CopulaFitResult("student_t",{"rho":o.x[0],"nu":o.x[1]},ll,2,_aic(ll,2)))

    o = optimize.minimize_scalar(lambda t: -_clay_ll(t,ua,ub), bounds=(1e-4,20.0), method="bounded")
    ll = _clay_ll(o.x,ua,ub); res.append(CopulaFitResult("clayton",{"theta":o.x},ll,1,_aic(ll,1)))

    o = optimize.minimize_scalar(lambda t: -_gumb_ll(t,ua,ub), bounds=(1.0,20.0), method="bounded")
    ll = _gumb_ll(o.x,ua,ub); res.append(CopulaFitResult("gumbel",{"theta":o.x},ll,1,_aic(ll,1)))

    o = optimize.minimize_scalar(lambda t: -_frank_ll(t,ua,ub), bounds=(-20.0,20.0), method="bounded")
    ll = _frank_ll(o.x,ua,ub); res.append(CopulaFitResult("frank",{"theta":o.x},ll,1,_aic(ll,1)))

    o = optimize.minimize(lambda p: -_bb1_ll(p,ua,ub), x0=[0.5,1.5],
                           bounds=[(1e-4,10.0),(1.0,10.0)], method="L-BFGS-B")
    ll = _bb1_ll(o.x,ua,ub); res.append(CopulaFitResult("bb1",{"theta":o.x[0],"delta":o.x[1]},ll,2,_aic(ll,2)))

    return sorted(res, key=lambda r: r.aic)


def select_copula(fits, tie=2.0):
    best = fits[0]
    t_fit = next((f for f in fits if f.family=="student_t"), None)
    if t_fit and (t_fit.aic - best.aic) < tie:
        return t_fit
    return best


def compute_copula_signal(ua, ub, copula):
    ua = np.clip(np.asarray(ua, dtype=float), 1e-6, 1-1e-6)
    ub = np.clip(np.asarray(ub, dtype=float), 1e-6, 1-1e-6)
    f  = copula.family
    p  = copula.params

    if f == "gaussian":
        r = p["rho"]; xa = stats.norm.ppf(ua); xb = stats.norm.ppf(ub)
        s = stats.norm.cdf((xb - r*xa)/np.sqrt(1-r**2))
    elif f == "student_t":
        r, nu = p["rho"], p["nu"]
        ta, tb = stats.t.ppf(ua,df=nu), stats.t.ppf(ub,df=nu)
        s = stats.t.cdf((tb-r*ta)/np.sqrt((1-r**2)*(nu+ta**2)/(nu+1)), df=nu+1)
    elif f == "clayton":
        t = p["theta"]
        s = ua**(-(t+1))*(ua**(-t)+ub**(-t)-1)**(-(1+1/t))
    elif f == "gumbel":
        t = p["theta"]; la, lb = -np.log(ua), -np.log(ub)
        S = la**t+lb**t; C = np.exp(-(S**(1/t)))
        s = C/ua*(S**(1/t-1))*la**(t-1)
    elif f == "frank":
        t = p["theta"]; et=np.exp(-t); etu=np.exp(-t*ua); etv=np.exp(-t*ub)
        s = etu*(1-etv)/((1-et)-(1-etu)*(1-etv))
    elif f == "bb1":
        t, d = p["theta"], p["delta"]
        x = (ua**(-t)-1)**d; y = (ub**(-t)-1)**d; S = x+y
        dS = -t*d*ua**(-t-1)*(ua**(-t)-1)**(d-1)
        s = np.exp(-(S**(1/d)))*S**(1/d-1)*dS/d
    else:
        raise ValueError(f"Unknown family: {f}")

    return np.clip(s, 1e-6, 1-1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Data pipeline (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_model():
    prices = yf.download(["AMZN","MSFT"], period="6y",
                          auto_adjust=True, progress=False)["Close"]
    # Handle MultiIndex columns from newer yfinance versions
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.droplevel(0, axis=1)
    prices = prices[["AMZN","MSFT"]].dropna()
    log_ret = np.log(prices / prices.shift(1)).dropna()

    ga = fit_gjr_garch(log_ret["AMZN"], "AMZN")
    gb = fit_gjr_garch(log_ret["MSFT"], "MSFT")
    dcc = fit_dcc(ga, gb)

    ua = pit_transform(ga)
    ub = pit_transform(gb)
    pit_a = stats.kstest(ua.values, "uniform").pvalue >= 0.05
    pit_b = stats.kstest(ub.values, "uniform").pvalue >= 0.05

    fits = fit_copulas(ua.values, ub.values)
    cop  = select_copula(fits)

    sig_vals = compute_copula_signal(ua.values, ub.values, cop)
    signal   = pd.Series(sig_vals, index=ua.index, name="signal")

    return TechResult(
        garch_amzn=ga, garch_msft=gb, dcc=dcc,
        u_amzn=ua, u_msft=ub,
        pit_amzn_valid=pit_a, pit_msft_valid=pit_b,
        copula_fits=fits, selected_copula=cop, signal=signal,
    )


@st.cache_data(show_spinner=False)
def load_prices():
    p = yf.download(["AMZN","MSFT"], period="5y",
                     auto_adjust=True, progress=False)["Close"]
    if isinstance(p.columns, pd.MultiIndex):
        p = p.droplevel(0, axis=1)
    return p[["AMZN","MSFT"]].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='font-family:Space Mono,monospace; font-size:0.75rem;
                color:#58a6ff; letter-spacing:1px; margin-bottom:16px;'>
    ◈ STRATEGY CONFIG
    </div>
    """, unsafe_allow_html=True)

    nav_input        = st.number_input("Portfolio NAV ($)", min_value=100_000,
                                        max_value=100_000_000, value=1_000_000, step=100_000)
    entry_lo         = st.slider("Entry threshold (low)", 0.01, 0.20, 0.10, 0.01)
    entry_hi         = 1.0 - entry_lo
    min_corr         = st.slider("Min correlation filter", 0.20, 0.60, 0.40, 0.05)
    time_stop        = st.slider("Time stop (days)", 5, 30, 15)
    risk_budget      = st.slider("Risk budget (% NAV)", 1, 10, 5) / 100

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#8b949e; line-height:1.6;'>
    <b style='color:#e6edf3;'>Model Stack</b><br/>
    GJR-GARCH(1,1) marginals<br/>
    DCC-GARCH dynamic correlation<br/>
    Copula h-function signal<br/>
    AIC copula family selection<br/><br/>
    <b style='color:#e6edf3;'>Exit Rules</b><br/>
    Full mean reversion → 0.50<br/>
    Partial profit → 0.30–0.70<br/>
    Time stop · Hard P&L stop<br/>
    Signal divergence stop
    </div>
    """, unsafe_allow_html=True)

    refresh = st.button("🔄 Refresh Model", use_container_width=True, type="primary")
    if refresh:
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("⚙️ Fitting GJR-GARCH → DCC → Copula pipeline..."):
    try:
        tr = load_model()
        prices = load_prices()
        model_ok = True
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        model_ok = False
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

latest_sig = float(tr.signal.iloc[-1])
latest_rho = float(tr.dcc.rho.iloc[-1])
latest_h   = float(tr.dcc.hedge_ratio.iloc[-1])
sa_ann     = float(tr.garch_amzn.conditional_vol.iloc[-1]) * np.sqrt(252) * 100
sb_ann     = float(tr.garch_msft.conditional_vol.iloc[-1]) * np.sqrt(252) * 100
cop_name   = tr.selected_copula.family.replace("_","-").title()

if latest_sig < entry_lo:
    zone_label = "🔴 Long AMZN / Short MSFT"
    zone_color = "#f85149"
elif latest_sig > entry_hi:
    zone_label = "🟢 Long MSFT / Short AMZN"
    zone_color = "#3fb950"
else:
    zone_label = "⚪ Neutral — No Trade"
    zone_color = "#8b949e"

st.markdown(f"""
<div class="main-header">
  <h1>📊 AMZN / MSFT Relative Value</h1>
  <p>GJR-GARCH → DCC → Copula Pipeline &nbsp;|&nbsp; Data: Yahoo Finance &nbsp;|&nbsp;
     Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>
""", unsafe_allow_html=True)

# KPI cards
k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.markdown(f"""
    <div class="kpi-card yellow">
      <div class="kpi-label">Copula Signal</div>
      <div class="kpi-value">{latest_sig:.4f}</div>
      <div class="kpi-sub" style="color:{zone_color};">{zone_label}</div>
    </div>""", unsafe_allow_html=True)

with k2:
    rho_color = "#3fb950" if latest_rho > 0.40 else "#f85149"
    st.markdown(f"""
    <div class="kpi-card green">
      <div class="kpi-label">DCC Correlation ρ_t</div>
      <div class="kpi-value">{latest_rho:.4f}</div>
      <div class="kpi-sub" style="color:{rho_color};">
        {"✅ Valid (ρ > 0.40)" if latest_rho > 0.40 else "⚠️ Below threshold"}
      </div>
    </div>""", unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card blue">
      <div class="kpi-label">Hedge Ratio h_t</div>
      <div class="kpi-value">{latest_h:.4f}</div>
      <div class="kpi-sub">β-neutral sizing</div>
    </div>""", unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card red">
      <div class="kpi-label">AMZN Ann. Vol</div>
      <div class="kpi-value">{sa_ann:.1f}%</div>
      <div class="kpi-sub">GJR-GARCH conditional</div>
    </div>""", unsafe_allow_html=True)

with k5:
    st.markdown(f"""
    <div class="kpi-card red">
      <div class="kpi-label">MSFT Ann. Vol</div>
      <div class="kpi-value">{sb_ann:.1f}%</div>
      <div class="kpi-sub">GJR-GARCH conditional</div>
    </div>""", unsafe_allow_html=True)

with k6:
    st.markdown(f"""
    <div class="kpi-card blue">
      <div class="kpi-label">Active Copula</div>
      <div class="kpi-value" style="font-size:1.1rem;">{cop_name}</div>
      <div class="kpi-sub">AIC selected</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Row 1: DCC + Signal
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">01 — DCC CORRELATION & COPULA SIGNAL</div>',
            unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    rho   = tr.dcc.rho
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=rho.index, y=rho.values, mode="lines",
        line=dict(color="#58a6ff", width=1.5), name="ρ_t",
    ))
    fig.add_hrect(y0=0, y1=0.40, fillcolor="rgba(248,81,73,0.08)", line_width=0)
    fig.add_hrect(y0=0.40, y1=0.70, fillcolor="rgba(210,153,34,0.06)", line_width=0)
    fig.add_hrect(y0=0.70, y1=1.0, fillcolor="rgba(63,185,80,0.06)", line_width=0)
    fig.add_hline(y=0.40, line_dash="dash", line_color="rgba(248,81,73,0.6)",
                  annotation_text="0.40 filter", annotation_position="right",
                  annotation_font_color="#f85149")
    fig.add_hline(y=0.70, line_dash="dash", line_color="rgba(63,185,80,0.6)",
                  annotation_text="0.70", annotation_position="right",
                  annotation_font_color="#3fb950")
    fig.update_layout(
        title=dict(text="DCC Time-Varying Correlation ρ_t", font=dict(size=13, color="#e6edf3")),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", range=[-0.05, 1.05]),
        height=350, margin=dict(l=50, r=120, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    sig = tr.signal
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=entry_lo, fillcolor="rgba(248,81,73,0.12)", line_width=0)
    fig.add_hrect(y0=entry_hi, y1=1.0, fillcolor="rgba(63,185,80,0.12)", line_width=0)
    fig.add_hrect(y0=0.15, y1=0.85, fillcolor="rgba(139,148,158,0.04)", line_width=0)
    fig.add_trace(go.Scatter(
        x=sig.index, y=sig.values, mode="lines",
        line=dict(color="#d2a8ff", width=1.5), name="Signal",
    ))
    for level, color, label in [
        (entry_lo, "#f85149", f"{entry_lo:.2f} entry"),
        (0.50,     "#8b949e", "0.50 neutral"),
        (entry_hi, "#3fb950", f"{entry_hi:.2f} entry"),
    ]:
        fig.add_hline(y=level, line_dash="dash",
                      line_color=color.replace(")", ",0.6)").replace("rgb","rgba") if "rgb" in color else color,
                      annotation_text=label, annotation_position="right",
                      annotation_font_color=color)
    fig.update_layout(
        title=dict(text="Copula Signal — h-function", font=dict(size=13, color="#e6edf3")),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", range=[0, 1]),
        height=350, margin=dict(l=50, r=120, t=50, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 2: Copula surface + Regime
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">02 — COPULA DENSITY SURFACE & REGIME INDICATOR</div>',
            unsafe_allow_html=True)

col3, col4 = st.columns([3, 2])

with col3:
    g      = np.linspace(0.03, 0.97, 35)
    UG, VG = np.meshgrid(g, g)
    uf     = UG.ravel()
    vf     = VG.ravel()
    h_     = 0.01
    spp    = compute_copula_signal(np.clip(uf+h_,1e-4,1-1e-4),
                                    np.clip(vf+h_,1e-4,1-1e-4), tr.selected_copula)
    spm    = compute_copula_signal(np.clip(uf+h_,1e-4,1-1e-4),
                                    np.clip(vf-h_,1e-4,1-1e-4), tr.selected_copula)
    dens   = np.clip((spp-spm)/(2*h_), 0, 8).reshape(35, 35)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=g, y=g, z=dens, colorscale="Plasma", opacity=0.9,
        colorbar=dict(title="c(u,v)", tickfont=dict(color="#8b949e"),
                      titlefont=dict(color="#8b949e"), len=0.6),
    ))
    fig.add_trace(go.Scatter3d(
        x=[float(tr.u_amzn.iloc[-1])],
        y=[float(tr.u_msft.iloc[-1])],
        z=[dens.max() * 0.75],
        mode="markers",
        marker=dict(size=10, color="#f85149", symbol="diamond"),
        name="Today",
    ))
    fig.update_layout(
        title=dict(text=f"Copula Density Surface — {cop_name}",
                   font=dict(size=13, color="#e6edf3")),
        scene=dict(
            xaxis=dict(title="u_AMZN", gridcolor="#21262d",
                        backgroundcolor="#161b22", color="#8b949e"),
            yaxis=dict(title="u_MSFT", gridcolor="#21262d",
                        backgroundcolor="#161b22", color="#8b949e"),
            zaxis=dict(title="c(u,v)", gridcolor="#21262d",
                        backgroundcolor="#161b22", color="#8b949e"),
            bgcolor="#161b22",
        ),
        paper_bgcolor="#161b22",
        font=dict(color="#8b949e", family="DM Sans"),
        height=420, margin=dict(l=0, r=0, t=50, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    pit_a_str = ("✅ PASS" if tr.pit_amzn_valid else "❌ FAIL")
    pit_b_str = ("✅ PASS" if tr.pit_msft_valid else "❌ FAIL")

    params_str = "  ".join([f"{k}={float(v):.4f}" for k,v in tr.selected_copula.params.items()])

    regime_html = f"""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                padding:16px; font-family:'DM Sans',sans-serif;">
      <div style="font-family:'Space Mono',monospace; font-size:0.7rem; color:#58a6ff;
                  letter-spacing:1px; margin-bottom:12px;">REGIME INDICATOR</div>
      <table class="regime-table">
        <tr><td>Active Copula</td><td>{cop_name}</td></tr>
        <tr><td>Parameters</td><td style="font-size:0.75rem;">{params_str}</td></tr>
        <tr><td>Log-Likelihood</td><td>{tr.selected_copula.log_likelihood:.2f}</td></tr>
        <tr><td>AIC</td><td>{tr.selected_copula.aic:.2f}</td></tr>
        <tr><td>PIT KS — AMZN</td><td>{pit_a_str}</td></tr>
        <tr><td>PIT KS — MSFT</td><td>{pit_b_str}</td></tr>
        <tr><td>Latest ρ_t</td><td>{latest_rho:.4f}</td></tr>
        <tr><td>Latest h_t</td><td>{latest_h:.4f}</td></tr>
      </table>
      <div style="margin-top:14px; font-family:'Space Mono',monospace; font-size:0.7rem;
                  color:#58a6ff; letter-spacing:1px;">COPULA AIC RANKINGS</div>
      <table style="width:100%; border-collapse:collapse; margin-top:8px; font-size:0.78rem;">
        <tr style="border-bottom:1px solid #30363d;">
          <th style="text-align:left; padding:4px 8px; color:#8b949e; font-weight:500;">Family</th>
          <th style="text-align:right; padding:4px 8px; color:#8b949e; font-weight:500;">AIC</th>
          <th style="text-align:center; padding:4px 8px; color:#8b949e; font-weight:500;">Rank</th>
        </tr>
    """
    for i, f in enumerate(tr.copula_fits):
        bg      = "#1c2128" if f.family == tr.selected_copula.family else "transparent"
        sel_txt = "★ selected" if f.family == tr.selected_copula.family else ""
        regime_html += f"""
        <tr style="background:{bg}; border-bottom:1px solid #21262d;">
          <td style="padding:5px 8px; color:#e6edf3;">{f.family.replace('_','-').title()}</td>
          <td style="padding:5px 8px; text-align:right; font-family:'Space Mono',monospace;
                     color:#e6edf3;">{f.aic:.1f}</td>
          <td style="padding:5px 8px; text-align:center; color:#3fb950; font-size:0.72rem;">{sel_txt}</td>
        </tr>"""
    regime_html += "</table></div>"
    st.markdown(regime_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 3: Spread Z-Score + Signal Zone distribution
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">03 — SPREAD Z-SCORE & SIGNAL DISTRIBUTION</div>',
            unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    try:
        hedge_a  = tr.dcc.hedge_ratio.reindex(prices.index).ffill()
        spread   = prices["AMZN"] - hedge_a * prices["MSFT"]
        mu_s     = spread.rolling(252).mean()
        sig_s    = spread.rolling(252).std().replace(0, np.nan)
        z_spread = ((spread - mu_s) / sig_s).dropna()
        latest_z = float(z_spread.iloc[-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=z_spread.index, y=z_spread.values, mode="lines",
            line=dict(color="#79c0ff", width=1.2), name="Z-score",
        ))
        for level, color in [(2.0,"#f85149"),(-2.0,"#f85149"),(1.0,"#d29922"),(-1.0,"#d29922")]:
            fig.add_hline(y=level, line_dash="dot", line_color=color, line_width=1)
        fig.add_hline(y=0, line_color="#30363d", line_width=0.8)

        fig.update_layout(
            title=dict(text=f"Beta-Adjusted Spread Z-Score  |  Latest: {latest_z:.2f}",
                       font=dict(size=13, color="#e6edf3")),
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font=dict(color="#8b949e", family="DM Sans"),
            xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
            yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
            height=330, margin=dict(l=50, r=50, t=50, b=40), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Spread chart unavailable: {e}")

with col6:
    sig_vals = tr.signal.values
    zones    = pd.cut(sig_vals,
                       bins=[0, 0.05, 0.10, 0.15, 0.85, 0.90, 0.95, 1.0],
                       labels=["Strong Short MSFT", "Mod Short MSFT", "Weak Low",
                                "Neutral", "Weak High", "Mod Long MSFT", "Strong Long MSFT"])
    zone_counts = zones.value_counts().reindex(
        ["Strong Short MSFT","Mod Short MSFT","Weak Low","Neutral",
         "Weak High","Mod Long MSFT","Strong Long MSFT"])
    zone_colors = ["#f85149","#ff7b72","#ffa657","#8b949e","#7ee787","#3fb950","#1a7f37"]

    fig = go.Figure(go.Bar(
        x=zone_counts.index.tolist(),
        y=zone_counts.values,
        marker_color=zone_colors,
        text=zone_counts.values,
        textposition="auto",
        textfont=dict(color="#e6edf3", size=10),
    ))
    fig.update_layout(
        title=dict(text="Signal Zone Distribution (historical)",
                   font=dict(size=13, color="#e6edf3")),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d", tickangle=-25, tickfont=dict(size=9)),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        height=330, margin=dict(l=50, r=50, t=50, b=80), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 4: Volatility + Hedge ratio
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">04 — CONDITIONAL VOLATILITY & HEDGE RATIO</div>',
            unsafe_allow_html=True)

col7, col8 = st.columns(2)

with col7:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tr.garch_amzn.conditional_vol.index,
        y=tr.garch_amzn.conditional_vol.values * np.sqrt(252) * 100,
        mode="lines", line=dict(color="#58a6ff", width=1.2), name="AMZN",
    ))
    fig.add_trace(go.Scatter(
        x=tr.garch_msft.conditional_vol.index,
        y=tr.garch_msft.conditional_vol.values * np.sqrt(252) * 100,
        mode="lines", line=dict(color="#3fb950", width=1.2), name="MSFT",
    ))
    fig.update_layout(
        title=dict(text="Annualized Conditional Volatility (GJR-GARCH)",
                   font=dict(size=13, color="#e6edf3")),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d", ticksuffix="%"),
        height=300, margin=dict(l=50, r=50, t=50, b=40),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d",
                    font=dict(color="#e6edf3")),
    )
    st.plotly_chart(fig, use_container_width=True)

with col8:
    hedge = tr.dcc.hedge_ratio
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=hedge.index, y=hedge.values, mode="lines",
        line=dict(color="#d2a8ff", width=1.2), name="h_t",
        fill="tozeroy", fillcolor="rgba(210,168,255,0.08)",
    ))
    fig.add_hline(y=float(hedge.mean()), line_dash="dash",
                  line_color="#8b949e", annotation_text="Mean",
                  annotation_position="right", annotation_font_color="#8b949e")
    fig.update_layout(
        title=dict(text=f"Time-Varying Hedge Ratio h_t  |  Latest: {latest_h:.4f}",
                   font=dict(size=13, color="#e6edf3")),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font=dict(color="#8b949e", family="DM Sans"),
        xaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        yaxis=dict(gridcolor="#21262d", linecolor="#30363d"),
        height=300, margin=dict(l=50, r=120, t=50, b=40), showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Row 5: Position sizing preview
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">05 — LIVE POSITION SIZING</div>',
            unsafe_allow_html=True)

sa_d = float(tr.garch_amzn.conditional_vol.iloc[-1])
sb_d = float(tr.garch_msft.conditional_vol.iloc[-1])
h_t  = latest_rho * (sa_d / sb_d)
var_sp = sa_d**2 + h_t**2*sb_d**2 - 2*h_t*latest_rho*sa_d*sb_d
sig_sp = np.sqrt(max(var_sp, 1e-12)) * np.sqrt(252)
nl     = (risk_budget * nav_input) / sig_sp if sig_sp > 0 else 0
ns     = h_t * nl
max_n  = 0.20 * nav_input
capped = False
if max(nl, ns) > max_n:
    sc = max_n / max(nl, ns)
    nl *= sc; ns *= sc
    capped = True

p1, p2, p3, p4, p5 = st.columns(5)
sizing_items = [
    ("Long Leg Notional", f"${nl:,.0f}", f"{nl/nav_input*100:.1f}% NAV", "blue"),
    ("Short Leg Notional", f"${ns:,.0f}", f"{ns/nav_input*100:.1f}% NAV", "blue"),
    ("Spread Vol (ann.)", f"{sig_sp*100:.2f}%", "Risk budget input", "yellow"),
    ("Hedge Ratio h_t", f"{h_t:.4f}", "ρ × σ_long/σ_short", "green"),
    ("Cap Applied", "Yes" if capped else "No", "20% NAV limit", "red" if capped else "green"),
]
for col, (label, val, sub, color) in zip([p1,p2,p3,p4,p5], sizing_items):
    with col:
        st.markdown(f"""
        <div class="kpi-card {color}">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value" style="font-size:1.2rem;">{val}</div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("<br/>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#30363d; font-family:'Space Mono',monospace;
            font-size:0.65rem; letter-spacing:1px; padding:16px; border-top:1px solid #21262d;">
AMZN/MSFT RELATIVE VALUE · GJR-GARCH → DCC → COPULA ·
MC IS FOR RISK CHARACTERIZATION ONLY · DATA: YAHOO FINANCE
</div>
""", unsafe_allow_html=True)
