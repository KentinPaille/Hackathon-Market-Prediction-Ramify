# bot_trade.py
"""
Bot Phase 1 - Hybrid online ML (RLS) + règles
Signature requise par la plateforme:
def make_decision(epoch: int, price: float) -> {'Asset A': float, 'Cash': float}
Optimisations:
 - historique limité avec deque (évite OOM)
 - unique RNG global (pas de RandomState à chaque appel)
 - protections numériques dans RLS (clamping)
 - robustesse pour faibles longueurs d'historique
"""

import math
import numpy as np
from collections import deque

# ----------------------------
# CONFIG / HYPERPARAMS (tune quickly)
# ----------------------------
VOL_TARGET = 0.03        # cible de vol (par période)
EMA_SHORT = 5
EMA_LONG = 21
RLS_DELTA = 0.999        # forgetting factor
RLS_P0 = 1e4             # initial P diag
MAX_POS = 1.0
MIN_POS = 0.0
STOP_LOSS_PCT = 0.07     # stop loss relatif depuis entrée (7%)
MAX_HOLD_EPOCHS = 50
DRAWDOWN_DEFUSE = 0.15
JITTER_STD = 0.007

# Limits
HISTORY_MAX_LEN = 200    # garder seulement les N derniers prix (évite OOM)
P_CLAMP_MAX = 1e8        # clamp pour la matrice P (évite explosion numérique)

# ----------------------------
# GLOBAL STATE (persist entre appels)
# ----------------------------
_history_prices = deque(maxlen=HISTORY_MAX_LEN)     # garde les derniers prix
_history_returns = deque(maxlen=HISTORY_MAX_LEN)    # derniers retours simples
_entry_price = None
_current_position = 0.0
_epoch_of_entry = None
_peak_equity_price = None
_prev_features = None

_theta = None       # param vector RLS
_P = None           # covariance matrix RLS

# global RNG (unique)
_rng = np.random.default_rng(123456)

# ----------------------------
# HELPERS
# ----------------------------
def _init_rls(n_features):
    global _theta, _P
    _theta = np.zeros((n_features,), dtype=float)
    _P = np.eye(n_features, dtype=float) * RLS_P0

def _rls_update(x, y):
    """
    Stable Recursive Least Squares with forgetting factor and clamping.
    x: np.array shape (n,)
    y: scalar
    """
    global _theta, _P
    # ensure correct shapes
    x = x.reshape(-1, 1)  # (n,1)
    # Px
    Px = _P.dot(x)        # (n,1)
    denom = float(RLS_DELTA + (x.T.dot(Px))[0, 0])
    # guard denom
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        denom = 1e-12
    K = Px / denom        # (n,1)
    err = float(y - float(np.dot(_theta, x.flatten())))
    _theta = _theta + (K.flatten() * err)
    # update P with numerical guard
    try:
        _P = (_P - K.dot(x.T).dot(_P)) / RLS_DELTA
    except Exception:
        # fallback tiny ridge update if numerical issue
        _P = _P + np.eye(_P.shape[0]) * 1e-6
    # clamp P to avoid explosion
    np.clip(_P, -P_CLAMP_MAX, P_CLAMP_MAX, out=_P)

def _safe_tanh(x, factor=10.0):
    # cap input to tanh to avoid overflow
    v = float(x * factor)
    if v > 50:
        v = 50.0
    if v < -50:
        v = -50.0
    return math.tanh(v)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def _ema_from_array(arr, span):
    # simple EMA that works with python list or ndarray
    if len(arr) == 0:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    s = float(arr[0])
    for val in arr[1:]:
        s = alpha * float(val) + (1 - alpha) * s
    return s

def _compute_features(prices_deque):
    """
    prices_deque: deque of recent prices
    returns np.array([bias, last_ret, ema_diff, mom5, vol_norm, zscore])
    Always returns a length-6 vector (no None) using safe fallbacks.
    """
    p = np.array(prices_deque, dtype=float)
    n = len(p)
    # last simple return
    if n >= 2:
        last_ret = (p[-1] - p[-2]) / (p[-2] + 1e-12)
    else:
        last_ret = 0.0

    # EMA short & long
    ema_s = _ema_from_array(p, EMA_SHORT) if n >= 1 else 0.0
    ema_l = _ema_from_array(p, EMA_LONG) if n >= 1 else 0.0
    ema_diff = (ema_s - ema_l) / (p[-1] + 1e-9) if n >= 1 else 0.0

    # momentum 5
    if n >= 6 and p[-6] != 0:
        mom5 = (p[-1] - p[-6]) / (p[-6] + 1e-12)
    else:
        mom5 = last_ret

    # volatility estimation
    if n >= 22:
        # compute 21 returns safely
        prev = p[-22:-1]   # length 21? actually -22:-1 yields 21 if n>=22
        curr = p[-21:]
        # ensure shapes equal
        if len(prev) == len(curr) and len(prev) > 0:
            rets = (curr - prev) / (prev + 1e-12)
        else:
            rets = np.diff(np.log(p + 1e-9))
    else:
        if n >= 2:
            rets = np.diff(np.log(p + 1e-9))
        else:
            rets = np.array([0.0])
    vol = float(np.std(rets)) if rets.size > 0 and np.isfinite(np.std(rets)) else 1e-6

    # zscore vs MA(21)
    window = min(21, n)
    if window > 0:
        ma21 = float(np.mean(p[-window:]))
    else:
        ma21 = float(p[-1] if n >= 1 else 0.0)
    zscore = (p[-1] - ma21) / (vol + 1e-9) if n >= 1 else 0.0

    vol_norm = vol if vol > 1e-9 else 1e-9

    features = np.array([
        1.0,
        last_ret,
        ema_diff,
        mom5,
        vol_norm,
        zscore
    ], dtype=float)

    return features

# ----------------------------
# MAIN DECISION FUNCTION
# ----------------------------
def make_decision(epoch: int, price: float):
    """
    Entrées:
      - epoch: int
      - price: float
    Sortie:
      {'Asset A': w, 'Cash': 1-w} with 0 <= w <= 1
    """
    global _history_prices, _history_returns, _entry_price, _current_position
    global _epoch_of_entry, _peak_equity_price, _prev_features, _theta, _P

    price = float(price)

    # update history (deque handles maxlen)
    if len(_history_prices) > 0:
        prev_price = _history_prices[-1]
        simple_ret = (price - prev_price) / (prev_price + 1e-12)
        _history_returns.append(simple_ret)
    else:
        simple_ret = 0.0
    _history_prices.append(price)

    # compute features
    features = _compute_features(_history_prices)
    n_feat = features.shape[0]
    if _theta is None or _P is None:
        _init_rls(n_feat)

    # RLS update: use prev_features to predict realized return
    if _prev_features is not None and len(_history_returns) >= 1:
        y = float(_history_returns[-1])
        try:
            _rls_update(_prev_features, y)
        except Exception:
            # fallback: tiny ridge step to theta
            _theta = _theta * 0.999 + 1e-6

    # ML prediction (linear model)
    pred = float(np.dot(_theta, features))

    # rule signals
    momentum_score = features[2] + 0.5 * features[3]
    meanrev_score = -features[5]

    est_vol = abs(features[4])
    vol_scale = 1.0 / (1.0 + est_vol * 50.0)

    alpha_ml = 0.6
    combined_score = alpha_ml * pred + 0.25 * momentum_score + 0.15 * meanrev_score
    combined_score *= vol_scale

    safe_vol = max(est_vol, 1e-6)
    raw_scale = VOL_TARGET / safe_vol
    mapped = _safe_tanh(combined_score, factor=10.0)
    desired_weight = max(0.0, raw_scale * mapped)
    desired_weight = float(max(MIN_POS, min(MAX_POS, desired_weight)))

    # risk filters
    if _current_position > 0.0 and _entry_price is not None:
        dd_from_entry = (price - _entry_price) / (_entry_price + 1e-12)
        if dd_from_entry <= -abs(STOP_LOSS_PCT):
            desired_weight = 0.0

    if _current_position > 0.0 and _epoch_of_entry is not None:
        if (epoch - _epoch_of_entry) > MAX_HOLD_EPOCHS:
            desired_weight = min(desired_weight, 0.5 * _current_position)

    if _entry_price is not None:
        if _peak_equity_price is None:
            _peak_equity_price = _entry_price
        _peak_equity_price = max(_peak_equity_price, price)
        if _peak_equity_price > 0:
            drawd = (_peak_equity_price - price) / (_peak_equity_price + 1e-12)
            if drawd > DRAWDOWN_DEFUSE:
                desired_weight *= 0.5

    # smoothing / turnover control
    lam = 0.6
    smoothed = lam * _current_position + (1 - lam) * desired_weight

    # jitter reproducible via global RNG
    jitter = float(_rng.normal(scale=JITTER_STD))
    smoothed = float(np.clip(smoothed + jitter, 0.0, 1.0))

    w_asset = smoothed
    w_cash = float(1.0 - w_asset)

    # update entry/exit trackers
    if w_asset > 0.001 and _current_position <= 0.001:
        _entry_price = price
        _epoch_of_entry = epoch
        _peak_equity_price = price
    if w_asset <= 0.001 and _current_position > 0.001:
        _entry_price = None
        _epoch_of_entry = None
        _peak_equity_price = None

    _current_position = w_asset
    _prev_features = features.copy()

    # numeric stability
    total = w_asset + w_cash
    if abs(total - 1.0) > 1e-9:
        w_asset = w_asset / total
        w_cash = 1.0 - w_asset

    return {'Asset A': float(w_asset), 'Cash': float(w_cash)}
