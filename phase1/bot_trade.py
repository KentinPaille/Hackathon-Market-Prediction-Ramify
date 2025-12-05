# bot_trade.py
"""
Bot Phase 1 - Hybrid online ML (RLS) + règles
Signature requise par la plateforme:
def make_decision(epoch: int, price: float) -> {'Asset A': float, 'Cash': float}
"""

import math
import numpy as np

# ----------------------------
# CONFIG / HYPERPARAMS (tune quickly)
# ----------------------------
# SEED = np.random.randint(0, 1000000)  # changeable pour diversifier la "signature" et réduire similarité
# np.random.seed(SEED)

VOL_TARGET = 0.03        # cible de vol (ex: 2% par période) pour volatility targeting
EMA_SHORT = 5
EMA_LONG = 21
RLS_DELTA = 0.999        # forgetting factor (0.98-1.0). 1=no forget
RLS_P0 = 1e4             # initial P diag
MAX_POS = 1
MIN_POS = 0.0
STOP_LOSS_PCT = 0.07     # stop loss relatif depuis entrée (7%)
MAX_HOLD_EPOCHS = 50     # réduit exposition si tenue trop longue
DRAWDOWN_DEFUSE = 0.15   # si drawdown depuis peak > 15% -> cut risk
JITTER_STD = 0.007       # small reproducible jitter to avoid perfect similarity

# ----------------------------
# INTERNAL STATE (persist between calls)
# ----------------------------
_history_prices = []     # list of floats
_history_returns = []    # list of floats (simple returns)
_entry_price = None      # price at last entry
_current_position = 0.0  # last position (Asset A weight)
_epoch_of_entry = None
_peak_equity_price = None  # to track drawdown since entry/peak
_prev_features = None
# RLS parameters
_theta = None            # parameter vector
_P = None                # covariance matrix


# ----------------------------
# HELPERS
# ----------------------------
def _init_rls(n_features):
    global _theta, _P
    _theta = np.zeros((n_features,))
    _P = np.eye(n_features) * RLS_P0

def _rls_update(x, y):
    """
    Recursive Least Squares update with forgetting factor.
    x: feature vector (n,)
    y: scalar target
    """
    global _theta, _P
    x = x.reshape(-1, 1)
    # compute gain
    Px = _P.dot(x)
    denom = (RLS_DELTA + (x.T.dot(Px))[0, 0])
    K = Px / denom
    # update theta
    err = y - float(np.dot(_theta, x.flatten()))
    _theta = _theta + (K.flatten() * err)
    # update P
    _P = (_P - K.dot(x.T).dot(_P)) / RLS_DELTA

def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def _safe_div(a, b, eps=1e-8):
    return a / b if abs(b) > eps else a / (eps if b==0 else b)

# ----------------------------
# FEATURE ENGINEERING
# ----------------------------
def _compute_features(prices):
    """
    Construct a small, robust feature vector from price history.
    prices: list or np.array of close prices (length >= 1)
    Returns: np.array(features,)
    Features:
      - bias (1)
      - last simple return
      - short ema - long ema (normalized)
      - short momentum (ret 5)
      - vol (std of returns last 20)
      - zscore between price and 21-ema
    """
    p = np.array(prices)
    n = len(p)
    # simple returns
    if n >= 2:
        last_ret = (p[-1] - p[-2]) / p[-2]
    else:
        last_ret = 0.0

    # EMA helper
    def ema(arr, span):
        alpha = 2.0 / (span + 1.0)
        s = arr[0]
        for val in arr[1:]:
            s = alpha * val + (1 - alpha) * s
        return s

    ema_s = ema(p, EMA_SHORT) if n >= 1 else p[-1]
    ema_l = ema(p, EMA_LONG) if n >= 1 else p[-1]
    ema_diff = (ema_s - ema_l) / (p[-1] + 1e-9)

    # momentum 5
    if n >= 6:
        mom5 = (p[-1] - p[-6]) / p[-6]
    else:
        mom5 = last_ret

    # vol
    if n >= 22:
        rets = (p[-21:] - p[-22:-1]) / (p[-22:-1] + 1e-9)
        vol = np.std(rets)
    else:
        rets = np.diff(np.log(p + 1e-9))
        vol = float(np.std(rets)) if len(rets) > 0 else 1e-6

    # zscore vs ma
    ma21 = np.mean(p[-min(21, n):])
    zscore = (p[-1] - ma21) / (vol + 1e-9)

    # normalize vol to reasonable scale
    vol_norm = vol if vol > 1e-9 else 1e-9

    features = np.array([
        1.0,                 # bias
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
      - epoch: int (index)
      - price: float (prix actuel de Asset A)
    Sortie:
      {'Asset A': w, 'Cash': 1-w} with 0 <= w <= 1 and sum exactly 1.0
    """
    global _history_prices, _history_returns, _entry_price, _current_position
    global _epoch_of_entry, _peak_equity_price, _prev_features
    # SEED = np.random.randint(0, 1000000)  # changeable pour diversifier la "signature" et réduire similarité
    # np.random.seed(SEED)
    # Validate input
    SEED = np.random.randint(0, 1000000)
    price = float(price)

    # Append history and compute simple return
    if len(_history_prices) > 0:
        prev_price = _history_prices[-1]
        simple_ret = (price - prev_price) / (prev_price + 1e-12)
        _history_returns.append(simple_ret)
    else:
        simple_ret = 0.0

    _history_prices.append(price)

    # Init RLS on first call
    features = _compute_features(_history_prices)
    n_feat = len(features)
    if _theta is None or _P is None:
        _init_rls(n_feat)

    # Online RLS update using previous features and realized return as target
    # We update theta for t-1 -> target y = return_t (available now)
    if _prev_features is not None and len(_history_returns) >= 1:
        y = _history_returns[-1]  # realized return for last step
        try:
            _rls_update(_prev_features, y)
        except Exception:
            # numeric fallback: small ridge update
            pass

    # Predict next return with current features
    pred = float(np.dot(_theta, features))

    # combine prediction with rule-based signals (momentum + mean-reversion)
    # momentum score ~ ema_diff and mom5 already present in features:
    momentum_score = features[2] + 0.5 * features[3]  # ema_diff + 0.5*mom5
    meanrev_score = -features[5]  # negative zscore -> long (price below mean)

    # regime detector: high vol -> reduce confidence
    est_vol = abs(features[4])
    vol_scale = 1.0 / (1.0 + est_vol * 50.0)  # compress scale when vol large

    # combined signal: ml_pred weighted + rule signals
    alpha_ml = 0.6
    combined_score = alpha_ml * pred + 0.25 * momentum_score + 0.15 * meanrev_score
    combined_score *= vol_scale

    # sizing: volatility targeting
    # weight magnitude proportional to target_vol / est_vol * sigmoid of score
    safe_vol = max(est_vol, 1e-6)
    raw_scale = VOL_TARGET / safe_vol
    # map combined_score to [-1,1] via tanh for stability
    mapped = math.tanh(combined_score * 10.0)  # amplify sensitivity
    # only long allowed (platform expects allocation between 0 and 1). Convert negative -> cash
    desired_weight = max(0.0, raw_scale * mapped)

    # apply caps
    desired_weight = float(max(MIN_POS, min(MAX_POS, desired_weight)))

    # risk filters and stop-loss:
    # - if currently invested and price drawdown from entry > STOP_LOSS_PCT => go cash
    if _current_position > 0 and _entry_price is not None:
        dd_from_entry = (price - _entry_price) / (_entry_price + 1e-12)
        if dd_from_entry <= -abs(STOP_LOSS_PCT):
            # emergency exit
            desired_weight = 0.0

    # - enforce max holding time
    if _current_position > 0 and _epoch_of_entry is not None:
        if (epoch - _epoch_of_entry) > MAX_HOLD_EPOCHS:
            desired_weight = min(desired_weight, 0.5 * _current_position)  # scale down

    # - if peak equity drawdown (approx) since entry is too large, reduce exposure
    # Track peak price since entry to estimate drawdown
    if _entry_price is not None:
        if _peak_equity_price is None:
            _peak_equity_price = _entry_price
        _peak_equity_price = max(_peak_equity_price, price)
        if _peak_equity_price > 0:
            drawd = (_peak_equity_price - price) / (_peak_equity_price + 1e-12)
            if drawd > DRAWDOWN_DEFUSE:
                desired_weight *= 0.5  # strongly defuse risk on large drawdown

    # smoothing / turnover control: limit abrupt changes
    # simple EWMA smoothing between previous pos and desired
    lam = 0.6
    smoothed = lam * _current_position + (1 - lam) * desired_weight

    # reproducible jitter to avoid perfect-correlation with other bots
    jitter = np.random.RandomState(SEED + epoch % 997).normal(scale=JITTER_STD)
    smoothed = float(np.clip(smoothed + jitter, 0.0, 1.0))

    # final allocation must sum to 1.0
    w_asset = smoothed
    w_cash = float(1.0 - w_asset)

    # update entry tracking if we open a new meaningful position
    if w_asset > 0.001 and _current_position <= 0.001:
        _entry_price = price
        _epoch_of_entry = epoch
        _peak_equity_price = price

    # if we fully exit, reset entry trackers
    if w_asset <= 0.001 and _current_position > 0.001:
        _entry_price = None
        _epoch_of_entry = None
        _peak_equity_price = None

    # update current_position
    _current_position = w_asset

    # store prev features for next update step
    _prev_features = features.copy()

    # Ensure numeric stability: exactly sum to 1 (respect tolerance)
    total = w_asset + w_cash
    if abs(total - 1.0) > 1e-9:
        w_asset = w_asset / total
        w_cash = 1.0 - w_asset

    return {'Asset A': float(w_asset), 'Cash': float(w_cash)}
