# bot_trade.py
"""
Ramify Hackathon - Phase 1
"Leaderboard-style" bot: ultra-aggressive trend / breakout follower.

Key ideas:
- Long-only, but we flip hard between 0 and 1.
- 1.0 allocation when we detect a strong uptrend.
- 0.0 otherwise.
- No volatility scaling, minimal risk controls (only basic sanity).
- Uses three main signals:
    * EMA slope (EMA_SHORT vs EMA_LONG)
    * Recent cumulative return
    * Price breakout vs rolling max

If the underlying path has persistent uptrends (which the top scores strongly suggest),
this style of bot can explode PnL compared to smoother strategies.
"""

import math
from collections import deque

# ============================================================================
#  HYPERPARAMS
# ============================================================================

# EMA windows
EMA_SHORT = 7          # fast trend
EMA_LONG = 35         # slower trend (slightly longer than before)

# Lookback windows
CUMRET_WINDOW = 40    # window for cumulative return test
HIGH_WINDOW = 60      # window for rolling max breakout

# Thresholds for "trend" regime
EMA_DIFF_THRESHOLD = 0.0003     # minimum normalized EMA_diff to call it trend
CUMRET_THRESHOLD = 0.02         # min cumulated return over window (2%)
BREAKOUT_EPS = 0.001            # how close to rolling max to count as breakout

# Minimal risk sanity (we don't want this to get in the way too much)
MAX_HOLD_EPOCHS = 1000          # basically no limit, but keeps us safe from pathologies
STOP_LOSS_PCT = 0.6             # very loose: -60% vs entry

# Smoothing (small, but not zero)
LAM = 0.1                       # low smoothing: react fast to signal

# ============================================================================
#  GLOBAL STATE
# ============================================================================

_history_prices = deque()
_history_returns = deque()

_current_position = 0.0
_entry_price = None
_epoch_of_entry = None

# ============================================================================
#  HELPERS
# ============================================================================

def _ema_from_array(arr, span: int) -> float:
    n = len(arr)
    if n == 0:
        return 0.0
    alpha = 2.0 / (span + 1.0)
    s = float(arr[0])
    for k in range(1, n):
        s = alpha * float(arr[k]) + (1.0 - alpha) * s
    return s


def _mean(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    return sum(xs) / float(n)


# ============================================================================
#  MAIN DECISION FUNCTION
# ============================================================================

def make_decision(epoch: int, price: float):
    """
    Called by the platform each timestep.

    Returns a dict with weights:
        {"Asset A": w, "Cash": 1-w}  with w in [0, 1].
    """
    global _history_prices, _history_returns
    global _current_position, _entry_price, _epoch_of_entry

    price = float(price)

    # --- 1) Update history ---
    if _history_prices:
        prev_price = _history_prices[-1]
        ret = (price - prev_price) / (prev_price + 1e-12)
        _history_returns.append(ret)
    else:
        ret = 0.0

    _history_prices.append(price)

    n = len(_history_prices)

    # We need some history before doing anything smart
    if n < max(EMA_LONG + 2, CUMRET_WINDOW + 2, HIGH_WINDOW + 2):
        # warmup: small fixed exposure
        w_asset = 0.2
        _current_position = w_asset
        _entry_price = _entry_price or price
        _epoch_of_entry = _epoch_of_entry if _epoch_of_entry is not None else epoch
        return {"Asset A": w_asset, "Cash": 1.0 - w_asset}

    # --- 2) Compute key signals ---

    prices_list = list(_history_prices)

    # EMA-based trend slope
    ema_s = _ema_from_array(prices_list, EMA_SHORT)
    ema_l = _ema_from_array(prices_list, EMA_LONG)
    ema_diff = (ema_s - ema_l) / (price + 1e-9)  # normalized

    # Recent cumulative return
    rets_list = list(_history_returns)
    recent_rets = rets_list[-CUMRET_WINDOW:]
    cum_ret = 1.0
    for r in recent_rets:
        cum_ret *= (1.0 + r)
    cum_ret -= 1.0  # cumulative simple return over window

    # Rolling max breakout
    recent_prices_for_high = prices_list[-HIGH_WINDOW:]
    rolling_high = max(recent_prices_for_high)
    if rolling_high <= 0.0:
        breakout_score = 0.0
    else:
        # closeness to rolling high in [0, 1]
        breakout_score = (price - (rolling_high * (1.0 - BREAKOUT_EPS))) / (rolling_high * BREAKOUT_EPS + 1e-9)

    is_breakout = (price >= rolling_high * (1.0 - BREAKOUT_EPS))

    # --- 3) Build a composite bull score ---

    bull_score = 0.0

    # EMA slope strongly positive => trend
    if ema_diff > EMA_DIFF_THRESHOLD:
        bull_score += 1.0

    # Cumulative return positive enough => medium/long-term up move
    if cum_ret > CUMRET_THRESHOLD:
        bull_score += 1.0

    # Breakout near rolling high
    if is_breakout:
        bull_score += 1.0

    # Last immediate move in the right direction
    last_ret = rets_list[-1] if rets_list else 0.0
    if last_ret > 0.0:
        bull_score += 0.5

    # --- 4) Turn bull_score into a 0/1 desired weight ---

    # Very simple rule:
    #   if bull_score >= 2.0 -> FULL LONG
    #   else -> FLAT
    if bull_score >= 2.0:
        desired_weight = 1.0
    else:
        desired_weight = 0.0

    # --- 5) Minimal risk sanity (loose) ---

    # Track entry info when we go from flat -> long
    if desired_weight > 0.0 and _current_position <= 0.001:
        _entry_price = price
        _epoch_of_entry = epoch

    if _current_position > 0.0 and _entry_price is not None:
        dd_from_entry = (price - _entry_price) / (_entry_price + 1e-12)
        if dd_from_entry <= -abs(STOP_LOSS_PCT):
            # Massive loss since entry -> go flat
            desired_weight = 0.0

    if _current_position > 0.0 and _epoch_of_entry is not None:
        if (epoch - _epoch_of_entry) > MAX_HOLD_EPOCHS:
            # Just a small protection: reduce exposure if we've been in forever
            desired_weight = 0.0

    # --- 6) Small smoothing so we don't jitter on a single tick ---

    w_asset = LAM * _current_position + (1.0 - LAM) * desired_weight
    w_asset = max(0.0, min(1.0, w_asset))
    w_cash = 1.0 - w_asset

    _current_position = w_asset

    return {"Asset A": float(w_asset), "Cash": float(w_cash)}
