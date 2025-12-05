# bot_trade.py
"""
Ramify Hackathon - Phase 1
Stratégie "Option A" : Regime-switch Hybrid + Robust Mean-Reversion

Idée :
- On garde ton meilleur modèle "hybrid" : RLS (online linear) + trend + léger mean-reversion.
- Mais ce modèle se fait démonter sur asset_a_train (marché range / choppy).
- On ajoute une brique de mean-reversion ROBUSTE (médiane + MAD) pour les régimes de range.
- On détecte le régime avec un simple signal de trend (EMA_short vs EMA_long).

Pipeline :
1) Features (returns, EMA diff, momentum, vol, z-score).
2) RLS online pour prédire le prochain retour.
3) Détection de régime :
   - EMA_diff > REGIME_TREND_THRESHOLD  -> régime "trend"
   - sinon                               -> régime "range"
4) Si trend   -> score = hybrid (RLS + momentum + tiny mean-rev).
   Si range   -> score = mean-reversion MAD (prix vs médiane, robuste aux outliers).
5) Position sizing basé sur score + target vol + risk filters.

Tout est pure Python, sans numpy.
"""

import math
import random
from collections import deque

# ============================================================================
#  HYPERPARAMS / CONFIG
# ============================================================================

# ---- Position sizing / agressivité ----
VOL_TARGET = 0.09          # risk budget par pas de temps (plus haut => plus agressif)
MAX_POS = 1.0
MIN_POS = 0.0

# ---- Fenêtres de trend (features) ----
EMA_SHORT = 7              # EMA rapide (sensibilité trend court terme)
EMA_LONG = 28              # EMA lente (trend plus lissé)

# ---- RLS (online regression linéaire) ----
RLS_DELTA = 0.995          # facteur d'oubli (proche de 1 => mémoire longue)
RLS_P0 = 5e3               # covariance initiale (confiance faible au début)
P_CLAMP_MAX = 5e7          # clamp numérique sur la matrice P

# ---- Mean-reversion robuste (MAD) ----
MR_WINDOW = 25             # taille de fenêtre pour médiane + MAD
MR_STRENGTH = 1.0          # intensité de la mean-reversion en régime range

# ---- Détection de régime ----
REGIME_TREND_THRESHOLD = 0.0  # seuil sur EMA_diff normalisée pour dire "trend"
REGIME_MIN_HISTORY = 30       # nombre mini de points avant de faire confiance au régime

# ---- Risk management ----
STOP_LOSS_PCT = 0.18       # stop-loss relatif au prix d'entrée (ex: -18%)
MAX_HOLD_EPOCHS = 200      # durée max de détention avant réduction forcée
DRAWDOWN_DEFUSE = 0.25     # réduction de taille si drawdown local > 25% depuis le pic

# ---- Bruit / stabilité ----
JITTER_STD = 0.0005        # petit jitter sur la position pour éviter les coin-flips
LAM = 0.4                  # lissage expo : pos_new = LAM*pos_old + (1-LAM)*desired

# ---- Historique ----
HISTORY_MAX_LEN = 300      # longueur max de l'historique de prix


# ============================================================================
#  GLOBAL STATE
# ============================================================================

_history_prices = deque(maxlen=HISTORY_MAX_LEN)
_history_returns = deque(maxlen=HISTORY_MAX_LEN)

_entry_price = None
_current_position = 0.0
_epoch_of_entry = None
_peak_equity_price = None

_prev_features = None

_theta = None   # coefficients RLS
_P = None       # matrice de covariance RLS

_rng = random.Random(123456)  # RNG global pour le jitter (reproductible)


# ============================================================================
#  HELPERS MATH (pure Python)
# ============================================================================

def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _matrix_vector_mul(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _matrix_clamp(M, limit):
    for i in range(len(M)):
        row = M[i]
        for j in range(len(row)):
            if row[j] > limit:
                row[j] = limit
            elif row[j] < -limit:
                row[j] = -limit


def _safe_tanh(x, factor=10.0):
    """tanh avec scaling d'entrée et protection overflow."""
    v = x * factor
    if v > 50.0:
        v = 50.0
    elif v < -50.0:
        v = -50.0
    return math.tanh(v)


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


def _std(xs):
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / float(n)
    return math.sqrt(var)


def _median(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    xs_sorted = sorted(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs_sorted[mid]
    else:
        return 0.5 * (xs_sorted[mid - 1] + xs_sorted[mid])


def _mad(xs, med=None):
    """Median Absolute Deviation."""
    if not xs:
        return 0.0
    if med is None:
        med = _median(xs)
    devs = [abs(x - med) for x in xs]
    return _median(devs)


# ============================================================================
#  RLS HELPERS
# ============================================================================

def _init_rls(n_features: int):
    """Initialisation de theta et P pour RLS."""
    global _theta, _P
    _theta = [0.0 for _ in range(n_features)]
    _P = [[0.0 for _ in range(n_features)] for _ in range(n_features)]
    for i in range(n_features):
        _P[i][i] = RLS_P0


def _rls_update(x, y):
    """
    Recursive Least Squares avec facteur d'oubli.

    x : liste de features (longueur n)
    y : retour réalisé (float)
    """
    global _theta, _P

    n = len(x)

    # Px = P * x
    Px = _matrix_vector_mul(_P, x)

    # denom = delta + x^T P x
    xPx = sum(x[i] * Px[i] for i in range(n))
    denom = RLS_DELTA + xPx
    if not math.isfinite(denom) or abs(denom) < 1e-12:
        denom = 1e-12

    # K = Px / denom
    K = [px / denom for px in Px]

    # erreur
    y_hat = _dot(_theta, x)
    err = y - y_hat

    # update theta
    _theta = [_theta[i] + K[i] * err for i in range(n)]

    # v = x^T P
    n_cols = n
    v = [0.0] * n_cols
    for j in range(n_cols):
        s = 0.0
        for i in range(n):
            s += x[i] * _P[i][j]
        v[j] = s

    # P_new[i][j] = (P[i][j] - K[i]*v[j]) / delta
    for i in range(n):
        Ki = K[i]
        row = _P[i]
        for j in range(n_cols):
            row[j] = (row[j] - Ki * v[j]) / RLS_DELTA

    _matrix_clamp(_P, P_CLAMP_MAX)


# ============================================================================
#  FEATURE ENGINEERING
# ============================================================================

def _compute_features(prices_deque):
    """
    Features:
      [0] 1.0       : biais
      [1] last_ret  : dernier retour simple
      [2] ema_diff  : (EMA_short - EMA_long) / prix
      [3] mom5      : momentum sur 5 pas
      [4] vol_norm  : estimation de vol locale
      [5] zscore    : (prix - moyenne_21) / vol
    """
    p = [float(x) for x in prices_deque]
    n = len(p)

    # last_ret
    if n >= 2:
        last_ret = (p[-1] - p[-2]) / (p[-2] + 1e-12)
    else:
        last_ret = 0.0

    # EMAs
    ema_s = _ema_from_array(p, EMA_SHORT) if n >= 1 else 0.0
    ema_l = _ema_from_array(p, EMA_LONG) if n >= 1 else 0.0
    if n >= 1:
        ema_diff = (ema_s - ema_l) / (p[-1] + 1e-9)
    else:
        ema_diff = 0.0

    # momentum sur 5 pas
    if n >= 6 and p[-6] != 0.0:
        mom5 = (p[-1] - p[-6]) / (p[-6] + 1e-12)
    else:
        mom5 = last_ret

    # vol locale (simple returns sur ~22 points)
    if n >= 2:
        rets = []
        start = max(0, n - 22)
        for k in range(start, n - 1):
            r = (p[k + 1] - p[k]) / (p[k] + 1e-12)
            rets.append(r)
    else:
        rets = [0.0]

    vol = _std(rets)
    if not math.isfinite(vol) or vol <= 0.0:
        vol = 1e-6
    vol_norm = max(vol, 1e-9)

    # z-score vs MA(21)
    if n >= 1:
        window = min(21, n)
        ma21 = _mean(p[-window:])
        zscore = (p[-1] - ma21) / (vol + 1e-9)
    else:
        zscore = 0.0

    return [
        1.0,
        last_ret,
        ema_diff,
        mom5,
        vol_norm,
        zscore,
    ]


# ============================================================================
#  MAIN DECISION FUNCTION
# ============================================================================

def make_decision(epoch: int, price: float):
    """
    Appelée par la plateforme à chaque pas de temps.

    Retourne :
        {'Asset A': w, 'Cash': 1-w} avec w ∈ [0, 1].
    """
    global _history_prices, _history_returns
    global _entry_price, _current_position, _epoch_of_entry, _peak_equity_price
    global _prev_features, _theta, _P

    price = float(price)

    # ----------------------------------------------------------------------
    # 1) Mise à jour de l'historique
    # ----------------------------------------------------------------------
    if len(_history_prices) > 0:
        prev_price = _history_prices[-1]
        simple_ret = (price - prev_price) / (prev_price + 1e-12)
        _history_returns.append(simple_ret)
    else:
        simple_ret = 0.0

    _history_prices.append(price)

    # ----------------------------------------------------------------------
    # 2) Features & RLS
    # ----------------------------------------------------------------------
    features = _compute_features(_history_prices)
    n_feat = len(features)

    if _theta is None or _P is None:
        _init_rls(n_feat)

    # RLS update (features(t-1) -> return(t))
    if _prev_features is not None and len(_history_returns) >= 1:
        y = float(_history_returns[-1])
        try:
            _rls_update(_prev_features, y)
        except Exception:
            # fallback très léger en cas de soucis numérique
            _theta = [t * 0.999 + 1e-6 for t in _theta]

    # Décomposition des features
    ema_diff = features[2]
    mom5 = features[3]
    zscore = features[5]
    est_vol = abs(features[4])

    momentum_score = ema_diff + 0.5 * mom5
    meanrev_score_z = -zscore   # version z-score (non robuste)

    # ----------------------------------------------------------------------
    # 3) Regime classification (trend / range / shock)
    # ----------------------------------------------------------------------
    # Trend score
    T = ema_diff

    # Volatility score
    V = est_vol

    # Momentum consistency
    if mom5 == 0:
        M = True
    else:
        M = (mom5 > 0 and ema_diff > 0) or (mom5 < 0 and ema_diff < 0)

    # Thresholds (tunable)
    T_threshold = 0.0005      # trend if slope stronger than this
    VOL_range_threshold = 0.015
    VOL_shock_threshold = 0.05

    # Classify
    if V > VOL_shock_threshold:
        regime = "shock"
    elif T > T_threshold and M:
        regime = "trend"
    elif T > 0 and V < VOL_range_threshold:
        regime = "trend"
    else:
        regime = "range"

    # ----------------------------------------------------------------------
    # 4) Combined score based on regime
    # ----------------------------------------------------------------------
    if regime == "trend":
        # --- Trend-mode hybrid ---
        pred = _dot(_theta, features)
        vol_scale = 1.0 / (1.0 + est_vol * 8.0)

        alpha_ml = 0.80
        w_mom = 0.18
        w_mr = 0.02

        combined_score = (
            alpha_ml * pred +
            w_mom * momentum_score +
            w_mr * (-zscore)
        ) * vol_scale

    elif regime == "range":
        # --- Mean-reversion robuste ---
        p = list(_history_prices)
        if len(p) >= MR_WINDOW:
            window = p[-MR_WINDOW:]
        else:
            window = p

        med = _median(window)
        madv = _mad(window, med)
        if madv <= 0.0 or not math.isfinite(madv):
            madv = 1e-6

        z_med = (price - med) / madv     # robust z-score
        mr_core = -z_med * MR_STRENGTH   # mean-reversion long-only
        mr_core += 0.05 * momentum_score # small trend assist

        vol_scale = 1.0 / (1.0 + est_vol * 5.0)
        combined_score = mr_core * vol_scale

    else:
        # --- Shock regime ---
        # Reduce aggressiveness but don't exit completely
        SHOCK_STRENGTH = 0.10   # very small position in shocks
        fallback = 0.1 * momentum_score
        combined_score = SHOCK_STRENGTH * fallback
        # no vol scale here: it's already shock mode

    # ----------------------------------------------------------------------
    # 5) Mapping score -> desired weight [0,1]
    # ----------------------------------------------------------------------
    safe_vol = max(est_vol, 1e-6)
    raw_scale = VOL_TARGET / safe_vol

    # tanh pour limiter les extrêmes, mais garder le signe
    mapped = _safe_tanh(combined_score, factor=10.0)

    # Long-only : on supprime les signaux négatifs
    desired_weight = max(0.0, raw_scale * mapped)

    if desired_weight < MIN_POS:
        desired_weight = MIN_POS
    elif desired_weight > MAX_POS:
        desired_weight = MAX_POS

    # ----------------------------------------------------------------------
    # 6) Filtres de risque (stop-loss, max holding, drawdown)
    # ----------------------------------------------------------------------
    # a) Stop-loss vs prix d'entrée
    if _current_position > 0.0 and _entry_price is not None:
        dd_from_entry = (price - _entry_price) / (_entry_price + 1e-12)
        if dd_from_entry <= -abs(STOP_LOSS_PCT):
            desired_weight = 0.0

    # b) Max holding
    if _current_position > 0.0 and _epoch_of_entry is not None:
        if (epoch - _epoch_of_entry) > MAX_HOLD_EPOCHS:
            desired_weight = min(desired_weight, 0.5 * _current_position)

    # c) Drawdown local vs pic depuis l'entrée
    if _entry_price is not None:
        if _peak_equity_price is None:
            _peak_equity_price = _entry_price
        if price > _peak_equity_price:
            _peak_equity_price = price

        if _peak_equity_price > 0.0:
            drawd = (_peak_equity_price - price) / (_peak_equity_price + 1e-12)
            if drawd > DRAWDOWN_DEFUSE:
                desired_weight *= 0.5

    # ----------------------------------------------------------------------
    # 7) Lissage + jitter
    # ----------------------------------------------------------------------
    smoothed = LAM * _current_position + (1.0 - LAM) * desired_weight

    if JITTER_STD > 0.0:
        jitter = _rng.gauss(0.0, JITTER_STD)
    else:
        jitter = 0.0

    smoothed = max(0.0, min(1.0, smoothed + jitter))

    w_asset = smoothed
    w_cash = 1.0 - w_asset

    # ----------------------------------------------------------------------
    # 8) Mise à jour de l'état de position
    # ----------------------------------------------------------------------
    if w_asset > 0.001 and _current_position <= 0.001:
        # nouvelle entrée
        _entry_price = price
        _epoch_of_entry = epoch
        _peak_equity_price = price

    if w_asset <= 0.001 and _current_position > 0.001:
        # sortie complète
        _entry_price = None
        _epoch_of_entry = None
        _peak_equity_price = None

    _current_position = w_asset
    _prev_features = features[:]  # copie

    # safety : renormalisation
    total = w_asset + w_cash
    if abs(total - 1.0) > 1e-9:
        w_asset = w_asset / total
        w_cash = 1.0 - w_asset

    return {"Asset A": float(w_asset), "Cash": float(w_cash)}
