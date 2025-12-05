# bot_trade.py

from typing import List, Dict
import math

# On garde un historique global, comme dans ta version actuelle
history: List[float] = []  # liste des prix successifs
_prev_weight: float = 0.5  # dernier poids utilisé


def _pct_change(prices: List[float]) -> List[float]:
    """Simple list-based pct_change, sans numpy/pandas."""
    if len(prices) < 2:
        return []
    return [(prices[i] / prices[i - 1] - 1.0) for i in range(1, len(prices))]


def _moving_average(prices: List[float], window: int) -> float:
    if len(prices) < window:
        if not prices:
            return 0.0
        # moyenne sur tout l'historique si pas assez de points
        return sum(prices) / len(prices)
    window_slice = prices[-window:]
    return sum(window_slice) / len(window_slice)


def _std(values: List[float]) -> float:
    """Std naïve, évite les dépendances externes."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return var ** 0.5


def make_decision(epoch: int, price: float) -> Dict[str, float]:
    """
    Triple K V1.1

    - Cœur = ta V1 (MA 10/50, vol 20, sigmoïde sur trend, vol target 2%).
    - Crash filter léger (dernier retour < -3% ou > 3%).
    - Smoothing des poids pour améliorer un peu le Sharpe.
    - Reset propre quand epoch == 0 (utile pour tes runs locaux).
    """

    global history, _prev_weight

    # Reset automatique quand on recommence un nouveau jeu de données
    if epoch == 0:
        history = []
        _prev_weight = 0.5

    # On log seulement le prix; l'epoch n'est pas nécessaire pour la strat
    history.append(price)

    # Phase de bootstrapping: les 20 premiers points, on reste prudent et simple
    if len(history) < 20:
        _prev_weight = 0.5
        return {"Asset B": 0.5, "Cash": 0.5}

    # Paramètres (identiques à ta V1)
    SHORT_WIN = 10
    LONG_WIN = 50
    VOL_WIN = 20

    short_ma = _moving_average(history, SHORT_WIN)
    long_ma = _moving_average(history, LONG_WIN)

    # Évite division par zéro
    if long_ma == 0:
        long_ma = short_ma if short_ma != 0 else price

    trend_raw = short_ma - long_ma
    trend_score = trend_raw / long_ma  # échelle relative (~pourcent)

    # Volatilité sur VOL_WIN derniers retours
    returns = _pct_change(history)
    recent_returns = returns[-VOL_WIN:] if len(returns) >= VOL_WIN else returns
    vol = _std(recent_returns)
    if vol <= 0:
        vol = 1e-6  # évite le 0

    # 1) Base weight via fonction sigmoïde sur le trend_score
    SCALE_TREND = 20.0  # comme ta V1
    x = trend_score * SCALE_TREND
    base_weight = 1.0 / (1.0 + math.exp(-x))

    # (optionnel mais très léger) :
    # si trend quasi nul, on revient vers neutre plutôt que de suivre le bruit
    if abs(trend_score) < 0.0005:  # ~0.05% d'écart seulement
        base_weight = 0.5

    # 2) Ajustement par la volatilité (vol targeting)
    TARGET_VOL = 0.02  # identique à ta V1
    vol_adjust = TARGET_VOL / vol
    # même bornes que V1 (0.5–2.0) pour ne pas changer le profil de risque
    vol_adjust = max(0.5, min(2.0, vol_adjust))

    weight = base_weight * vol_adjust

    # 3) Crash filter: gros retour négatif => on coupe une partie du risque
    last_ret = returns[-1] if returns else 0.0
    if last_ret < -0.03:       # -3% en un jour => on réduit
        weight *= 0.5
    elif last_ret > 0.03 and trend_score > 0:
        # gros up et tendance positive => on autorise un peu plus de risque
        weight *= 1.1

    # 4) Contraintes de risque globales
    MIN_RISK = 0.05  # ne jamais être totalement hors marché
    MAX_RISK = 0.95  # ne jamais être all-in
    weight = max(MIN_RISK, min(MAX_RISK, weight))

    # 5) Lissage des positions (nouveau)
    #    -> limite les changements de poids trop violents,
    #       ce qui typiquement améliore Sharpe / réduit le churn.
    SMOOTH_ALPHA = 0.3  # 0 = très lissé, 1 = pas de lissage
    weight = _prev_weight + SMOOTH_ALPHA * (weight - _prev_weight)
    _prev_weight = weight

    asset_b = float(weight)
    cash = 1.0 - asset_b

    return {"Asset B": asset_b, "Cash": cash}
