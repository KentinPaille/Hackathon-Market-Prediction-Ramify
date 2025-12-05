# ============================================================================
# ⚠️  ATTENTION: Ce fichier n’a pas de lien avec le développement de votre bot.
# ⚠️  Vous pouvez l’ignorer.
# ============================================================================

import math
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_prices(paths_prices: list[str]) -> pd.DataFrame:
    # Read prices
    prices_list = [
        pd.read_csv(path_prices, index_col=0) for path_prices in paths_prices
    ]
    prices = pd.concat(prices_list, axis=1)
    prices["Cash"] = 1
    return prices


def get_positions(path_positions: str) -> pd.DataFrame:
    # Read positions
    with open(file=path_positions, mode="rb") as f:
        positions_data = json.load(fp=f)
    positions = pd.DataFrame(positions_data).set_index("epoch")
    return positions


def compute_stats(
    pnl: pd.Series,
    positions: pd.DataFrame,
    trading_days: int = 252,
    var_alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compute a compact set of performance and trading metrics.

    Parameters
    ----------
    pnl : pd.Series
        Equity curve (e.g. starting at 1.0), one observation per trading day.
    positions : pd.DataFrame
        Portfolio positions (weights or exposures) indexed like `pnl`.
        Used for time-in-market and exposure-based metrics.
    trading_days : int
        Number of trading days per year (for annualization).
    var_alpha : float
        Tail probability for VaR / CVaR (e.g. 0.05 for 5%).

    Returns
    -------
    dict[str, Any]
        {
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "var_5",
            "cvar_5",
            "time_in_market",
            "avg_exposition_market",
            "exposure_timing_accuracy",
            "expected_value_per_trade",
        }
    """
    if pnl.isna().all():
        raise ValueError("pnl is empty or all NaN")

    pnl = pnl.dropna()
    if len(pnl) < 2:
        raise ValueError("Need at least 2 observations in pnl")

    if len(positions) != len(pnl):
        raise ValueError("pnl and positions must have the same length")
    if not pnl.index.equals(positions.index):
        raise ValueError("pnl and positions must share the same index")

    # ---------- returns ----------
    rets = pnl.pct_change().dropna()
    if rets.empty:
        raise ValueError("Cannot compute returns from pnl")

    n = len(rets)

    # cumulative return
    cumulative_return = pnl.iloc[-1] / pnl.iloc[0] - 1.0

    # geometric mean of simple returns -> annualized
    geom_daily = (1.0 + rets).prod() ** (1.0 / n) - 1.0
    annualized_return = (1.0 + geom_daily) ** trading_days - 1.0

    # annualized volatility of simple returns
    daily_std = rets.std(ddof=1)
    annualized_volatility = daily_std * math.sqrt(trading_days)

    # Sharpe ratio (no risk-free)
    sharpe_ratio = (
        annualized_return / annualized_volatility
        if annualized_volatility > 0
        else np.nan
    )

    # ---------- drawdowns ----------
    running_max = pnl.cummax()
    drawdown = pnl / running_max - 1.0
    max_drawdown = drawdown.min()

    # ---------- VaR / CVaR (annualized, via sqrt(T) scaling) ----------
    var_daily = rets.quantile(var_alpha)
    cvar_daily = rets[rets <= var_daily].mean()

    var_5 = var_daily * math.sqrt(trading_days)
    cvar_5 = cvar_daily * math.sqrt(trading_days)

    # ---------- trading metrics ----------
    # Time in market: any non-zero total exposure
    total_exposure = positions.abs().sum(axis=1)
    time_in_market = (total_exposure > 0).mean()

    # Average exposure to market: only non-CASH columns
    non_cash_cols = [c for c in positions.columns if str(c).upper() != "CASH"]
    if non_cash_cols:
        market_exposure = positions[non_cash_cols].abs().sum(axis=1)
        avg_exposition_market = market_exposure.mean()
    else:
        market_exposure = pd.Series(0.0, index=positions.index)
        avg_exposition_market = 0.0

    # Exposure Timing Accuracy:
    # treat each change in total market exposure as an implicit prediction
    # of the sign of the next return.
    exp_current = market_exposure.loc[rets.index]
    exp_prev = exp_current.shift(1)

    exposure_change = (exp_current != exp_prev) & exp_prev.notna()
    n_changes = int(exposure_change.sum())

    if n_changes > 0:
        delta_exp = exp_current - exp_prev

        # "Correct" decisions: increase exposure when return > 0,
        # decrease exposure when return < 0.
        successes = (
            ((delta_exp > 0) & (rets > 0)) | ((delta_exp < 0) & (rets < 0))
        ) & exposure_change

        exposure_timing_accuracy = successes.sum() / n_changes

        # Expected value per trade: mean return on change days
        expected_value_per_trade = rets[exposure_change].mean()
    else:
        exposure_timing_accuracy = np.nan
        expected_value_per_trade = np.nan

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "var_5": var_5,
        "cvar_5": cvar_5,
        "time_in_market": time_in_market,
        "avg_exposition_market": avg_exposition_market,
        "exposure_timing_accuracy": exposure_timing_accuracy,
        "expected_value_per_trade": expected_value_per_trade,
    }


def backtest(
    prices: pd.DataFrame,
    positions: pd.DataFrame,
    initial_capital: float = 1.0,
    transaction_fees: float = 0.0001,
):
    """
    Run a simple backtest with periodic rebalancing.

    Parameters
    ----------
    prices : DataFrame
        Asset prices indexed over time.
    positions : DataFrame
        Target portfolio weights at each date (same shape/columns as `prices`).
        By convention, we assume that positions at time t have been created with info up until time t-1.
    initial_capital : float
        Starting capital.
    transaction_fees : float
        Proportional transaction cost applied to traded notional.

    Returns
    -------
    dict
        Contains:
        - "pnl": cumulative returns series
        - "stats": output of `compute_stats`
    """
    # Ensure column alignment
    prices_positions_diff = set(prices.columns.difference(positions.columns))
    if prices_positions_diff:
        raise ValueError(
            f"Columns in positions but not in prices: {prices_positions_diff}"
        )

    positions_prices_diff = set(positions.columns.difference(prices.columns))
    if positions_prices_diff:
        raise ValueError(f"Columns in prices not in positions: {positions_prices_diff}")

    # Ensure same time dimension
    if len(prices) != len(positions):
        raise ValueError(
            f"Prices and positions not the same length: got {len(prices)=} and {len(positions)=}"
        )

    # Number of units per asset
    nb_units = pd.DataFrame(None, columns=prices.columns, index=prices.index)

    for i in nb_units.index:
        current_prices = prices.loc[i]
        target_weights = positions.loc[i]

        if i == 0:
            # Initial allocation
            nb_units.loc[i] = (
                (target_weights * initial_capital)
                / current_prices
                * (1 - transaction_fees)
            )
        else:
            prev_nb_units = nb_units.loc[i - 1]

            # Portfolio value before rebalancing
            capital_before_rebalance = (prev_nb_units * current_prices).sum()

            # Ideal holdings before transaction costs
            ideal_nb_units = (
                target_weights * capital_before_rebalance
            ) / current_prices

            # TC applied to traded notional
            transaction_costs = (
                np.abs((ideal_nb_units - prev_nb_units) * current_prices).sum()
                * transaction_fees
            )
            capital_after_tc = capital_before_rebalance - transaction_costs

            # Actual holdings after transaction costs
            actual_nb_units = (target_weights * capital_after_tc) / current_prices
            nb_units.loc[i] = actual_nb_units

    # Capital path
    capital_evolution = (nb_units * prices).sum(axis=1)
    capital_evolution.iloc[0] = initial_capital

    # Returns and cumulative PnL
    returns = capital_evolution.pct_change().fillna(0)
    pnl = (1 + returns).cumprod()

    return {"pnl": pnl, "stats": compute_stats(pnl=pnl, positions=positions)}


def get_base_score(
    sharpe: float,
    cum_ret: float,
    mdd: float,
    initial_capital: float = 1000,
    sharpe_max: float = 2.0,
    cum_ret_max: float = 5.0,
    mdd_min: float = -1,
    sharpe_w: float = 0.3,
    pnl_w: float = 0.6,
    mdd_w: float = 0.1,
):
    assert np.isclose(
        sharpe_w + pnl_w + mdd_w, 1
    ), f"[BASE SCORE] Sum of weights must be equal to 1, got {sharpe_w + pnl_w + mdd_w}"

    pnl = initial_capital * cum_ret
    pnl_max = initial_capital * cum_ret_max

    sharpe_score = max(sharpe, 0) / sharpe_max
    pnl_score = max(pnl, 0) / pnl_max
    mdd_score = 1 - mdd / mdd_min

    return {
        "sharpe_score": sharpe_score,
        "pnl_score": pnl_score,
        "mdd_score": mdd_score,
        "base_score": sharpe_w * sharpe_score + pnl_w * pnl_score + mdd_w * mdd_score,
    }


def get_local_score(
    prices: pd.DataFrame, positions: pd.DataFrame, initial_capital: float = 1_000
) -> dict[str, dict]:

    # Backtest
    backtest_results = backtest(
        prices=prices, positions=positions, initial_capital=initial_capital
    )

    pnl = backtest_results["pnl"]
    stats = backtest_results["stats"]
    scores = get_base_score(
        sharpe=stats["sharpe_ratio"],
        cum_ret=stats["cumulative_return"],
        mdd=stats["max_drawdown"],
        initial_capital=initial_capital,
    )

    return {
        "pnl": pnl.to_dict(),
        "stats": stats,
        "scores": scores,
    }


# ============================================================================
# affichage
# ============================================================================

def show_result(local_score: dict, is_show_graph: bool = False):
    # Affichage formaté de tous les scores
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS")
    print("=" * 70)

    print("\n🎯 SCORES:")
    print("-" * 70)
    scores = local_score["scores"]
    print(f"  Sharpe Score:     {scores['sharpe_score']:.4f}")
    print(f"  PnL Score:        {scores['pnl_score']:.4f}")
    print(f"  Max Drawdown Score: {scores['mdd_score']:.4f}")
    print(f"  ⭐ Base Score:     {scores['base_score']:.4f}")
  
    print("\n" + "=" * 70)

    print("🎯 Performance:")
    print(f"  Brut PnL:        { local_score["stats"]["cumulative_return"]*100:.2f}%")

    if is_show_graph:
        print("\033[94mune page graphique va s'ouvrir pour vous montrer les résultats du pnl\033[0m")
        # Graphique du PnL avec matplotlib
        pnl_dict = local_score["pnl"]
        pnl_series = pd.Series(pnl_dict).sort_index()

        plt.figure(figsize=(12, 6))
        plt.plot(pnl_series.index, pnl_series.values, linewidth=2, color='#2E86AB')
        plt.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Capital initial')
        plt.fill_between(pnl_series.index, pnl_series.values, 1.0, 
                        where=(pnl_series.values >= 1.0), alpha=0.3, color='green', label='Profit')
        plt.fill_between(pnl_series.index, pnl_series.values, 1.0, 
                        where=(pnl_series.values < 1.0), alpha=0.3, color='red', label='Perte')
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('PnL (Multiplicateur)', fontsize=12, fontweight='bold')
        plt.title('Évolution du PnL au fil du temps', fontsize=14, fontweight='bold', pad=20)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()