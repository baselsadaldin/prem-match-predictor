"""Does the best available price beat Pinnacle's margin-free probabilities?

No model is involved. Pinnacle's early odds (PSH/PSD/PSA), with the margin removed proportionally,
are taken as the true probabilities. A 1-unit bet is placed on every outcome whose best market
price beats them by more than a threshold.

Best-price columns differ by era:
- 2013/14-2018/19: BbMx* (Betbrain maximum), which may include stale or outlier prices
- 2019/20 onwards: Max* (market maximum, collected with the other early odds)
Either maximum may include exchange prices before commission, so returns are optimistic.
"""
import numpy as np
import pandas as pd

from prepare_data import DATA_DIR, SEASONS

OUTCOMES = ["H", "D", "A"]
PINNACLE = ["PSH", "PSD", "PSA"]
PINNACLE_CLOSING = ["PSCH", "PSCD", "PSCA"]
THRESHOLDS = [0.0, 0.01, 0.02, 0.03, 0.05]


def fair_probabilities(odds):
    implied = 1 / odds
    return implied / implied.sum(axis=1, keepdims=True)


def season_bets(season, threshold, league="E0", max_missing=0):
    """Bets for one season; up to `max_missing` matches without a full set of odds are skipped."""
    matches = pd.read_csv(DATA_DIR / f"{league}_{season}.csv").dropna(subset=["FTR"])
    best = ["MaxH", "MaxD", "MaxA"] if "MaxH" in matches else ["BbMxH", "BbMxD", "BbMxA"]
    missing = matches[PINNACLE + PINNACLE_CLOSING + best].isna().any(axis=1)
    if missing.sum() > max_missing:
        raise ValueError(f"Season {season} has missing odds")
    matches = matches.loc[~missing]
    prices = matches[best].to_numpy()
    edge = prices * fair_probabilities(matches[PINNACLE].to_numpy()) - 1
    closing_fair = fair_probabilities(matches[PINNACLE_CLOSING].to_numpy())
    rows, outcome = np.nonzero(edge > threshold)
    won = matches["FTR"].to_numpy()[rows] == np.asarray(OUTCOMES)[outcome]
    return pd.DataFrame({
        "Season": season,
        "Era": "Max" if best[0] == "MaxH" else "BbMx",
        "Odds": prices[rows, outcome],
        "Edge": edge[rows, outcome],
        # Expected value at Pinnacle's closing probabilities: positive means the price beat the close.
        "ClosingValue": prices[rows, outcome] * closing_fair[rows, outcome] - 1,
        "Profit": np.where(won, prices[rows, outcome] - 1, -1.0),
    })


def summarise(bets, rng):
    if bets.empty:
        return "no bets"
    profit = bets["Profit"].to_numpy()
    boot = profit[rng.integers(0, len(profit), (5000, len(profit)))].mean(axis=1)
    return (
        f"{len(bets):>5} bets  avg odds {bets['Odds'].mean():5.2f}  avg edge {bets['Edge'].mean():+.1%}"
        f"  ROI {profit.mean():+6.1%} [{np.percentile(boot, 2.5):+.1%}, {np.percentile(boot, 97.5):+.1%}]"
        f"  CLV {bets['ClosingValue'].mean():+.1%}"
    )


def main():
    rng = np.random.default_rng(0)
    seasons = SEASONS
    for threshold in THRESHOLDS:
        bets = pd.concat([season_bets(season, threshold) for season in seasons], ignore_index=True)
        print(f"\nBest price beats Pinnacle fair odds by more than {threshold:.0%}:")
        for era, part in bets.groupby("Era", sort=False):
            span = f"{part['Season'].min()}-{part['Season'].max()}"
            print(f"  {era:<4} {span}: {summarise(part, rng)}")
        if threshold == 0.02:
            by_season = bets.groupby("Season")["Profit"].agg(["size", "sum", "mean"]).round(3)
            print(by_season.to_string())


if __name__ == "__main__":
    main()
