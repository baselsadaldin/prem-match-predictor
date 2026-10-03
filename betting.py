"""Milestone 5: would the model have made money against Pinnacle's odds?

Every probability comes from the walk-forward setup in backtest.py: a season is predicted by a
model trained only on earlier seasons. Each forecast time is matched with the odds available
at that time:
- before lineups: Football-Data's early Pinnacle odds (PSH/PSD/PSA, collected a day or more before kickoff)
- after lineups: Pinnacle closing odds (PSCH/PSCD/PSCA), a retrospective benchmark;
  these are not verified executable prices at the T-60 min lineup forecast

Strategy: at most one flat 1-unit bet per match, on the outcome with the highest expected value,
placed only if that expected value exceeds a threshold. The threshold is chosen on seasons
before the test season. Test is then scored once.
"""
import numpy as np
import pandas as pd

from backtest import FIRST_SCORED
from prepare_data import DATA_DIR, OUTPUT_FILE, TEST_SEASON
from train import FEATURE_SETS, SELECTED, fit

OUTCOMES = ["H", "D", "A"]
ODDS = {
    "Before lineups": ["PSH", "PSD", "PSA"],
    "After lineups": ["PSCH", "PSCD", "PSCA"],
}
CLOSING = ODDS["After lineups"]
THRESHOLDS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]


def load_odds(seasons, league="E0", max_missing=0):
    """Pinnacle early and closing odds for each match, keyed like matches_elo.csv.

    Up to `max_missing` matches without a full set of odds are dropped.
    """
    frames = []
    for season in seasons:
        odds = pd.read_csv(DATA_DIR / f"{league}_{season}.csv").dropna(subset=["FTR"])
        odds["Date"] = pd.to_datetime(odds["Date"], format="mixed", dayfirst=True)
        odds["Season"] = season
        frames.append(odds[["Season", "Date", "HomeTeam", "AwayTeam"] + ODDS["Before lineups"] + CLOSING])
    odds = pd.concat(frames, ignore_index=True)
    missing = odds[ODDS["Before lineups"] + CLOSING].isna().any(axis=1)
    if missing.sum() > max_missing:
        raise ValueError(f"{missing.sum()} matches without Pinnacle odds")
    odds = odds.loc[~missing]
    prices = odds[ODDS["Before lineups"] + CLOSING]
    if prices.le(1).any().any():
        raise ValueError("Missing or impossible Pinnacle odds")
    return odds


def walk_forward_probabilities(frame, features):
    """Out-of-sample H/D/A probabilities for every season from the FIRST_SCORED-th on."""
    seasons = sorted(frame["Season"].unique())
    parts = []
    for season in seasons[FIRST_SCORED:]:
        train, scored = frame.loc[frame["Season"] < season], frame.loc[frame["Season"] == season]
        model = fit(train[features], train["FTR"])
        probabilities = pd.DataFrame(model.predict_proba(scored[features]), columns=model.classes_, index=scored.index)
        parts.append(probabilities[OUTCOMES])
    return pd.concat(parts)


def place_bets(matches, probabilities, odds_columns, threshold):
    """One flat 1-unit bet per match on the best-value outcome, if its expected value beats the threshold."""
    if not matches.index.equals(probabilities.index):
        raise ValueError("Match and probability rows must have the same index and order")
    prices = matches[odds_columns].to_numpy()
    expected = probabilities[OUTCOMES].to_numpy() * prices - 1
    choice = expected.argmax(axis=1)
    rows = np.arange(len(matches))
    best = expected[rows, choice]
    # Closing-line value: the bet's expected value at Pinnacle's margin-free closing probabilities.
    # Bets placed at the closing odds themselves get minus the margin by construction.
    closing = 1 / matches[CLOSING].to_numpy()
    closing_fair = closing / closing.sum(axis=1, keepdims=True)
    bets = pd.DataFrame({
        "Season": matches["Season"].to_numpy(),
        "Pick": np.asarray(OUTCOMES)[choice],
        "Odds": prices[rows, choice],
        "ExpectedValue": best,
        "ClosingValue": prices[rows, choice] * closing_fair[rows, choice] - 1,
        "Won": np.asarray(OUTCOMES)[choice] == matches["FTR"].to_numpy(),
    }).loc[best > threshold]
    bets["Profit"] = np.where(bets["Won"], bets["Odds"] - 1, -1.0)
    return bets


def summarise(bets, rng):
    """Bets placed, return per unit staked with a 95% bootstrap interval, and closing-line value."""
    if bets.empty:
        return {"Bets": 0}
    profit = bets["Profit"].to_numpy()
    boot = profit[rng.integers(0, len(profit), (5000, len(profit)))].mean(axis=1)
    return {
        "Bets": len(bets),
        "Hit rate": f"{bets['Won'].mean():.1%}",
        "Avg odds": round(float(bets["Odds"].mean()), 2),
        "Profit": round(float(profit.sum()), 1),
        "ROI": f"{profit.mean():+.1%}",
        "ROI 95% CI": f"[{np.percentile(boot, 2.5):+.1%}, {np.percentile(boot, 97.5):+.1%}]",
        "CLV": f"{bets['ClosingValue'].mean():+.1%}",
    }


def simulate(scored, probabilities, odds_columns, rng):
    """Choose a threshold on the decision seasons, then score the test season once with it."""
    decision = scored["Season"].ne(TEST_SEASON)
    seasons = sorted(scored.loc[decision, "Season"].unique())
    print(f"Decision seasons {seasons[0]}-{seasons[-1]} ({decision.sum()} matches), flat 1-unit stakes:")
    results = {}
    for threshold in THRESHOLDS:
        bets = place_bets(scored.loc[decision], probabilities.loc[decision], odds_columns, threshold)
        results[threshold] = bets
        print(f"  EV > {threshold:>4.0%}: {summarise(bets, rng)}")
    # The threshold with the best decision-season return is the only one scored on test.
    chosen = max(THRESHOLDS, key=lambda t: results[t]["Profit"].mean() if len(results[t]) else -np.inf)
    by_season = results[chosen].groupby("Season")["Profit"].agg(["size", "sum", "mean"]).round(3)
    print(f"  Chosen threshold {chosen:.0%}, by season:\n{by_season.to_string()}")

    test = ~decision
    bets = place_bets(scored.loc[test], probabilities.loc[test], odds_columns, chosen)
    print(f"Test {TEST_SEASON} ({test.sum()} matches), EV > {chosen:.0%}: {summarise(bets, rng)}")


def attach_odds(frame, league="E0", max_missing=0):
    """Join Pinnacle odds; matches without them (at most `max_missing`) are dropped."""
    matches = frame.merge(
        load_odds(sorted(frame["Season"].unique()), league, max_missing),
        on=["Season", "Date", "HomeTeam", "AwayTeam"], how="inner", validate="one_to_one",
    )
    if len(frame) - len(matches) > max_missing:
        raise ValueError("Matches without odds")
    return matches.reset_index(drop=True)


def main():
    matches = attach_odds(pd.read_csv(OUTPUT_FILE, dtype={"Season": str}, parse_dates=["Date"]))
    rng = np.random.default_rng(0)
    for forecast_time, set_name in SELECTED.items():
        probabilities = walk_forward_probabilities(matches, FEATURE_SETS[set_name])
        odds_columns = ODDS[forecast_time]
        print(f"\n{forecast_time}: {set_name} vs Pinnacle {'/'.join(odds_columns)}")
        simulate(matches.loc[probabilities.index], probabilities, odds_columns, rng)


if __name__ == "__main__":
    main()
