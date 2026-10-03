"""Walk-forward backtest: for each season, train on all earlier seasons and score that season.

One validation season is too noisy to choose features with. This scores every feature
set on many out-of-sample seasons. Feature decisions use seasons up to VALIDATION_SEASON.
"""

import numpy as np
import pandas as pd

from prepare_data import OUTPUT_FILE, TEST_SEASON
from train import fit

FIRST_SCORED = 3  # Scoring starts once 3 earlier seasons are available for training.
E, S = "EloDifference", "ShotsOnTargetDiffFormDiff"
M, R = "ManagerTenureDiff", "RestDaysDiff"
FEATURE_SETS = {
    "Elo": [E],
    "Elo+SoT": [E, S],
    "Elo+xG": [E, "ShotXGDiffFormDiff"],
    "+shots": [E, S, "ShotsDiffFormDiff"],
    "+squad": [E, S, "SquadValueDiff"],
    "+lineup": [E, S, "LineupValueDiff"],
    "+manager": [E, S, M],
    "+rest": [E, S, R],
    "+mgr+rest": [E, S, M, R],
    "xG+lineup": [E, "ShotXGDiffFormDiff", "LineupValueDiff"],
}
COMPARISONS = [
    ("Elo+SoT", "Elo"), ("Elo+xG", "Elo+SoT"), ("+shots", "Elo+SoT"), ("+squad", "Elo+SoT"),
    ("+lineup", "Elo+SoT"), ("+manager", "Elo+SoT"), ("+rest", "Elo+SoT"), ("+mgr+rest", "Elo+SoT"),
    ("xG+lineup", "Elo+xG"),
]


def per_match_losses(frame):
    seasons = sorted(frame["Season"].unique())
    losses = {name: {} for name in FEATURE_SETS}
    for season in seasons[FIRST_SCORED:]:
        train, scored = frame.loc[frame["Season"] < season], frame.loc[frame["Season"] == season]
        for name, features in FEATURE_SETS.items():
            model = fit(train[features], train["FTR"])
            probabilities = model.predict_proba(scored[features])
            actual = np.searchsorted(model.classes_, scored["FTR"])
            losses[name][season] = -np.log(probabilities[np.arange(len(scored)), actual])
    return losses


def pooled_gain(losses, a, b, seasons, rng):
    """Mean log-loss change of `a` over `b` across seasons, with a 95% bootstrap interval."""
    gain = np.concatenate([losses[a][s] - losses[b][s] for s in seasons])
    boot = gain[rng.integers(0, len(gain), (5000, len(gain)))].mean(axis=1)
    return gain.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)


def main():
    frame = pd.read_csv(OUTPUT_FILE, dtype={"Season": str})
    losses = per_match_losses(frame)
    table = pd.DataFrame({name: {s: l.mean() for s, l in by.items()} for name, by in losses.items()})
    print("Log loss by season (trained on all earlier seasons):")
    print(table.round(4).to_string())

    rng = np.random.default_rng(0)
    decision_seasons = [s for s in table.index if s != TEST_SEASON]
    for label, seasons in [("Decision seasons (excluding test)", decision_seasons), ("All seasons", list(table.index))]:
        print(f"\n{label}: {seasons[0]}-{seasons[-1]}, {len(seasons)} seasons")
        for a, b in COMPARISONS:
            mean, low, high = pooled_gain(losses, a, b, seasons, rng)
            improved = sum(table.loc[s, a] < table.loc[s, b] for s in seasons)
            print(f"  {a:<10} vs {b:<8} {mean:+.4f}  95% CI [{low:+.4f}, {high:+.4f}]"
                  f"  better in {improved}/{len(seasons)} seasons")


if __name__ == "__main__":
    main()
