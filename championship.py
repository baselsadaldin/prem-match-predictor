"""The same pipeline on the Championship (Football-Data E1), a market expected to be softer.

Features: Elo and recent form only. Transfermarkt's dataset covers top divisions only, so there is
no squad or lineup value. Form restarts for every newcomer, promoted or relegated.
Feature sets are compared walk-forward on the decision seasons (2016/17-2023/24); the best one
is then bet against Pinnacle, and 2024/25 is scored once.
"""
import numpy as np
import pandas as pd

import dixon_coles
import line_shopping
from backtest import FIRST_SCORED
from betting import ODDS, attach_odds, simulate, walk_forward_probabilities
from elo import load_elo
from prepare_data import (
    DATA_DIR, HISTORY_SEASONS, SEASONS, TEST_SEASON, add_elo_features, add_form_features, load_all_matches,
)

LEAGUE = "E1"
OUTPUT = DATA_DIR / "championship.csv"
OUTCOMES = ["H", "D", "A"]
# Coventry, Rotherham and Wycombe on 12 Sep 2020: the archive's first ratings after promotion
# from League One are dated 15 Sep, so these three matches have no pre-match Elo and are dropped.
MAX_UNRATED = 3
E, S, X = "EloDifference", "ShotsOnTargetDiffFormDiff", "ShotXGDiffFormDiff"
FEATURE_SETS = {"Elo": [E], "Elo+SoT": [E, S], "Elo+xG": [E, X]}
# Five matches lack some Pinnacle odds (one each in 2018/19 and 2019/20, three in 2020/21).
# They are dropped so every model and the market are scored on the same matches.
MAX_MISSING_ODDS = 5
# Dixon-Coles settings chosen on the Premier League; not re-tuned here.
DC_HALF_LIFE, DC_ALPHA = 365, 0.005


def build():
    matches = load_all_matches(HISTORY_SEASONS + SEASONS, LEAGUE)
    matches = add_form_features(add_elo_features(matches, load_elo(), max_unrated=MAX_UNRATED))
    matches = matches.loc[matches["Split"].ne("history") & matches["EloDifference"].notna()]
    matches = matches.reset_index(drop=True)
    matches.to_csv(OUTPUT, index=False)
    return matches


def log_losses(matches, probabilities):
    actual = matches.loc[probabilities.index, "FTR"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    loss = -np.log(probabilities[OUTCOMES].to_numpy()[np.arange(len(actual)), actual])
    return pd.Series(loss, index=probabilities.index)


def main():
    build()
    matches = attach_odds(pd.read_csv(OUTPUT, dtype={"Season": str}, parse_dates=["Date"]), LEAGUE, MAX_MISSING_ODDS)
    first_scored = sorted(matches["Season"].unique())[FIRST_SCORED]

    candidates = {name: walk_forward_probabilities(matches, features) for name, features in FEATURE_SETS.items()}
    all_matches = load_all_matches(HISTORY_SEASONS + SEASONS, LEAGUE)
    dc_matches, dc = dixon_coles.walk_forward(all_matches, DC_HALF_LIFE, DC_ALPHA, first_scored)
    keys = ["Season", "Date", "HomeTeam", "AwayTeam"]
    dc = dc_matches.loc[dc.index, keys].join(dc[OUTCOMES])
    index = matches.loc[matches["Season"] >= first_scored, keys].reset_index()
    candidates["Dixon-Coles"] = index.merge(dc, on=keys, validate="one_to_one").set_index("index")[OUTCOMES]
    for name, closing in [("Pinnacle early", False), ("Pinnacle closing", True)]:
        implied = 1 / matches.loc[index["index"], ODDS["After lineups" if closing else "Before lineups"]].to_numpy()
        candidates[name] = pd.DataFrame(implied / implied.sum(axis=1, keepdims=True), columns=OUTCOMES,
                                        index=index["index"].to_numpy())

    seasons = matches["Season"]
    table = pd.DataFrame({
        name: log_losses(matches, p).groupby(seasons.loc[p.index]).mean() for name, p in candidates.items()
    })
    print("Championship walk-forward log loss by season:")
    print(table.round(4).to_string())
    decision = table.drop(TEST_SEASON)
    print("Decision-season mean:\n" + decision.mean().round(4).to_string())

    models = [name for name in candidates if not name.startswith("Pinnacle")]
    chosen = min(models, key=lambda name: decision[name].mean())
    print(f"\nChosen on decision seasons: {chosen}")
    rng = np.random.default_rng(0)
    for odds_name, odds_columns in [("early", ODDS["Before lineups"]), ("closing", ODDS["After lineups"])]:
        print(f"\n{chosen} vs Pinnacle {odds_name} odds {'/'.join(odds_columns)}")
        probabilities = candidates[chosen]
        simulate(matches.loc[probabilities.index], probabilities, odds_columns, rng)

    print("\nLine shopping (best price vs Pinnacle early fair odds), Championship:")
    for threshold in [0.0, 0.02]:
        bets = pd.concat([line_shopping.season_bets(s, threshold, LEAGUE, MAX_MISSING_ODDS) for s in SEASONS], ignore_index=True)
        for era, part in bets.groupby("Era", sort=False):
            span = f"{part['Season'].min()}-{part['Season'].max()}"
            print(f"  > {threshold:.0%} {era:<4} {span}: {line_shopping.summarise(part, rng)}")


if __name__ == "__main__":
    main()
