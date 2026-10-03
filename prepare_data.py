"""Prepare pre-match features; no model fitting happens here."""
from pathlib import Path

import numpy as np
import pandas as pd

from elo import load_elo
from squad import add_squad_features
from team_news import add_team_news_features

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_FILE = DATA_DIR / "matches_elo.csv"
MATCH_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
# Football-Data league codes: E0 = Premier League, E1 = Championship.
MATCHES_PER_SEASON = {"E0": 380, "E1": 552}
SEASONS = [
    "1314", "1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425",
]
# Loaded only so early-season form has history; dropped from the output.
HISTORY_SEASONS = ["1213"]
VALIDATION_SEASON = "2324"
TEST_SEASON = "2425"
FORM_WINDOW = 5
# Per-team statistics averaged over each team's previous FORM_WINDOW matches.
# ShotXGDiff is a shot-count stand-in for xG; no licensed shot-location data covers these seasons.
FORM_STATS = ["Points", "GoalDiff", "ShotsOnTargetDiff", "ShotsDiff", "ShotXGDiff"]
# Every column a model may use. Anything else in the CSV is for inspection only.
# SquadValueDiff, ManagerTenureDiff and RestDaysDiff are known days ahead;
# LineupValueDiff only once lineups are announced (~1h before kickoff).
FEATURE_COLUMNS = (
    ["EloDifference"] + [f"{stat}FormDiff" for stat in FORM_STATS]
    + ["SquadValueDiff", "LineupValueDiff", "ManagerTenureDiff", "RestDaysDiff"]
)
# Match-day statistics, kept for building pre-match form later. Never features as-is.
MATCH_STAT_COLUMNS = ["HS", "AS", "HST", "AST"]
# Football-Data and this Elo archive use different Forest names.
TEAM_ALIASES = {"Nott'm Forest": "Nottm Forest"}


def load_matches(season, league="E0"):
    """Load one season (e.g. "2425"), downloading it once and caching it locally."""
    match_file = DATA_DIR / f"{league}_{season}.csv"
    if not match_file.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        pd.read_csv(MATCH_URL.format(season=season, league=league), encoding="latin-1").to_csv(
            match_file, index=False
        )
    matches = pd.read_csv(match_file)
    columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"] + MATCH_STAT_COLUMNS
    matches = matches[columns].dropna(how="all").copy()
    matches["Date"] = pd.to_datetime(matches["Date"], format="mixed", dayfirst=True, errors="raise")
    result_columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    if matches[result_columns].isna().any().any() or not matches["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError(f"Season {season} contains missing values or invalid results")
    # Form skips missing statistics. The only gap is Bolton v Brentford (Championship,
    # 27 Apr 2019), never played because of a players' strike, with its result awarded.
    if matches[MATCH_STAT_COLUMNS].isna().any(axis=1).sum() > 1:
        raise ValueError(f"Season {season} is missing match statistics")
    if len(matches) != MATCHES_PER_SEASON[league]:
        raise ValueError(f"{league} {season} has {len(matches)} matches, expected {MATCHES_PER_SEASON[league]}")
    matches.insert(0, "Season", season)
    return matches


def load_all_matches(seasons=SEASONS, league="E0"):
    matches = pd.concat([load_matches(season, league) for season in seasons], ignore_index=True)
    if matches.duplicated(["Date", "HomeTeam", "AwayTeam"]).any():
        raise ValueError("Duplicate matches")
    return matches.sort_values("Date", kind="stable").reset_index(drop=True)


def add_elo_features(matches, ratings, max_unrated=0):
    """Backward join, excluding same-day snapshots to avoid ambiguous timing.

    Up to `max_unrated` matches may lack a snapshot from the previous 31 days; their Elo is left
    missing for the caller to drop. The Premier League allows none.
    """
    matches = matches.sort_values("Date", kind="stable").reset_index(drop=True).copy()
    ratings = ratings.loc[
        ratings["country"].eq("ENG") & (ratings["date"] < matches["Date"].max()),
        ["date", "club", "elo"],
    ].copy()
    ratings["club"] = ratings["club"].replace(TEAM_ALIASES)
    if ratings.isna().any().any() or ratings.duplicated(["date", "club"]).any():
        raise ValueError("Missing or duplicate English Elo snapshots")
    unrated = pd.Series(False, index=matches.index)
    for side in ["Home", "Away"]:
        key = f"{side}EloClub"
        date_column = f"{side}EloDate"
        matches[key] = matches[f"{side}Team"].replace(TEAM_ALIASES)
        unknown = set(matches[key]) - set(ratings["club"])
        if unknown:
            raise ValueError(f"Unmapped {side.lower()} teams: {sorted(unknown)}")
        snapshots = ratings.rename(columns={
            "club": key, "date": date_column, "elo": f"{side}Elo"
        }).sort_values(date_column)
        matches = pd.merge_asof(
            matches, snapshots, left_on="Date", right_on=date_column,
            by=key, direction="backward", allow_exact_matches=False,
        )
        if not (matches[date_column].dropna() < matches.loc[matches[date_column].notna(), "Date"]).all():
            raise ValueError("Elo snapshot is not strictly before the match")
        matches[f"{side}EloAgeDays"] = (matches["Date"] - matches[date_column]).dt.days
        # Missing or stale (more than 31 days old) snapshots count as unrated.
        side_unrated = matches[f"{side}Elo"].isna() | matches[f"{side}EloAgeDays"].gt(31)
        matches.loc[side_unrated, f"{side}Elo"] = np.nan
        unrated |= side_unrated
        matches = matches.drop(columns=key)
    if unrated.sum() > max_unrated:
        raise ValueError(f"{unrated.sum()} matches without a pre-match Elo rating from the previous 31 days")
    matches["EloDifference"] = matches["HomeElo"] - matches["AwayElo"]
    matches["Split"] = "train"
    matches.loc[matches["Season"].isin(HISTORY_SEASONS), "Split"] = "history"
    matches.loc[matches["Season"].eq(VALIDATION_SEASON), "Split"] = "validation"
    matches.loc[matches["Season"].eq(TEST_SEASON), "Split"] = "test"
    return matches


def shot_xg_weights(matches):
    """Goals per shot on target and per shot off target, fitted on the history seasons only.

    Fitting on seasons that are never scored keeps the weights free of hindsight.
    Off-target shots include blocked shots.
    """
    history = matches.loc[matches["Season"].isin(HISTORY_SEASONS)]
    if history.empty:
        raise ValueError("Shot xG weights need the history seasons")
    shots = np.column_stack([
        np.concatenate([history["HST"], history["AST"]]),
        np.concatenate([history["HS"] - history["HST"], history["AS"] - history["AST"]]),
    ])
    goals = np.concatenate([history["FTHG"], history["FTAG"]])
    on_target, off_target = np.linalg.lstsq(shots, goals, rcond=None)[0]
    return on_target, off_target


def team_match_table(matches):
    """Two rows per match, one from each team's point of view."""
    on_target, off_target = shot_xg_weights(matches)
    # Football-Data shot columns: HS/HST for the home side, AS/AST for the away side.
    shot_xg = {
        side: on_target * matches[f"{side[0]}ST"] + off_target * (matches[f"{side[0]}S"] - matches[f"{side[0]}ST"])
        for side in ["Home", "Away"]
    }
    sides = []
    for side, other, own_goals, other_goals, own_sot, other_sot in [
        ("Home", "Away", "FTHG", "FTAG", "HST", "AST"),
        ("Away", "Home", "FTAG", "FTHG", "AST", "HST"),
    ]:
        goal_diff = matches[own_goals] - matches[other_goals]
        sides.append(pd.DataFrame({
            "MatchId": matches.index,
            "Side": side,
            "Season": matches["Season"],
            "Split": matches["Split"],
            "Date": matches["Date"],
            "Team": matches[f"{side}Team"],
            "Points": goal_diff.gt(0) * 3 + goal_diff.eq(0) * 1,
            "GoalDiff": goal_diff,
            "ShotsOnTargetDiff": matches[own_sot] - matches[other_sot],
            "ShotsDiff": matches[f"{side[0]}S"] - matches[f"{other[0]}S"],
            "ShotXGDiff": shot_xg[side] - shot_xg[other],
        }))
    return pd.concat(sides, ignore_index=True).sort_values(["Team", "Date"], kind="stable")


def add_form_features(matches, window=FORM_WINDOW):
    """Average each team's statistics over its previous `window` matches, never the current one.

    Form carries over from the previous season but restarts after promotion. A promoted
    team's first match has no history, so it gets the average form of promoted teams
    in the training seasons. Missing statistics are skipped,
    so form averages each team's last `window` matches where the statistic was recorded.
    """
    matches = matches.reset_index(drop=True).copy()
    teams = team_match_table(matches)
    seasons = sorted(matches["Season"].unique())
    previous_season = dict(zip(seasons[1:], seasons[:-1]))
    played = set(zip(teams["Team"], teams["Season"]))
    teams["Promoted"] = [
        season in previous_season and (team, previous_season[season]) not in played
        for team, season in zip(teams["Team"], teams["Season"])
    ]
    # A new spell starts at the first match of each promoted season.
    spell_start = teams["Promoted"] & ~teams.duplicated(["Team", "Season"])
    teams["Spell"] = spell_start.groupby(teams["Team"]).cumsum()

    keys = [teams["Team"], teams["Spell"]]
    for stat in FORM_STATS:
        recorded = teams.loc[teams[stat].notna(), stat]
        # Average of the last `window` recorded values, up to and including each match...
        upto = recorded.groupby([keys[0][recorded.index], keys[1][recorded.index]]).rolling(
            window, min_periods=1
        ).mean().reset_index(level=[0, 1], drop=True)
        # ...then shift(1) so each match sees only the value from before it.
        teams[stat] = upto.reindex(teams.index).groupby(keys).shift(1).groupby(keys).ffill()

    # Fixed prior from the unscored history season, never later training results.
    history = teams.loc[teams["Season"].isin(HISTORY_SEASONS)]
    fill_values = history[FORM_STATS].mean()
    no_history = teams.groupby(["Team", "Spell"]).cumcount().eq(0) & teams["Promoted"]
    if no_history.any() and fill_values.isna().any():
        raise ValueError("Promoted-team form needs complete history-season priors")
    teams.loc[no_history, FORM_STATS] = fill_values.to_numpy()

    for side in ["Home", "Away"]:
        side_form = teams.loc[teams["Side"].eq(side)].set_index("MatchId")[FORM_STATS]
        matches = matches.join(side_form.add_prefix(side).add_suffix("Form"))
    for stat in FORM_STATS:
        matches[f"{stat}FormDiff"] = matches[f"Home{stat}Form"] - matches[f"Away{stat}Form"]

    used = matches["Split"].ne("history")
    if matches.loc[used, [f"{stat}FormDiff" for stat in FORM_STATS]].isna().any().any():
        raise ValueError("Missing form features outside the history seasons")
    return matches


def build_dataset():
    matches = add_form_features(add_elo_features(load_all_matches(HISTORY_SEASONS + SEASONS), load_elo()))
    # History seasons only feed form; Transfermarkt has no lineups for them.
    matches = matches.loc[matches["Split"].ne("history")]
    matches = add_team_news_features(add_squad_features(matches))
    return matches.sort_values("Date", kind="stable").reset_index(drop=True)


def load_training_data(features=FEATURE_COLUMNS):
    """Return X_train, X_val, X_test, y_train, y_val, y_test using only whitelisted features."""
    features = list(features)
    not_allowed = set(features) - set(FEATURE_COLUMNS)
    if not_allowed:
        raise ValueError(f"Not whitelisted as features: {sorted(not_allowed)}")
    frame = pd.read_csv(OUTPUT_FILE, dtype={"Season": str})
    parts = [frame.loc[frame["Split"].eq(split)] for split in ["train", "validation", "test"]]
    if any(part[features].isna().any().any() for part in parts):
        raise ValueError("Missing feature values")
    return (
        *(part[features].copy() for part in parts),
        *(part["FTR"].copy() for part in parts),
    )


if __name__ == "__main__":
    frame = build_dataset()
    frame.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(frame)} matches to {OUTPUT_FILE}")
    print(frame.groupby(["Split", "Season"])["Date"].agg(["count", "min", "max"]))
    print("Maximum snapshot age (days):", frame[["HomeEloAgeDays", "AwayEloAgeDays"]].max().to_dict())
    print("Shot xG weights (goals per shot on / off target, fitted on 2012/13): %.3f / %.3f" % shot_xg_weights(
        load_all_matches(HISTORY_SEASONS)
    ))
