"""Leakage checks for the form features. Run: .venv\\Scripts\\python.exe check_features.py"""

import numpy as np

from prepare_data import (
    FORM_STATS, FORM_WINDOW, HISTORY_SEASONS, SEASONS, add_form_features, load_all_matches,
)


def load():
    matches = load_all_matches(HISTORY_SEASONS + SEASONS)
    # Elo isn't needed here; only the split labels are.
    matches["Split"] = np.where(matches["Season"].isin(HISTORY_SEASONS), "history", "train")
    return matches


def team_rows(matches, team):
    played = matches["HomeTeam"].eq(team) | matches["AwayTeam"].eq(team)
    return matches.loc[played].sort_values("Date", kind="stable")


def check_form_by_hand(matches, featured, team="Arsenal", season="2324", match_number=10):
    """Recompute one team's points form from raw results, using only earlier matches."""
    history = team_rows(matches, team)
    target = history.loc[history["Season"].eq(season)].index[match_number]
    earlier = history.loc[: target].iloc[:-1].tail(FORM_WINDOW)
    points = [
        3 if (row.HomeTeam == team) == (row.FTR == "H") and row.FTR != "D"
        else 1 if row.FTR == "D" else 0
        for row in earlier.itertuples()
    ]
    side = "Home" if matches.loc[target, "HomeTeam"] == team else "Away"
    assert len(earlier) == FORM_WINDOW
    assert earlier["Date"].lt(matches.loc[target, "Date"]).all()
    assert np.isclose(featured.loc[target, f"{side}PointsForm"], np.mean(points))
    print(f"ok: {team} points form recomputed by hand ({np.mean(points):.2f})")


def check_own_result_ignored(matches, featured, match_id=1000):
    """Changing a match's result must not change that match's features, only later ones."""
    changed = matches.copy()
    changed.loc[match_id, ["FTHG", "FTAG", "HST", "AST"]] = [9, 0, 20, 0]
    refeatured = add_form_features(changed)
    columns = [f"{side}{stat}Form" for side in ["Home", "Away"] for stat in FORM_STATS]
    assert np.allclose(
        featured.loc[match_id, columns].astype(float), refeatured.loc[match_id, columns].astype(float)
    )
    team = matches.loc[match_id, "HomeTeam"]
    later = team_rows(matches, team).loc[match_id:].index[1]
    side = "Home" if matches.loc[later, "HomeTeam"] == team else "Away"
    assert not np.isclose(
        featured.loc[later, f"{side}GoalDiffForm"], refeatured.loc[later, f"{side}GoalDiffForm"]
    )
    print("ok: a match's own result does not affect its features, but does affect the next match")


def check_promoted_restart(matches, featured):
    """A promoted team's first match uses the fill value; its second uses only the first."""
    first_season = matches.loc[matches["Season"].eq(SEASONS[0])]
    previous = matches.loc[matches["Season"].eq(HISTORY_SEASONS[-1])]
    promoted = sorted(set(first_season["HomeTeam"]) - set(previous["HomeTeam"]))
    assert len(promoted) == 3, promoted
    for team in promoted:
        rows = team_rows(first_season, team)
        first, second = rows.index[:2]
        first_side = "Home" if matches.loc[first, "HomeTeam"] == team else "Away"
        second_side = "Home" if matches.loc[second, "HomeTeam"] == team else "Away"
        goal_diff = matches.loc[first, "FTHG"] - matches.loc[first, "FTAG"]
        if first_side == "Away":
            goal_diff = -goal_diff
        assert np.isclose(featured.loc[second, f"{second_side}GoalDiffForm"], goal_diff)
        assert not np.isnan(featured.loc[first, f"{first_side}GoalDiffForm"])
    print(f"ok: promoted teams restart their form ({', '.join(promoted)})")


if __name__ == "__main__":
    matches = load()
    featured = add_form_features(matches)
    check_form_by_hand(matches, featured)
    check_own_result_ignored(matches, featured)
    check_promoted_restart(matches, featured)
