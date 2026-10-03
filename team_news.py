"""Pre-match team news from Transfermarkt games (CC0): manager changes and fixture congestion.

Both are known before kickoff: a new manager or caretaker is announced before his first
match, and fixture dates are published weeks ahead.
"""
import unicodedata

import numpy as np
import pandas as pd

from squad import load_table, match_games

# The League Cup only appears from 2023/24, so it is left out to keep rest days
# comparable across seasons. League, FA Cup and European matches cover every season.
EXCLUDED_COMPETITIONS = ["CGB"]
# Transfermarkt games start in August 2012, a year before the first scored season,
# so tenure is only measurable up to one year. Longer tenures are capped.
TENURE_CAP_DAYS = 365
# Beyond a week's rest, more rest is assumed not to matter (and covers season openers).
REST_CAP_DAYS = 7


def normalise_name(name):
    """Transfermarkt spells some managers both with and without accents (Rúben / Ruben Amorim)."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().strip().lower()


def club_schedule(club_ids):
    """One row per club per match in any covered competition, with tenure and rest days."""
    games = load_table("games")
    games = games.loc[~games["competition_id"].isin(EXCLUDED_COMPETITIONS)]
    rows = pd.concat([
        games[["game_id", "date", f"{side}_club_id", f"{side}_club_manager_name"]].set_axis(
            ["game_id", "Date", "club_id", "Manager"], axis=1
        )
        for side in ["home", "away"]
    ], ignore_index=True)
    rows = rows.loc[rows["club_id"].isin(club_ids)].copy()
    # The only gap is Tottenham v Rennes (9 Dec 2021), cancelled and awarded, never played.
    unplayed = rows["Manager"].isna()
    if unplayed.sum() > 1:
        raise ValueError("Missing manager names")
    rows = rows.loc[~unplayed]
    rows["Date"] = pd.to_datetime(rows["Date"])
    rows["Manager"] = rows["Manager"].map(normalise_name)
    rows = rows.sort_values(["club_id", "Date"], kind="stable")
    if rows.duplicated(["club_id", "Date"]).any():
        raise ValueError("A club plays twice on one date")

    by_club = rows.groupby("club_id")
    # A spell starts whenever a club's manager differs from its previous match's manager.
    new_spell = rows["Manager"].ne(by_club["Manager"].shift())
    spell_start = rows["Date"].where(new_spell).groupby(rows["club_id"]).ffill()
    rows["TenureDays"] = (rows["Date"] - spell_start).dt.days.clip(upper=TENURE_CAP_DAYS)
    rows["RestDays"] = (rows["Date"] - by_club["Date"].shift()).dt.days
    rows["RestDays"] = rows["RestDays"].fillna(REST_CAP_DAYS).clip(upper=REST_CAP_DAYS)
    return rows


def add_team_news_features(matches):
    """Add ManagerTenureDiff (log days in charge, capped) and RestDaysDiff (capped), home minus away."""
    matches = match_games(matches)
    clubs = club_schedule(set(matches["home_club_id"]) | set(matches["away_club_id"]))
    for side in ["Home", "Away"]:
        club = f"{side.lower()}_club_id"
        matches = matches.merge(
            clubs[["game_id", "club_id", "Manager", "TenureDays", "RestDays"]].rename(columns={
                "club_id": club, "Manager": f"{side}Manager",
                "TenureDays": f"{side}TenureDays", "RestDays": f"{side}RestDays",
            }),
            on=["game_id", club], how="left", validate="one_to_one",
        )
    news_columns = [f"{side}{kind}" for side in ["Home", "Away"] for kind in ["TenureDays", "RestDays"]]
    if matches[news_columns].isna().any().any():
        raise ValueError("Missing manager tenure or rest days")
    matches["ManagerTenureDiff"] = np.log1p(matches["HomeTenureDays"]) - np.log1p(matches["AwayTenureDays"])
    matches["RestDaysDiff"] = matches["HomeRestDays"] - matches["AwayRestDays"]
    return matches.drop(columns=["game_id", "home_club_id", "away_club_id"])
