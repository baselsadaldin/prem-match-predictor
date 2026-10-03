"""Pre-match squad and starting-XI market values from Transfermarkt (CC0).

Source: https://github.com/dcaribou/transfermarkt-datasets
Valuations are point-in-time: a match only sees valuations dated strictly before it.
"""
from pathlib import Path

import numpy as np
import pandas as pd

TM_DIR = Path(__file__).resolve().parent / "data" / "transfermarkt"
TM_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/{table}.csv.gz"
# Squad value counts a club's most valuable players, roughly a 25-man registered squad.
SQUAD_SIZE = 25
# Transfermarkt club id -> Football-Data team name.
TM_CLUBS = {
    11: "Arsenal", 29: "Everton", 31: "Liverpool", 148: "Tottenham", 180: "Southampton",
    281: "Man City", 350: "Sheffield United", 379: "West Ham", 399: "Leeds",
    405: "Aston Villa", 512: "Stoke", 543: "Wolves", 603: "Cardiff", 631: "Chelsea",
    677: "Ipswich", 703: "Nott'm Forest", 762: "Newcastle", 873: "Crystal Palace",
    931: "Fulham", 984: "West Brom", 985: "Man United", 989: "Bournemouth",
    1003: "Leicester", 1010: "Watford", 1031: "Luton", 1110: "Huddersfield",
    1123: "Norwich", 1132: "Burnley", 1148: "Brentford", 1237: "Brighton", 2288: "Swansea",
    289: "Sunderland", 641: "Middlesbrough", 1032: "Reading", 1039: "QPR", 1071: "Wigan", 3008: "Hull",
}


def load_table(table, **kwargs):
    path = TM_DIR / f"{table}.csv.gz"
    if not path.exists():
        TM_DIR.mkdir(parents=True, exist_ok=True)
        pd.read_csv(TM_URL.format(table=table), **kwargs).to_csv(path, index=False)
    return pd.read_csv(path, **kwargs)


def match_games(matches):
    """Attach each match's Transfermarkt game_id, checking teams and dates agree."""
    games = load_table("games")
    games = games.loc[games["competition_id"].eq("GB1")].copy()
    # Transfermarkt's season 2017 is Football-Data's "1718".
    games["Season"] = games["season"].map(lambda year: f"{year % 100:02d}{(year + 1) % 100:02d}")
    games["HomeTeam"] = games["home_club_id"].map(TM_CLUBS)
    games["AwayTeam"] = games["away_club_id"].map(TM_CLUBS)
    games["TMDate"] = pd.to_datetime(games["date"])
    games = games.loc[games["Season"].isin(matches["Season"].unique())]
    if games[["HomeTeam", "AwayTeam"]].isna().any().any():
        raise ValueError("Transfermarkt clubs missing from TM_CLUBS")
    merged = matches.merge(
        games[["Season", "HomeTeam", "AwayTeam", "game_id", "home_club_id", "away_club_id", "TMDate"]],
        on=["Season", "HomeTeam", "AwayTeam"], how="left", validate="one_to_one",
    )
    if merged["game_id"].isna().any():
        raise ValueError("Matches missing from Transfermarkt games")
    if not merged["TMDate"].eq(merged["Date"]).all():
        raise ValueError("Transfermarkt dates disagree with Football-Data dates")
    return merged.drop(columns="TMDate")


def latest_values(pairs, valuations):
    """For (player_id, Date) rows, the player's latest valuation strictly before Date."""
    pairs = pairs.sort_values("Date")
    valuations = valuations.sort_values("ValuationDate")
    return pd.merge_asof(
        pairs, valuations, left_on="Date", right_on="ValuationDate", by="player_id",
        direction="backward", allow_exact_matches=False,
    )


def squad_values(dates, valuations):
    """Top-SQUAD_SIZE value of each club on each date, from players' latest prior valuations.

    A player belongs to the club named on his latest valuation before the date.
    """
    rows = []
    for date in sorted(dates):
        known = valuations.loc[valuations["ValuationDate"] < date]
        latest = known.drop_duplicates("player_id", keep="last")
        top = latest.sort_values("Value", ascending=False).groupby("ValuationClub").head(SQUAD_SIZE)
        totals = top.groupby("ValuationClub")["Value"].sum()
        rows.append(pd.DataFrame({"Date": date, "club_id": totals.index, "SquadValue": totals.to_numpy()}))
    return pd.concat(rows, ignore_index=True)


def add_squad_features(matches):
    """Add log squad value and log starting-XI value differences (home minus away).

    SquadValueDiff is known well before kickoff. LineupValueDiff uses the announced
    starting XI, so it is only available about an hour before kickoff.
    """
    matches = match_games(matches)
    valuations = load_table("player_valuations").rename(columns={
        "date": "ValuationDate", "market_value_in_eur": "Value", "current_club_id": "ValuationClub",
    })
    valuations["ValuationDate"] = pd.to_datetime(valuations["ValuationDate"])
    valuations = valuations[["player_id", "ValuationDate", "Value", "ValuationClub"]].sort_values(
        "ValuationDate", kind="stable"
    )

    squads = squad_values(matches["Date"].unique(), valuations)
    lineups = load_table("game_lineups", usecols=["game_id", "club_id", "player_id", "type"])
    lineups = lineups.loc[
        lineups["game_id"].isin(matches["game_id"]) & lineups["type"].eq("starting_lineup")
    ].merge(matches[["game_id", "Date"]], on="game_id")
    starters = latest_values(lineups, valuations[["player_id", "ValuationDate", "Value"]])
    if not starters["ValuationDate"].dropna().lt(starters["Date"].loc[starters["ValuationDate"].notna()]).all():
        raise ValueError("A starter's valuation is not strictly before the match")
    # A starter with no earlier valuation is an unknown youngster; count him as worth nothing.
    starters["Value"] = starters["Value"].fillna(0)
    xi = starters.groupby(["game_id", "club_id"]).agg(
        LineupValue=("Value", "sum"), Starters=("player_id", "size")
    ).reset_index()
    if not xi["Starters"].eq(11).all() or len(xi) != 2 * len(matches):
        raise ValueError("Every match needs 11 starters for both clubs")

    for side in ["Home", "Away"]:
        club = f"{side.lower()}_club_id"
        matches = matches.merge(
            squads.rename(columns={"club_id": club, "SquadValue": f"{side}SquadValue"}),
            on=["Date", club], how="left",
        ).merge(
            xi.drop(columns="Starters").rename(columns={"club_id": club, "LineupValue": f"{side}LineupValue"}),
            on=["game_id", club], how="left",
        )
    value_columns = [f"{side}{kind}Value" for side in ["Home", "Away"] for kind in ["Squad", "Lineup"]]
    if matches[value_columns].isna().any().any() or matches[value_columns].le(0).any().any():
        raise ValueError("Missing or zero squad or lineup values")
    for kind in ["Squad", "Lineup"]:
        matches[f"{kind}ValueDiff"] = np.log(matches[f"Home{kind}Value"]) - np.log(matches[f"Away{kind}Value"])
    return matches.drop(columns=["game_id", "home_club_id", "away_club_id"])
