from pathlib import Path

import pandas as pd


URL = (
    "https://raw.githubusercontent.com/xgabora/"
    "Club-Football-Match-Data/main/data/EloRatings.csv"
)
DATA_FILE = Path(__file__).resolve().parent / "data" / "EloRatings.csv"


def load_elo():
    """Download the historical ratings once, then use the local copy."""
    if DATA_FILE.exists():
        ratings = pd.read_csv(DATA_FILE)
    else:
        ratings = pd.read_csv(URL)
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        ratings.to_csv(DATA_FILE, index=False)

    ratings.columns = ratings.columns.str.strip().str.lower()
    required = {"date", "club", "elo"}
    missing = required - set(ratings.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    ratings["date"] = pd.to_datetime(ratings["date"], errors="raise")
    ratings["elo"] = pd.to_numeric(ratings["elo"], errors="raise")
    return ratings.sort_values(["date", "club"]).reset_index(drop=True)


if __name__ == "__main__":
    elo = load_elo()
    print(elo.head())
    print("Columns:", elo.columns.tolist())
    print("Rows:", len(elo))
    print("Date range:", elo["date"].min(), "to", elo["date"].max())
