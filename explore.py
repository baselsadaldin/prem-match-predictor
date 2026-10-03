import pandas as pd
from prepare_data import load_matches

matches = load_matches("2425")
columns = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]


matches = matches.sort_values("Date").reset_index(drop=True)

split_date = pd.Timestamp("2025-01-01")

train = matches[matches["Date"] < split_date].copy()
test = matches[matches["Date"] >= split_date].copy()

baseline_prediction = train["FTR"].value_counts().idxmax()
baseline_accuracy = (test["FTR"] == baseline_prediction).mean()
home_goals = train[["HomeTeam", "FTHG"]].rename(
    columns={"HomeTeam": "Team", "FTHG": "Goals"}
)

away_goals = train[["AwayTeam", "FTAG"]].rename(
    columns={"AwayTeam": "Team", "FTAG": "Goals"}
)

team_goals = pd.concat([home_goals, away_goals], ignore_index=True)

average_goals = team_goals.groupby("Team")["Goals"].mean()

home_conceded = train[["HomeTeam", "FTAG"]].rename(
    columns={"HomeTeam": "Team", "FTAG": "Conceded"}
)

away_conceded = train[["AwayTeam", "FTHG"]].rename(
    columns={"AwayTeam": "Team", "FTHG": "Conceded"}
)

team_conceded = pd.concat(
    [home_conceded, away_conceded], ignore_index=True
)

average_conceded = team_conceded.groupby("Team")["Conceded"].mean()

team_summary = pd.DataFrame({
    "Scored": average_goals,
    "Conceded": average_conceded
})

team_summary["Difference"] = (
    team_summary["Scored"] - team_summary["Conceded"]
)

test["HomeStrength"] = test["HomeTeam"].map(team_summary["Difference"])
test["AwayStrength"] = test["AwayTeam"].map(team_summary["Difference"])

test["Prediction"] = "D"
test.loc[test["HomeStrength"] > test["AwayStrength"], "Prediction"] = "H"
test.loc[test["HomeStrength"] < test["AwayStrength"], "Prediction"] = "A"

accuracy = (test["Prediction"] == test["FTR"]).mean()

print(pd.crosstab(
    test["FTR"],
    test["Prediction"],
    rownames=["Actual"],
    colnames=["Predicted"]
))
