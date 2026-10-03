# Premier League Match Predictor

A Premier League match predictor (home win, draw or away win) built with Claude Code, which writes and runs the code.

## Milestones

1. Set up Python and load historical match data.
2. Explore the data and predict home win, draw, or away win.
3. Compare against a simple baseline using chronological training and test data. Use only information available before each match.
4. Examine home-win, draw, and away-win probabilities and evaluate calibration.
5. Compare probabilities with market prices in simulations.

## Results

All five milestones are done; [FEATURE_PLAN.md](FEATURE_PLAN.md) has the full log. Every figure below is walk-forward: each season is predicted by a model trained only on earlier seasons.

- **Model:** logistic regression on Elo difference and 5-match shot-xG form, plus starting-XI market value once lineups are known.
- **Accuracy:** about 55.7% walk-forward over 2016/17–2023/24 (always picking home wins: about 45%).
- **Calibration:** good. Calibration error is about 0.015, against 0.017 for Pinnacle's odds.
- **Against the market:** Pinnacle's odds predict better in every season, with log loss lower by 0.012 on average.
- **Betting:** value betting against Pinnacle shows no edge in the Premier League or the Championship. Closing-line value is about −2%, and returns are within noise.
- **Tried and not adopted:** points and goal-difference form, squad value, manager tenure, rest days, a Dixon-Coles goals model and temperature scaling.

## Setup

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`data/` isn't committed. The scripts download Football-Data CSVs, Elo snapshots and the Transfermarkt dataset into it on first run.

## Running it

1. Run `.venv\Scripts\python.exe prepare_data.py`. It builds pre-match Elo, recent-form (including shot-based xG), squad-value, starting-XI-value, manager-tenure and rest-day features for 2013/14–2024/25 in `data/matches_elo.csv`. The first run downloads about 140 MB of Transfermarkt data into `data/transfermarkt/`.
2. Run `.venv\Scripts\python.exe train.py`. It fits on 2013/14–2022/23 and scores validation (2023/24) and test (2024/25).
3. Run `.venv\Scripts\python.exe backtest.py`. It scores each season from 2016/17, training on all earlier seasons.
4. Run `.venv\Scripts\python.exe betting.py`. It simulates flat-stake value betting against Pinnacle odds using the walk-forward probabilities.

- `elo.py` loads cached historical Elo snapshots.
- `calibration.py` checks walk-forward probability calibration against Pinnacle and tests temperature scaling.
- `championship.py` runs the Elo + form pipeline, Dixon-Coles, betting simulation and line shopping on the Championship.
- `line_shopping.py` tests betting the best market price against Pinnacle's margin-free odds (no model).
- `dixon_coles.py` fits a walk-forward Dixon-Coles goals model and tunes its time decay.
- `team_news.py` builds manager-tenure and rest-day features from Transfermarkt games.
- `squad.py` builds squad and starting-XI market values from the Transfermarkt dataset (CC0).
- `check_features.py` checks that the form features use only earlier matches.
- `explore.py` is the original 2024/25 exploration.
- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) explains the leakage rules behind the Elo features.
- [FEATURE_PLAN.md](FEATURE_PLAN.md) holds the feature roadmap and the results log.
