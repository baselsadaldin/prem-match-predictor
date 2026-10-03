# Your first trained predictor

Run `.venv\Scripts\python.exe prepare_data.py` to rebuild the data. No model is fitted by this script. Each raw season CSV is cached locally so subsequent runs work offline.

`data/matches_elo.csv` contains 4,560 matches from 2013/14 to 2024/25. 2013/14–2022/23 are training (3,800), 2023/24 is validation (380) and 2024/25 is test (380). The original version of this guide used only 2024/25, split on 1 January 2025. `FTR` is the target: H = home win, D = draw, A = away win.

Each team receives the latest available Elo snapshot **strictly before** match day. Same-day snapshots are excluded because their publication timing is unclear. Snapshot dates and ages are included for inspection; the builder rejects missing ratings, duplicate snapshots and ages above 31 days. Both spellings of Nottingham Forest are normalized in matches and ratings.

The archive stores snapshots on the 1st and 15th, so these features are less current than daily ratings. Its documentation identifies the snapshots through 1 June 2025 as ClubElo data; that covers every season used here. Source: https://github.com/xgabora/Club-Football-Match-Data. Match source: https://www.football-data.co.uk/mmz4281/{season}/E0.csv.

## First task: create train.py yourself

Start with this data-loading step:

```python
from prepare_data import load_training_data

X_train, X_test, y_train, y_test = load_training_data()
print(X_train.shape, X_test.shape)
print(y_train.value_counts())
```

The first model uses only `EloDifference = HomeElo - AwayElo`. A positive value means the home team has the higher rating. The feature whitelist prevents accidentally feeding goals, results, dates or split labels into training. HomeElo and AwayElo remain in the CSV for inspection and later experiments.

## Then implement training in small steps

1. Build a scikit-learn pipeline with `StandardScaler` followed by `LogisticRegression`. Start with `max_iter=1000` and otherwise default settings. Scaling and fitting must use training rows only. If scikit-learn is missing, install it in your environment using `.venv\Scripts\python.exe -m pip install scikit-learn`.
2. Fit on `X_train` and `y_train`, then predict H/D/A on `X_test`. Despite its name, logistic regression predicts class probabilities. With three classes it can learn separate responses for home wins, draws and away wins; its intercepts can capture a home advantage without manually adding Elo points.
3. Calculate test accuracy and a confusion matrix with an explicit class order `["H", "D", "A"]`. Compare with both the majority class learned from `y_train` and your existing 50% strength-rule result on the same test rows. Improvement is an experiment, not a guarantee.
4. Use `predict_proba` and inspect the classifier's `classes_` before labelling probability columns. Do not assume their order is H/D/A. Calculate log loss as well as accuracy: log loss evaluates how much probability the model assigns to the actual outcome, including whether wrong predictions are overconfident.

Keep the January test period for final evaluation. If you want to choose settings, make an earlier chronological validation split within the 188 training matches; fit preprocessing on that earlier subset. Repeatedly adjusting settings to improve January-May results makes the test score optimistic.

Elo snapshots can update during the test period: each prediction represents a forecast made on that match's day using information already available then. This evaluates rolling pre-match forecasts, not forecasts for the whole season made on 1 January.

Once you write the loading and fitting steps, ask for a review. We'll check the feature selection, fitting boundary, class ordering and evaluation before moving on to probabilities and calibration.
