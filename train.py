"""Compare feature sets on the validation season, then score the chosen set on test."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from prepare_data import load_training_data

LABELS = ["H", "D", "A"]
FORM = ["PointsFormDiff", "GoalDiffFormDiff", "ShotsOnTargetDiffFormDiff"]
FEATURE_SETS = {
    "Elo only": ["EloDifference"],
    **{f"Elo + {name}": ["EloDifference", name] for name in FORM},
    "Elo + all form": ["EloDifference"] + FORM,
    "Elo + SoT form + squad value": ["EloDifference", "ShotsOnTargetDiffFormDiff", "SquadValueDiff"],
    "Elo + SoT form + lineup value": ["EloDifference", "ShotsOnTargetDiffFormDiff", "LineupValueDiff"],
    "Elo + SoT form + squad + lineup": [
        "EloDifference", "ShotsOnTargetDiffFormDiff", "SquadValueDiff", "LineupValueDiff",
    ],
    "Elo + shot xG form": ["EloDifference", "ShotXGDiffFormDiff"],
    "Elo + shot xG form + lineup value": ["EloDifference", "ShotXGDiffFormDiff", "LineupValueDiff"],
    "Elo + SoT form + manager + rest": [
        "EloDifference", "ShotsOnTargetDiffFormDiff", "ManagerTenureDiff", "RestDaysDiff",
    ],
}
# Chosen from validation and backtest.py (seasons before the test season); test is scored for these sets alone.
# One model per forecast time: lineups are only announced about an hour before kickoff.
SELECTED = {
    "Before lineups": "Elo + shot xG form",
    "After lineups": "Elo + shot xG form + lineup value",
}


def fit(X_train, y_train):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    # The pipeline learns both scaling and model parameters from training only.
    return model.fit(X_train, y_train)


def evaluate(name, y_true, probabilities, classes):
    """Print log loss and accuracy for probability columns ordered like `classes`."""
    predictions = np.asarray(classes)[probabilities.argmax(axis=1)]
    print(
        f"  {name:<34} log loss {log_loss(y_true, probabilities, labels=classes):.4f}"
        f"   accuracy {accuracy_score(y_true, predictions):.2%}"
    )


def main():
    print("Validation (2023/24) comparison:")
    for name, features in FEATURE_SETS.items():
        X_train, X_val, _, y_train, y_val, _ = load_training_data(features)
        model = fit(X_train, y_train)
        evaluate(name, y_val, model.predict_proba(X_val), model.classes_)

    for forecast_time, name in SELECTED.items():
        report(forecast_time, name)


def report(forecast_time, name):
    X_train, X_val, X_test, y_train, y_val, y_test = load_training_data(FEATURE_SETS[name])
    model = fit(X_train, y_train)
    classes = model.classes_
    print(f"\n{forecast_time}: {name} {FEATURE_SETS[name]}")

    # Baseline: predict the training outcome frequencies for every match.
    frequencies = y_train.value_counts(normalize=True).reindex(classes).to_numpy()

    for split, X, y in [("Validation", X_val, y_val), ("Test", X_test, y_test)]:
        print(f"{split} ({len(y)} matches):")
        evaluate("Class-frequency baseline", y, np.tile(frequencies, (len(y), 1)), classes)
        probabilities = model.predict_proba(X)
        evaluate(name, y, probabilities, classes)
        print("  Mean predicted vs actual rate:")
        print(pd.DataFrame({
            "Predicted": probabilities.mean(axis=0),
            "Actual": y.value_counts(normalize=True).reindex(classes).to_numpy(),
        }, index=classes).reindex(LABELS).round(3).to_string())

    matrix = pd.DataFrame(
        confusion_matrix(y_val, model.predict(X_val), labels=LABELS),
        index=pd.Index(LABELS, name="Actual"),
        columns=pd.Index(LABELS, name="Predicted"),
    )
    print("Validation confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
