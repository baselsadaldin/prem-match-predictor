"""Milestone 4: are the model's H/D/A probabilities calibrated?

Uses the walk-forward probabilities from backtest.py (each season predicted by a model trained
on earlier seasons) for both selected models, with Pinnacle's margin-free odds as a reference.
Then tests temperature scaling: for each season, a temperature fitted on the walk-forward
predictions of earlier seasons only.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from betting import CLOSING, ODDS, attach_odds, walk_forward_probabilities
from prepare_data import OUTPUT_FILE, TEST_SEASON
from train import FEATURE_SETS, SELECTED

OUTCOMES = ["H", "D", "A"]
BINS = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])


def reliability(probabilities, actual):
    """Per outcome and probability bin: matches, mean predicted and actual frequency."""
    rows = []
    for outcome in OUTCOMES:
        predicted = probabilities[outcome].to_numpy()
        happened = (actual == outcome).to_numpy()
        bins = np.digitize(predicted, BINS[1:-1])
        for b in np.unique(bins):
            mask = bins == b
            rows.append({
                "Outcome": outcome, "Bin": f"{BINS[b]:.0%}-{BINS[b + 1]:.0%}", "Matches": mask.sum(),
                "Predicted": predicted[mask].mean(), "Actual": happened[mask].mean(),
            })
    table = pd.DataFrame(rows)
    table["Gap"] = table["Actual"] - table["Predicted"]
    # Standard error of the actual frequency, to tell real gaps from noise.
    table["SE"] = np.sqrt(table["Predicted"] * (1 - table["Predicted"]) / table["Matches"])
    return table


def scores(probabilities, actual):
    """Log loss, multi-class Brier score and expected calibration error (mean over outcomes)."""
    p = probabilities[OUTCOMES].to_numpy()
    onehot = np.column_stack([(actual == o).to_numpy() for o in OUTCOMES]).astype(float)
    table = reliability(probabilities, actual)
    ece = (table["Matches"] * table["Gap"].abs()).groupby(table["Outcome"]).sum().mean() / len(actual)
    return {
        "Log loss": -np.log((p * onehot).sum(axis=1)).mean(),
        "Brier": ((p - onehot) ** 2).sum(axis=1).mean(),
        "ECE": ece,
    }


def temperature_scale(probabilities, temperature):
    logits = np.log(probabilities[OUTCOMES].to_numpy()) / temperature
    scaled = np.exp(logits - logits.max(axis=1, keepdims=True))
    return pd.DataFrame(scaled / scaled.sum(axis=1, keepdims=True), columns=OUTCOMES, index=probabilities.index)


def fit_temperature(probabilities, actual):
    onehot = np.column_stack([(actual == o).to_numpy() for o in OUTCOMES])

    def loss(temperature):
        return -np.log(temperature_scale(probabilities, temperature).to_numpy()[onehot]).mean()

    return minimize_scalar(loss, bounds=(0.5, 2.0), method="bounded").x


def walk_forward_temperature(probabilities, actual, seasons):
    """Each season scaled by a temperature fitted on earlier seasons' walk-forward predictions."""
    parts, temperatures = [], {}
    ordered = sorted(seasons.unique())
    for season in ordered[1:]:
        earlier = seasons < season
        temperatures[season] = fit_temperature(probabilities.loc[earlier], actual.loc[earlier])
        now = seasons == season
        parts.append(temperature_scale(probabilities.loc[now], temperatures[season]))
    return pd.concat(parts), temperatures


def market_probabilities(matches, columns):
    implied = 1 / matches[columns].to_numpy()
    return pd.DataFrame(implied / implied.sum(axis=1, keepdims=True), columns=OUTCOMES, index=matches.index)


def show_reliability(name, probabilities, actual):
    table = reliability(probabilities, actual)
    flagged = table["Gap"].abs() > 2 * table["SE"]
    table["Flag"] = np.where(flagged, "*", "")
    print(f"\n{name}: reliability (* = gap beyond 2 standard errors)")
    for column in ["Predicted", "Actual", "Gap", "SE"]:
        table[column] = table[column].map(lambda value: f"{value:.3f}")
    print(table.to_string(index=False))


def main():
    matches = attach_odds(pd.read_csv(OUTPUT_FILE, dtype={"Season": str}, parse_dates=["Date"]))
    for forecast_time, set_name in SELECTED.items():
        probabilities = walk_forward_probabilities(matches, FEATURE_SETS[set_name])
        scored = matches.loc[probabilities.index]
        actual, seasons = scored["FTR"], scored["Season"]
        decision = seasons.ne(TEST_SEASON)
        market = market_probabilities(scored, ODDS[forecast_time])
        print(f"\n===== {forecast_time}: {set_name} =====")

        show_reliability("Model, decision seasons", probabilities.loc[decision], actual.loc[decision])
        show_reliability("Pinnacle, decision seasons", market.loc[decision], actual.loc[decision])

        print("\nMean predicted vs actual by season:")
        by_season = pd.concat({
            "Model": probabilities.groupby(seasons).mean(),
            "Actual": pd.get_dummies(actual)[OUTCOMES].astype(float).groupby(seasons).mean(),
        }, axis=1)
        print(by_season.round(3).to_string())

        scaled, temperatures = walk_forward_temperature(probabilities, actual, seasons)
        print("\nTemperature by season (fitted on earlier seasons; >1 softens, <1 sharpens):",
              {season: round(float(t), 3) for season, t in temperatures.items()})
        # Temperature scaling needs one earlier season, so the comparison starts a season later.
        compare = scaled.index[seasons.loc[scaled.index].ne(TEST_SEASON)]
        test = scaled.index[seasons.loc[scaled.index].eq(TEST_SEASON)]
        rows = {}
        for label, index in [("Decision", compare), ("Test", test)]:
            for name, p in [("Model", probabilities), ("Model + temperature", scaled), ("Pinnacle", market)]:
                rows[(label, name)] = scores(p.loc[index], actual.loc[index])
        print(f"\nScores ({seasons.loc[compare].min()}-{seasons.loc[compare].max()} decision, {TEST_SEASON} test):")
        print(pd.DataFrame(rows).T.round(4).to_string())

        gain = (
            -np.log(temperature_scale(probabilities.loc[compare], 1).to_numpy()[
                np.column_stack([(actual.loc[compare] == o) for o in OUTCOMES])])
            + np.log(scaled.loc[compare].to_numpy()[np.column_stack([(actual.loc[compare] == o) for o in OUTCOMES])])
        )
        rng = np.random.default_rng(0)
        boot = gain[rng.integers(0, len(gain), (5000, len(gain)))].mean(axis=1)
        print(f"Temperature log-loss change (decision): {-gain.mean():+.4f} "
              f"95% CI [{-np.percentile(boot, 97.5):+.4f}, {-np.percentile(boot, 2.5):+.4f}]")


if __name__ == "__main__":
    main()
