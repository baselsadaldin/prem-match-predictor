"""Dixon-Coles goals model: team attack/defence ratings with time decay and a low-score correction.

Each team's goals are Poisson with log mean = attack(team) - defence(opponent) + home advantage,
plus shared terms for promoted teams (not in the previous Premier League season), who have no
Premier League history to rate them on. Matches are weighted by exp(-days ago / half-life * ln 2).
The Dixon-Coles rho term then adjusts 0-0, 1-0, 0-1 and 1-1, which independent Poisson misprices.

The model is refitted every week on matches strictly before that week's Monday, so every
forecast uses only earlier results. Settings are chosen on seasons before the test season.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor

from prepare_data import HISTORY_SEASONS, SEASONS, load_all_matches

OUTCOMES = ["H", "D", "A"]
MAX_GOALS = 10
# Older matches carry almost no weight beyond about four half-lives.
WINDOW_DAYS = 1095


def team_rows(matches):
    """Two rows per match: the home side's goals, then the away side's goals."""
    seasons = sorted(matches["Season"].unique())
    previous = dict(zip(seasons[1:], seasons[:-1]))
    played = set(zip(matches["HomeTeam"], matches["Season"]))
    promoted = {
        (team, season) for team, season in played
        if season in previous and (team, previous[season]) not in played
    }
    rows = []
    for attack, defence, goals, home in [
        ("HomeTeam", "AwayTeam", "FTHG", 1), ("AwayTeam", "HomeTeam", "FTAG", 0)
    ]:
        rows.append(pd.DataFrame({
            "Date": matches["Date"].to_numpy(),
            "Attack": matches[attack].to_numpy(),
            "Defence": matches[defence].to_numpy(),
            "Goals": matches[goals].to_numpy(),
            "Home": home,
            "PromotedAttack": [(t, s) in promoted for t, s in zip(matches[attack], matches["Season"])],
            "PromotedDefence": [(t, s) in promoted for t, s in zip(matches[defence], matches["Season"])],
        }))
    return rows


def design(rows, teams):
    """Feature matrix: one-hot attacker, one-hot defender, home, promoted attacker/defender."""
    attack = (rows["Attack"].to_numpy()[:, None] == teams).astype(float)
    defence = (rows["Defence"].to_numpy()[:, None] == teams).astype(float)
    extra = rows[["Home", "PromotedAttack", "PromotedDefence"]].to_numpy(dtype=float)
    return np.hstack([attack, defence, extra])


def tau(home_goals, away_goals, mu_home, mu_away, rho):
    """Dixon-Coles adjustment to the probability of low scores."""
    out = np.ones_like(mu_home)
    out = np.where((home_goals == 0) & (away_goals == 0), 1 - mu_home * mu_away * rho, out)
    out = np.where((home_goals == 0) & (away_goals == 1), 1 + mu_home * rho, out)
    out = np.where((home_goals == 1) & (away_goals == 0), 1 + mu_away * rho, out)
    out = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, out)
    return out


def fit_week(history_home, history_away, weights, alpha):
    teams = np.unique(np.concatenate([history_home["Attack"], history_home["Defence"]]))
    rows = pd.concat([history_home, history_away], ignore_index=True)
    model = PoissonRegressor(alpha=alpha, max_iter=1000)
    model.fit(design(rows, teams), rows["Goals"], sample_weight=np.concatenate([weights, weights]))
    mu_home = model.predict(design(history_home, teams))
    mu_away = model.predict(design(history_away, teams))
    home_goals, away_goals = history_home["Goals"].to_numpy(), history_away["Goals"].to_numpy()

    def negative_log_likelihood(rho):
        adjustment = tau(home_goals, away_goals, mu_home, mu_away, rho)
        return -(weights * np.log(np.clip(adjustment, 1e-10, None))).sum()

    # Keep all four low-score corrections positive for the fitted means.
    lower = max(-0.2, float(np.max(-1 / mu_home)), float(np.max(-1 / mu_away)))
    upper = min(0.2, float(np.min(1 / (mu_home * mu_away))))
    rho = minimize_scalar(negative_log_likelihood, bounds=(lower + 1e-10, upper - 1e-10), method="bounded").x
    return model, teams, rho


def outcome_probabilities(mu_home, mu_away, rho):
    # New fixtures can have more extreme means than the fitted fixtures.
    # Constrain rho to valid low-score corrections for each forecast separately.
    lower = np.maximum(-1 / mu_home, -1 / mu_away)
    upper = np.minimum(1, 1 / (mu_home * mu_away))
    rho = np.clip(rho, lower + 1e-10, upper - 1e-10)[:, None, None]
    goals = np.arange(MAX_GOALS + 1)
    home = poisson.pmf(goals[None, :], mu_home[:, None])
    away = poisson.pmf(goals[None, :], mu_away[:, None])
    grid = home[:, :, None] * away[:, None, :]
    h, a = np.meshgrid(goals, goals, indexing="ij")
    grid = grid * tau(h[None], a[None], mu_home[:, None, None], mu_away[:, None, None], rho)
    grid /= grid.sum(axis=(1, 2), keepdims=True)
    return np.column_stack([
        (grid * (h > a)).sum(axis=(1, 2)),
        (grid * (h == a)).sum(axis=(1, 2)),
        (grid * (h < a)).sum(axis=(1, 2)),
    ])


def walk_forward(matches, half_life_days, alpha, first_season):
    """H/D/A probabilities and expected goals for every match from `first_season` on."""
    matches = matches.sort_values("Date", kind="stable").reset_index(drop=True)
    home_rows, away_rows = team_rows(matches)
    week = matches["Date"] - pd.to_timedelta(matches["Date"].dt.weekday, unit="D")
    scored = matches["Season"] >= first_season
    results = []
    for monday in sorted(week[scored].unique()):
        past = (matches["Date"] < monday) & (matches["Date"] >= monday - pd.Timedelta(days=WINDOW_DAYS))
        age = (monday - matches.loc[past, "Date"]).dt.days.to_numpy()
        weights = 0.5 ** (age / half_life_days)
        model, teams, rho = fit_week(home_rows.loc[past], away_rows.loc[past], weights, alpha)
        now = scored & week.eq(monday)
        mu_home = model.predict(design(home_rows.loc[now], teams))
        mu_away = model.predict(design(away_rows.loc[now], teams))
        probabilities = outcome_probabilities(mu_home, mu_away, rho)
        results.append(pd.DataFrame(
            np.column_stack([probabilities, mu_home, mu_away]),
            columns=OUTCOMES + ["DCHomeGoals", "DCAwayGoals"], index=matches.index[now],
        ))
    return matches, pd.concat(results).sort_index()


def season_log_loss(matches, probabilities):
    actual = matches.loc[probabilities.index, "FTR"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    loss = -np.log(probabilities[OUTCOMES].to_numpy()[np.arange(len(actual)), actual])
    return pd.Series(loss, index=probabilities.index).groupby(matches.loc[probabilities.index, "Season"]).mean()


if __name__ == "__main__":
    from prepare_data import TEST_SEASON
    matches = load_all_matches(HISTORY_SEASONS + SEASONS)
    grid = {}
    for half_life in [180, 365, 730]:
        for alpha in [0.0005, 0.005, 0.01]:
            matches, probabilities = walk_forward(matches, half_life, alpha, first_season="1617")
            by_season = season_log_loss(matches, probabilities)
            decision = by_season.drop(TEST_SEASON).mean()
            grid[(half_life, alpha)] = by_season
            print(f"half-life {half_life:>3} days, alpha {alpha}: decision-season log loss {decision:.4f}")
    best = min(grid, key=lambda key: grid[key].drop(TEST_SEASON).mean())
    print(f"\nChosen: half-life {best[0]} days, alpha {best[1]}")
    print(grid[best].round(4).to_string())
