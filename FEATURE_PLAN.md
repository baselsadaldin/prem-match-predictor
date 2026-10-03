# Feature plan: beyond Elo

Goal: find out whether recent form and squad market values improve on the Elo-only model, without leaking future information. Claude implements each phase and records the results here.

## Ground rules (apply to every phase)

- **Pre-match only.** Every feature value must be computable before kickoff. Anything from the match itself (its goals, shots, half-time score) is off-limits.
- **Whitelist.** New features enter the model only by being added to `FEATURE_COLUMNS`.
- **Odds stay out.** Betting odds (`B365H`, `PSH`, `AvgH`, ...) are the benchmark for milestone 5, not features.
- **Three-way split by season.**
  - Train: 2018/19 to 2022/23 (about 1,900 matches)
  - Validation: 2023/24, used to choose features and settings
  - Test: 2024/25, scored once per phase and never used to make decisions
- **Metric order.** Log loss first, then calibration, then accuracy. Always compare against the Elo-only model on the same rows.
- **Checks.** Every new feature gets a validation check in the style of `add_elo_features`: raise an error on missing values, impossible values or a date that isn't strictly before the match.

## Phase 1: multiple seasons

Feature work starts here because 188 training matches can't support more than one or two features.

1. Make `load_matches` take a season code (`"1819"`, ..., `"2425"`). Cache each season as `data/E0_<code>.csv` and add a `Season` column.
2. Keep the extra columns you'll need later (`HS`, `AS`, `HST`, `AST`), but don't let them reach `FEATURE_COLUMNS`.
3. Concatenate the seasons, then run `add_elo_features` on the combined frame. Fix any new team-name mismatches in `TEAM_ALIASES`; the unmapped-teams check will name them.
4. Replace the date split with the season split above. `load_training_data` should return train, validation and test.
5. Re-run the Elo-only model and record its validation and test log loss and accuracy. **These become the baseline for every later phase.**

**Things to decide**

- 2019/20 and 2020/21 were partly played without crowds, and home advantage almost disappeared. Should they stay in? Try both and compare validation log loss.
- The Elo archive switches source after June 2025 (see `TRAINING_GUIDE.md`). That doesn't affect these seasons, but re-check it if you ever add 2025/26.

**Done when:** the dataset holds about 2,660 matches from 7 seasons with no missing Elo values, and there's a baseline results table.

## Phase 2: recent form

1. Build a team-match table with two rows per match, one for each team. Columns: `Season`, `Date`, `Team`, `IsHome`, `GoalsFor`, `GoalsAgainst`, `Points`, `ShotsOnTargetFor`, `ShotsOnTargetAgainst`.
2. Sort it by team and date. Within each team, `shift(1)` first and then `rolling(5)` mean, so a match never counts toward its own form.
3. Start with three form features:
   - points per game over the last 5 matches
   - goal difference over the last 5 matches
   - shots-on-target difference over the last 5 matches
4. Merge the form values back onto the matches for both teams, and use differences (home minus away) as features.
5. Handle the first matches of each season: decide whether form carries over from last season, and what promoted teams get. Write the choice down.
6. Testing:
   - Add a test that recomputes form by hand for one team in a few matches and asserts it matches.
   - Add a test that changing a match's result doesn't change that match's own features.
7. Compare on the validation season:
   - Elo only
   - Elo plus each form feature on its own
   - Elo plus all three

   Keep only what lowers validation log loss.

**Done when:** a validation results table exists for each feature set, and the selected set has been scored once on test.

## Phase 3: squad market values

1. Data source: [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets). Don't scrape Transfermarkt directly; their terms forbid it.
2. First version: squad value at the start of each season, summed over the players registered to each club. Use the valuations dated before the season's first match.
3. Map club names to Football-Data names, and add a check for unmapped clubs.
4. Feature: `log(home value) - log(away value)`. Values are heavily skewed, so use the log.
5. Compare on the validation season: the Phase 2 model with and without this feature.
6. Optional later improvement: update the values at each valuation date during the season, using the latest valuation strictly before each match. It costs more work, because squads change in January.

**Done when:** a validation comparison exists, and the feature is kept or dropped based on it.

## Phase 4: back to calibration (milestone 4)

Using the final feature set:
- Make reliability tables for H, D and A on the validation season.
- Compare each outcome's mean predicted probability with how often it actually happened.
- Then score the test season once and move on to the market comparison (milestone 5).

## Results log

Fill in as you go.

| Phase | Features | Val log loss | Val accuracy | Test log loss | Test accuracy |
|---|---|---|---|---|---|
| 0 | EloDifference (2024/25 only, Jan split) | — | — | 0.969* | 56.8%* |
| — | Class-frequency baseline (2018/19–2022/23) | 1.054 | 46.1% | 1.081 | 40.8% |
| 1 | EloDifference | 0.927 | 57.6% | 0.988 | 52.1% |

\* Jan–May 2025 only; not comparable with the full-season test rows.

**Phase 1 decision: no-crowd seasons stay in.** Validation log loss with all 5 training seasons is 0.9270, without 2019/20 and 2020/21 it is 0.9242, and without 2020/21 only it is 0.9251. A gap of 0.003 is noise at 380 matches, and dropping them costs 40% of the training data.

**Phase 2 decisions.** Form uses each team's last 5 matches. It carries over from the previous season, which comes from 2017/18 for the first training season. It restarts after promotion; a promoted team's first match gets the average form of promoted teams in the training seasons. `check_features.py` holds the leakage checks.

Validation log loss by feature set (Elo only: 0.9270):

| Feature set | Val log loss |
|---|---|
| Elo + points form | 0.9273 |
| Elo + goal-difference form | 0.9271 |
| **Elo + shots-on-target form** | **0.9165** |
| Elo + all three form features | 0.9197 |

- **Shots on target help; points and goals don't.** Points and goal-difference form add nothing on top of Elo, which is itself built from results. Shots on target carry information Elo doesn't have.
- **The gain isn't noise.** A paired bootstrap of the difference has a 95% interval of [−0.018, −0.003]. The gain is similar for windows of 5, 8 and 10 matches (windows 3 and 15: about −0.007, with intervals touching zero), so the window was left at the planned 5 rather than tuned.
- **Test agrees, by less.** Test log loss went from 0.988 to 0.985: a smaller gain, in the same direction.

**xG form attempt (not used).**
- **Understat:** its robots.txt disallows all automated access, so it wasn't used.
- **[RezaGooner/english-premier-league-match-dataset](https://github.com/RezaGooner/english-premier-league-match-dataset) (CC BY-NC 4.0):** it joined cleanly, but its xG values were unusable. From 2019/20 onward, per-match xG difference correlates about 0.0 with goal difference and shots-on-target difference; real xG should be about 0.6–0.7, as it is in that file's 2018/19 rows. Example: Newcastle 5–1 Aston Villa (12 Aug 2023, 13 vs 6 shots on target) is recorded as xG 0.5 vs 2.3. The values look scrambled across matches.
- **Kept from the attempt:** form now skips missing values and averages the last 5 recorded ones. This gives identical results for the current features, since none are missing.
- **Still open:** xG needs a reliable source that allows downloading. FBref removed its xG in January 2026. A Kaggle Understat dump exists, but it has no licence and stops in September 2024.

**Phase 3 decisions.** `squad.py` builds two features from the [Transfermarkt dataset](https://github.com/dcaribou/transfermarkt-datasets) (CC0).
- **`SquadValueDiff`:** log of the top-25 squad value, home minus away. A player counts for the club named on his latest valuation dated strictly before the match. The club on each valuation is historical (Declan Rice is listed at West Ham until his first Arsenal valuation), so there is no hindsight in squad membership. This is known days before kickoff.
- **`LineupValueDiff`:** log of the starting XI's value, using each starter's latest valuation before the match. It captures injuries, suspensions and rotation, which is the useful part of team news. Lineups are announced about an hour before kickoff, so this model is a T−60 min forecast. For milestone 5 it must be compared with closing odds, not opening odds.
- **Coverage:** all 31 clubs map one-to-one, every match has 11 starters for both clubs, and the dates agree with Football-Data.

Validation gains over Elo + SoT form (0.9165), with 95% bootstrap intervals:

| Added | Gain | 95% interval |
|---|---|---|
| Squad value | −0.005 | [−0.010, +0.000] |
| Lineup value | −0.013 | [−0.021, −0.004] |
| Both | −0.009 | — |

**Test (2024/25), scored once per forecast time: the gains did not hold.**

| Added | Test log loss | Change | 95% interval |
|---|---|---|---|
| Squad value | 0.986 | +0.002 | [−0.003, +0.006] |
| Lineup value | 0.988 | +0.003 | [−0.004, +0.010] |

Both are slightly worse than Elo + SoT form (0.985) and within noise. A likely reason is that 2024/25 broke the link between money and results: Tottenham had the 5th most valuable squad and finished 17th, and Man United had the 6th and finished 15th. Train and validation together cover only 6 seasons, so one unusual season can change the answer. The features stay in the pipeline, and the decision isn't re-made from test results.
| 2 | EloDifference + ShotsOnTargetDiffFormDiff | 0.917 | 59.0% | 0.985 | 53.2% |
| 3a | + SquadValueDiff (before lineups) | 0.911 | 58.4% | 0.986 | 52.4% |
| 3b | + LineupValueDiff (after lineups, ~1h before kickoff) | 0.904 | 59.0% | 0.988 | 51.6% |

## Phase 3b: ten training seasons and a walk-forward backtest

**More data.** Training now covers 2013/14–2022/23: 3,800 matches, double before. 2012/13 is loaded only as form history, because Transfermarkt has no lineups for it. Six more clubs were added to `TM_CLUBS`. Validation and test are unchanged.

**Backtest.** One validation season proved too noisy: 2023/24 and 2024/25 disagreed about squad and lineup value. `backtest.py` trains on all earlier seasons and scores each season from 2016/17 on. Decisions use 2016/17–2023/24 (8 seasons, 3,040 matches), which excludes the test season.

| Added | Log-loss change | 95% interval | Seasons better |
|---|---|---|---|
| SoT form, over Elo | −0.0038 | [−0.0075, −0.0002] | 6/8 |
| Squad value, over Elo + SoT | −0.0016 | [−0.0037, +0.0004] | 5/8 |
| **Lineup value, over Elo + SoT** | **−0.0042** | **[−0.0071, −0.0014]** | **6/8** |

Walk-forward accuracy (2016/17–2023/24):

| Model | Accuracy |
|---|---|
| Elo | 55.3% |
| + SoT form | 55.5% |
| + squad value | 55.8% (9 more correct out of 3,040) |
| + lineup value | 55.8% |

**Decision.**
- **Before lineups:** Elo + SoT form. Squad value is dropped, since its gain isn't distinguishable from noise.
- **After lineups:** Elo + SoT form + lineup value.

2024/25 is one of lineup value's 3 bad seasons out of 9.

| Phase | Features (10 training seasons) | Val log loss | Val acc | Test log loss | Test acc |
|---|---|---|---|---|---|
| 3b | Elo + SoT form (before lineups) | 0.915 | 57.9% | 0.986 | 52.1% |
| 3b | Elo + SoT form + lineup value (after lineups) | 0.904 | 59.7% | 0.989 | 51.8% |

**xG sources checked and rejected:**
- **FootyStats:** CSVs need an account, the site was down when checked, and its terms are unverified.
- **TheStatsDontLie:** only season totals for each team, and All Rights Reserved.
- **footballxg.com:** a paid subscription with an undisclosed xG model, and All Rights Reserved.
- **Statof (statof.com):** its terms forbid scraping, crawling or automated extraction, and redistribution without written permission. Its robots.txt calls itself "explicit anti-scraping" and backs this with IP blocking and rate limits. It has no export or public API (`/api/` is disallowed), and it doesn't say where its xG comes from or how many seasons it covers.
- **xg.football:** per-match xG looks plausible: Newcastle 5–1 Aston Villa, 12 Aug 2023, is 3.41 vs 1.77, and its shots on target match Football-Data. But xG starts only in 2023/24; pages from 2016/17, 2020/21 and 2022/23 have shots but no xG. That leaves no xG in any training season (2013/14–2022/23). The site has no terms page or licence ("© 2026"), and its data appears to come from API-Football (api-sports.io), which uses an undisclosed xG model.

## Phase 4: shot-based xG and team news

**Shot xG.** No source with a licence that allows downloading has shot locations for these seasons, so real xG can't be built. `ShotXGDiff` is the closest substitute: each team's shots are weighted by goals per shot, with shots on target and off target weighted separately. The weights are fitted only on 2012/13, which is never scored, so they can't see the seasons being predicted. They come out at **0.178 goals per shot on target and 0.010 per shot off target** (off-target shots include blocked ones). The fit is circular: every goal is a shot on target, so within the same match off-target shots explain almost nothing. As a result, shot xG is about 0.18 × shots on target, and its form correlates 1.00 with SoT form. `ShotsDiff` (total-shots form) was added separately, so the model could learn its own weight for all shots.

**Team news** (`team_news.py`, built from Transfermarkt `games.csv`, CC0):
- **`ManagerTenureDiff`:** log of days since each club's current manager (or caretaker) took his first match, capped at 365, home minus away. The data starts in August 2012, so tenures longer than a year can't be measured. Accents are removed before comparing names ("Rúben" and "Ruben Amorim" are the same manager). Spot check: Man United show Ten Hag at the 365 cap, then a reset to 0 for van Nistelrooy (3 Nov 2024) and for Amorim (24 Nov).
- **`RestDaysDiff`:** days since each club's previous match in the league, FA Cup, Europe, Community Shield or Club World Cup, capped at 7, home minus away. The League Cup is left out because Transfermarkt has it only from 2023/24. Tottenham v Rennes (9 Dec 2021) was cancelled and awarded, so it is dropped.
- **News articles weren't used:** pre-kickoff articles back to 2013 with reliable publish times aren't freely available. Their main signal (injuries, suspensions, rotation) is already in lineup value.

Walk-forward backtest, decision seasons 2016/17–2023/24 (change vs Elo + SoT form):

| Change | Log-loss change | 95% interval | Seasons better | Accuracy |
|---|---|---|---|---|
| **Shot xG instead of SoT** | **−0.0005** | **[−0.0009, −0.0001]** | **7/8** | 55.6% (+3 matches) |
| + total-shots form | −0.0005 | [−0.0024, +0.0013] | 5/8 | — |
| + manager tenure | −0.0003 | [−0.0015, +0.0009] | 5/8 | 55.6% |
| + rest days | +0.0001 | [−0.0004, +0.0007] | 2/8 | 55.5% |
| + manager + rest | −0.0002 | [−0.0015, +0.0012] | 5/8 | — |
| Lineup value, over Elo + shot xG | −0.0042 | [−0.0071, −0.0013] | 6/8 | 55.7% |

**Decision.**
- **Shot xG replaces SoT form** in both models. Its interval excludes zero and it's better in 7 of 8 seasons. The gain is tiny, as expected for a feature that is 0.18 × SoT plus a small weight on off-target shots.
- **Manager tenure, rest days and total shots are dropped,** since none is distinguishable from noise. They stay in the CSV and the whitelist for later experiments.

| Phase | Features | Val log loss | Val acc | Test log loss | Test acc |
|---|---|---|---|---|---|
| 4 | Elo + shot xG form (before lineups) | 0.915 | 58.4% | 0.985 | 53.2% |
| 4 | Elo + shot xG form + lineup value (after lineups) | 0.903 | 60.0% | 0.989 | 51.6% |

Test scored once: before lineups, 0.986 → 0.985 log loss and 52.1% → 53.2% accuracy (4 more correct). After lineups, it's unchanged at 0.989.

## Milestone 5: betting against Pinnacle (`betting.py`)

**Setup.**
- **Probabilities:** walk-forward, so each season is predicted by a model trained only on earlier seasons.
- **Odds:** before-lineups model vs Football-Data's early Pinnacle odds (`PSH/PSD/PSA`); after-lineups model vs Pinnacle closing odds (`PSCH/PSCD/PSCA`).
- **Bets:** at most one flat 1-unit bet per match, on the outcome with the highest expected value, placed only if that value is above a threshold.
- **Threshold:** chosen on 2016/17–2023/24, then 2024/25 scored once.
- **Closing-line value (CLV):** a bet's expected value at Pinnacle's margin-free closing probabilities. It's meaningful only for early-odds bets; bets placed at closing get minus the margin by definition.

**Market check first.** On log loss, Pinnacle beats the model in every season by 0.01–0.03; for example, 2024/25 closing odds score 0.966 against our 0.985.

Decision seasons (3,040 matches):

| Model vs odds | Threshold | Bets | ROI | 95% interval | CLV |
|---|---|---|---|---|---|
| Before lineups vs early odds | EV > 0% | 2,923 | −5.2% | [−11.5%, +1.3%] | −2.3% |
| Before lineups vs early odds | **EV > 15% (chosen)** | 1,328 | −5.2% | [−15.4%, +5.6%] | −1.7% |
| After lineups vs closing odds | EV > 0% | 2,945 | −0.7% | [−7.4%, +6.3%] | −2.4% |
| After lineups vs closing odds | **EV > 10% (chosen)** | 1,893 | −0.2% | [−9.5%, +9.5%] | −2.4% |

Test 2024/25, scored once:

| Model | Bets | Profit | ROI | 95% interval | CLV |
|---|---|---|---|---|---|
| Before lineups (EV > 15%) | 104 | −10.5 units | −10.1% | [−43%, +27%] | −2.3% |
| After lineups (EV > 10%) | 202 | +10.2 units | +5.0% | [−25%, +37%] | −2.9% |

**Conclusion: no evidence of an edge.**
- **No profit beyond luck.** Every interval includes zero, and the per-season returns swing between −34% and +17%.
- **CLV is negative.** Early-odds bets lose about 2% against the closing price, so the market moves *against* the model's picks. This is the most reliable signal here, because it doesn't depend on match results.
- **The test profit is luck.** The after-lineups +5% on test comes with negative CLV and an interval 60 points wide.
- **The model "finds value" almost everywhere.** It bets on 96% of matches at average odds of about 4.5, which means it routinely disagrees with a better forecaster. Away and draw bets lost about 10%, and home bets broke even. Choosing "home only" now would be picked after seeing the results, so it doesn't count.
- **What would have to change:** the model needs information the market prices badly, not more versions of what Elo and form already capture.

## Milestone 5b: line shopping and a Dixon-Coles model

**Line shopping (`line_shopping.py`, no model).** Pinnacle's early odds, with the margin removed, are taken as the true probabilities. A 1-unit bet goes on every outcome whose best market price (`BbMx*` up to 2018/19, `Max*` from 2019/20) beats them.

| Era | Edge threshold | Bets | ROI | 95% interval | CLV |
|---|---|---|---|---|---|
| BbMx 2013/14–2018/19 | > 0% | 3,451 | +1.4% | [−5.0%, +8.0%] | +2.3% |
| Max 2019/20–2024/25 | > 0% | 3,017 | −3.9% | [−11.4%, +3.8%] | +2.8% |
| BbMx 2013/14–2018/19 | > 2% | 1,438 | +1.5% | [−10.0%, +13.8%] | +4.6% |
| Max 2019/20–2024/25 | > 2% | 1,248 | −2.4% | [−15.7%, +11.9%] | +5.4% |

- **CLV is positive in every row:** these prices beat Pinnacle's close, unlike the model's bets (CLV about −2%). This is the one place in the project where an edge shows up.
- **Realised profit isn't established.** The bets are mostly long odds (averaging 5–8), and every interval includes zero. In the Max era, positive CLV came with −2% to −4% returns.
- **The best price is optimistic:** it's the top of many bookmakers at one moment, may include exchange prices before commission, and bookmakers limit accounts that keep taking it.
- **Margin removal checked:** the power method shifts bets to shorter odds (Max era, > 0%: +1.2% ROI, [−4.7%, +7.2%]) but doesn't change the conclusion. Proportional removal stays, since it matches actual results better on long odds (odds 8–15: predicted 9.6%, actual 9.5%; power method 8.9%).

**Dixon-Coles (`dixon_coles.py`).**
- **Model:** Poisson attack and defence ratings for each team, with shared terms for promoted teams, a home-advantage term and the Dixon-Coles low-score correction (rho).
- **Fitting:** refitted every week on matches before that Monday, with weights halving every 365 days over a 3-year window, and penalty `alpha=0.005`. Half-life and alpha were chosen on 2016/17–2023/24 from {180, 365, 540, 730} × {0.0005, 0.005, 0.01, 0.02, 0.05}.

Walk-forward log loss, decision seasons 2016/17–2023/24:

| Model | Mean log loss | Change | 95% interval | Seasons better |
|---|---|---|---|---|
| Elo + shot xG (current) | 0.9558 | — | — | — |
| Dixon-Coles alone | 0.9605 | +0.0047 | [−0.0014, +0.0109] | 4/8 |
| Elo + shot xG + DC goal ratio | 0.9556 | −0.0002 | [−0.0021, +0.0017] | 6/8 |
| + DC draw probability | 0.9547 | −0.0009 more | [−0.0027, +0.0009] | 6/8 |
| Elo + shot xG + lineup + DC | 0.9517 | +0.0001 vs without DC | [−0.0017, +0.0019] | 5/8 |

**Decision: Dixon-Coles isn't adopted.** It's no better than Elo + shot xG on its own and adds nothing as a feature. Both are results-based team-strength ratings, so they carry the same information. On 2024/25 Dixon-Coles alone scores 0.972 against 0.986, the best of any model, but one season isn't grounds for a decision, and it was the worst model in 2022/23 (1.000).

## Milestone 5c: the Championship (`championship.py`)

**Setup.**
- **Pipeline:** the same one on Football-Data `E1`, 2013/14–2024/25 (552 matches a season), with Elo and form only. Transfermarkt covers top divisions only, so there is no squad or lineup value. Form restarts for every newcomer, promoted or relegated.
- **Dropped matches (8):**
  - Coventry, Rotherham and Wycombe on 12 Sep 2020: the Elo archive's first ratings after their promotion are dated 15 Sep.
  - Five matches missing some Pinnacle odds, so every model and the market are scored on the same matches.
- **Missing stats:** Bolton v Brentford (27 Apr 2019, not played because of a players' strike) has no shot stats; form skips it.
- **Settings:** shot-xG weights are fitted on Championship 2012/13. Dixon-Coles uses the Premier League settings without re-tuning.

Walk-forward log loss, decision seasons 2016/17–2023/24 (4,409 matches):

| Model | Mean log loss |
|---|---|
| Elo | 1.0493 |
| Elo + SoT form | 1.0456 |
| **Elo + shot xG form (chosen)** | **1.0450** |
| Dixon-Coles | 1.0514 |
| Pinnacle early | 1.0334 |
| Pinnacle closing | 1.0316 |

**The Championship market isn't softer for this model.** The gap to Pinnacle early odds is 0.012, against 0.014 in the Premier League (0.9558 vs about 0.942). The model beat Pinnacle in only one season (2022/23: 1.0525 vs 1.0570).

Betting with Elo + shot xG (threshold chosen on decision seasons; test scored once):

| Odds | Chosen threshold | Decision ROI | 95% interval | CLV | Test 2024/25 |
|---|---|---|---|---|---|
| Pinnacle early | EV > 20% | +2.5% (954 bets) | [−9.0%, +14.7%] | −2.1% | −12.8% (99 bets), CLV −5.5% |
| Pinnacle closing | EV > 20% | −2.0% (1,174 bets) | [−12.0%, +8.2%] | −2.7% | −5.2% (144 bets) |

Line shopping in the Championship gives smaller edges than in the Premier League: CLV +1.1% at > 0% and +2.5–3.3% at > 2%, with every return interval including zero.

**Conclusion.** The same picture as the Premier League: negative CLV, decision-season returns within noise, and a test loss. The +2.5% at EV > 20% is the best of six thresholds and has negative CLV, so it is selection noise, not an edge.

## Milestone 4: calibration (`calibration.py`)

**Setup.** Walk-forward probabilities for both selected models, decision seasons 2016/17–2023/24, with Pinnacle's margin-free odds (early for before lineups, closing for after) as the reference. Reliability is measured in 10-point probability bins for each outcome; a bin is flagged when its gap exceeds 2 standard errors. With about 22 bins per table, one flag is expected by chance.

**The model is well calibrated, about as well as Pinnacle.**

| Decision seasons | Log loss | Brier | ECE |
|---|---|---|---|
| Before lineups model | 0.9588 | 0.5675 | 0.014 |
| Pinnacle early | 0.9466 | 0.5591 | 0.017 |
| After lineups model | 0.9553 | 0.5650 | 0.016 |
| Pinnacle closing | 0.9427 | 0.5565 | 0.017 |

(ECE is the expected calibration error: the mean of |actual − predicted| across bins, averaged over H/D/A. These rows start in 2017/18, because temperature scaling needs one earlier season.)

- **Bins:** home and away probabilities track actual frequencies across the whole range. Each model has one flagged bin:
  - Before lineups: draws at 10–20% (predicted 16.8%, actual 13.5%, 526 matches).
  - After lineups: away at 60–70% (predicted 64.5%, actual 73.7%, 167 matches), a hint that big away favourites are underrated.
  - Pinnacle has flagged bins too.
- **Draws:** the model's draw probability almost never leaves 17–30%, and 2,511 of 3,040 matches fall in 20–30%. Pinnacle uses 8–35% and is right at both ends (30–40%: 224 matches, 30.8% draws). The model is calibrated on draws but can't tell which matches are draw-prone. That's a ranking weakness, not a calibration one, and it's part of the gap to the market.
- **Season-level misses are mostly about home advantage.** The model predicts 44–46% home wins every season. Actual: 37.9% in 2020/21 (no crowds) and 40.8% in 2024/25, against 48–49% in 2016/17 and 2022/23. Pinnacle misses those seasons as well. This is the biggest source of test-season error: ECE 0.036–0.040 on 2024/25, against Pinnacle's 0.042–0.043.

**Temperature scaling isn't adopted.** For each season, the temperature is fitted on the earlier seasons' walk-forward predictions. It made things worse: +0.0027 log loss before lineups, 95% interval [+0.0003, +0.0051], and +0.0025 after lineups, [−0.0000, +0.0052]. Worse on test too. After the first few seasons the fitted temperatures settle at 0.97–1.00, meaning the raw probabilities need no rescaling. The early, smaller samples asked for sharpening (0.77–0.92), which hurt.

**Possible next step:** let home advantage drift, for example by weighting recent seasons more in training. It would need to be tested walk-forward, because 2020/21 was a one-off and home advantage partly recovered afterwards.
