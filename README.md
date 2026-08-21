# MLB Season Wins Prediction — Moneyball Analytics

## Project Overview

This project predicts the **total number of MLB regular-season wins (`W`)** for each team-season using historical team-level data from the Lahman Baseball Database.

The project was structured with notebooks with the following responsibility:

1. validate source data;
2. understand the data and its relationship;
3. engineer reusable baseball features;
4. select the best model + feature-set combinations;
5. test optional cluster features;
6. train and validate the final model / ensemble.

The primary evaluation metric is **Mean Absolute Error (MAE) in total season wins**.

---

## Final Result

The best validated submission configuration is:

- **90% ElasticNet**
- **10% Ridge**
- both using the `all_domain_raw` feature set
- predictions rounded to the nearest whole win

Validation results:

| Validation | MAE |
|---|---:|
| Best rounded OOF MAE | **2.7031** |
| Rounded chronological MAE | **2.3613** |
| Best float OOF MAE | 2.7244 |
| Best float chronological MAE | 2.3687 |

The best continuous blend was approximately **55% ElasticNet + 45% Ridge**, but the best rounded OOF result came from **90% ElasticNet + 10% Ridge**.

The final decision is based primarily on **5-fold GroupKFold by season**, with a later-season chronological holdout used as a secondary robustness check.

---

## Data

### Training data

`data/raw/data_year_team_franchise.csv`

- 1,812 team-season rows
- contains historical outcomes including `W`
- includes batting, pitching, fielding, season, and league-context measures

### Prediction data

`data/raw/predict_year_team_franchise.csv`

- 453 rows
- same source feature schema where applicable
- does not contain the target `W`
- contains `ID` for final submission mapping

---

## Cleaned Project Workflow

```text
00_data_quality.ipynb
        ↓
01_eda.ipynb
        ↓
02_feature_engineering.ipynb
        ↓
data/processed/train_fe.csv
data/processed/pred_fe.csv
        ↓
03_model_feature_selection.ipynb
        ↓
outputs/model_selection/
        ↓
04_cluster_feature_experiment.ipynb
        ↓
outputs/cluster_experiment/
        ↓
05_final_model_training.ipynb
        ↓
outputs/final_model/
outputs/submissions/submission_predict.csv
```

---

## 00 — Data Quality

**Purpose:** validate the raw train and prediction files before modeling.

Key checks include:

- schema alignment;
- duplicate / missing values;
- train-only columns;
- identity-key reliability;
- season/team/franchise reconciliation;
- numeric consistency;
- source row traceability.

### Important finding

`teamID`, `franchID`, and `yearID` are retained for metadata and validation, but are **not trusted as historical identity keys for lag or franchise-history features** without further reconciliation.

During data-quality review, we found that the combination of team / franchise / year is not consistently unique or stable across the dataset. In some seasons, the same franchise or team identifier can appear in ways that make a simple chronological `groupby(...).shift(1)` unsafe. This means that "previous row for this team/franchise" is not always guaranteed to mean "the same team in the immediately previous season."

That creates two risks:
- a lag feature may link the wrong historical row;
- a missing season can cause `shift(1)` to use an older season as if it were the previous year.

For example, if the available rows are:
```text
BOS 1966
BOS 1968
```

a simple `shift(1)` would assign the 1966 values as the lag for 1968, even though 1967 is missing.

Because of these identity and continuity issues, the feature pipeline does not currently create:
- `lag_*` features;
- year-over-year deltas;
- franchise historical averages;
- franchise rolling statistics.

---

## 01 — Exploratory Data Analysis

**Purpose:** understand the relationship between team performance and season wins before engineering features.

EDA covers:

- target distribution;
- season-length variation;
- scoring and run prevention;
- batting and pitching relationships;
- era / decade context;
- Pythagorean win expectation.

The project uses a **Pythagorean exponent of 1.83** based on the EDA work.

### Findings
- Feature engineering uses baseball-appropriate denominators, not blindly divide all counting stats by G.
- `season_G` is kept explicitly because the target is **total wins** (`W`).
- Both engineered and selected `raw_` features are retained so downstream modeling can test whether original season totals still add value.
- Lag and franchise-history features are excluded because `teamID` / `franchID` / `yearID` are not reliable enough for safe chronological linking.
- `train_fe.csv` and `pred_fe.csv` are the canonical feature tables used by all downstream model experiments.

---

## 02 — Feature Engineering

**Purpose:** create the canonical model-ready feature tables.

Outputs:

```text
data/processed/train_fe.csv
data/processed/pred_fe.csv
```

### Design principles

1. Preserve original measurements rather than silently overwriting them.
2. Do **not** blindly divide every counting statistic by games played.
3. Use baseball-appropriate denominators.
4. Keep `G` explicitly because the target is total season wins.
5. Keep identity fields as metadata only.
6. Retain both engineered and selected raw features so downstream modeling can test whether the original measurements still add signal.

### Feature families

#### `season_`

Season-length information such as:

- `season_G`

#### `core_`

High-level run and win-expectation features such as:

- runs per game;
- runs allowed per game;
- run differential per game;
- Pythagorean win percentage;
- league-adjusted run measures.

#### `bat_`

Batting quality / efficiency features such as:

- batting average;
- approximate OBP;
- slugging;
- approximate OPS;
- ISO;
- HR / AB;
- BB and SO opportunity rates.

#### `pitch_`

Pitching quality / efficiency features such as:

- ERA;
- WHIP;
- K/BB;
- K/9;
- BB/9;
- H/9;
- HR/9;
- CG / game;
- SHO / game;
- SV / game.

#### `field_`

Fielding features such as:

- fielding percentage;
- errors / game;
- double plays / game.

#### `ctx_`

League / era / decade context.

#### `raw_`

Selected original season totals retained for controlled comparison.

### Important result

The strongest regularized linear models performed best when using:

```text
all_domain_raw
```

This shows that the original season totals still contain useful scale / opportunity information that is not fully captured by engineered rate features.

---

## 03 — Model & Feature Selection

**Purpose:** compare model choice and feature-set choice together.

The notebook compares **5 models × 8 reusable feature sets = 40 combinations**.

### Models

- Ridge
- ElasticNet
- Random Forest
- HistGradientBoosting
- LightGBM

### Feature sets

Defined centrally in `src/moneyball/feature_sets.py`:

- `core`
- `core_bat`
- `core_pitch`
- `core_bat_pitch`
- `core_bat_pitch_field`
- `all_domain`
- `raw`
- `all_domain_raw`

### Validation

Primary screening uses:

```text
5-fold GroupKFold grouped by yearID
```

The same folds and MAE metric are used for all combinations.

### Key results

| Model | Best feature set | MAE |
|---|---|---:|
| **ElasticNet** | `all_domain_raw` | **2.7278** |
| **Ridge** | `all_domain_raw` | **2.7293** |
| HistGradientBoosting | `all_domain_raw` | 3.0702 |
| LightGBM | `core_pitch` | 3.1159 |
| Random Forest | `core_pitch` | 3.1538 |

### Decision

Carry forward:

- ElasticNet;
- Ridge;
- HGB as the strongest nonlinear model;
- LightGBM only as an ensemble-diversity challenger.

Random Forest is dropped from the final path.

### Coefficient inspection

Ridge and ElasticNet both benefit from the raw features, but the coefficient patterns show substantial multicollinearity.

ElasticNet gives the cleaner interpretation and places meaningful weight on raw measures such as:

- `IPouts`;
- `AB`;
- `SV`;
- `CG`;
- `R`.

This supports keeping `all_domain_raw` rather than removing raw totals simply because they are correlated with engineered rates.

---

## 04 — Cluster Feature Experiment

**Purpose:** test whether KMeans-derived team-profile features add predictive value.

The experiment compares each shortlisted model:

```text
same model + same selected feature set
vs
same model + same selected feature set + cluster distances
```

KMeans is fitted **inside each CV training fold** to avoid validation leakage.

Tested:

- K = 4
- K = 6
- K = 8

### Result

Clusters did not improve the two strongest models.

| Model | Baseline MAE | Best cluster result | Assessment |
|---|---:|---:|---|
| ElasticNet | **2.7278** | 2.7344 | worse |
| Ridge | **2.7293** | 2.7379 | worse |
| HGB | 3.0702 | **3.0644** (K=4) | tiny improvement |

### Decision

**Reject cluster features from the final pipeline.**

The small HGB improvement does not justify the additional clustering complexity, especially because HGB remains substantially weaker than Ridge and ElasticNet.

---

## 05 — Final Model Training & Ensemble

**Purpose:** choose the actual final prediction method.

### Candidate models

- ElasticNet
- Ridge
- HGB
- LightGBM

Each model uses the feature set on which it performed best in 03.

### Primary validation

OOF predictions using:

```text
5-fold GroupKFold by yearID
```

OOF predictions are saved for every training row so residuals and ensembles can be compared fairly.

### Secondary validation

Chronological holdout:

```text
earliest ~80% of unique seasons → training
latest ~20% of unique seasons   → validation
```

This is a robustness check, not the primary optimization target.

### Ensemble analysis

Residual correlation is inspected before blending.

The main ensemble search tests:

- ElasticNet + Ridge over a broad weight range;
- ElasticNet + HGB with small nonlinear weights;
- ElasticNet + LightGBM with small nonlinear weights.

The nonlinear models did not improve enough to justify their added complexity.

### Final ensemble

Best continuous configuration:

```text
55% ElasticNet + 45% Ridge
Float OOF MAE: 2.7244
Float chronological MAE: 2.3687
```

Best rounded configuration:

```text
90% ElasticNet + 10% Ridge
Rounded OOF MAE: 2.7031
Rounded chronological MAE: 2.3613
```

### Final decision

Use:

```text
90% ElasticNet
10% Ridge
rounded to whole wins
```

The final blend was selected because it produced the lowest rounded OOF MAE among the tested configurations and remained strong on the chronological holdout.

---

## Validation Strategy

The cleaned project deliberately does **not** optimize across every possible CV scheme.

Earlier experiments compared:

- GroupKFold;
- KFold;
- chronological holdout;
- random holdout.

The cleaned pipeline uses:

### Primary

**GroupKFold by `yearID`**

Why:

- keeps complete seasons together;
- prevents teams from the same season appearing in both train and validation;
- gives a conservative measure of generalization across seasons.

### Secondary

**Chronological holdout**

Why:

- checks whether the selected model remains sensible when trained on earlier seasons and evaluated on later seasons.

Ordinary KFold and random holdout are no longer used as final decision metrics because they can mix teams from the same season across train and validation.

---

## Reusable Project Code

Shared logic is kept under:

```text
src/moneyball/
```

### `project_config.py`

Shared:

- project paths;
- dataset filenames;
- common column names;
- modeling constants;
- notebook display settings.

### `feature_engineering.py`

Reusable deterministic feature-construction pipeline.

### `feature_sets.py`

Single source of truth for:

- feature families;
- named feature-set combinations;
- feature-set validation.

### `model_factory.py`

Single source of truth for screening model definitions:

- Ridge;
- ElasticNet;
- Random Forest;
- HGB;
- LightGBM.

This prevents different notebooks from silently using different preprocessing or model settings under the same model name.

---

## Project Structure

```text
baseball_v2/
│
├── data/
│   ├── raw/
│   │   ├── data_year_team_franchise.csv
│   │   └── predict_year_team_franchise.csv
│   ├── interim/
│   └──  processed/
│       ├── train_fe.csv
│       └── pred_fe.csv
│
├── notebooks/
│   ├── 00_data_quality.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_feature_selection.ipynb
│   ├── 04_cluster_feature_experiment.ipynb
│   └── 05_final_model_training.ipynb
│
├── outputs/
│   ├── data_quality/
│   ├── model_selection/
│   ├── cluster_experiment/
│   ├── final_model/
│   └── submissions/
│       └── submission_predict.csv
│
├── src/
│   └── moneyball/
│       ├── __init__.py
│       ├── project_config.py
│       ├── feature_engineering.py
│       ├── feature_sets.py
│       └── model_factory.py
│
├── environment.yml
├── pyproject.toml
├── .gitignore
└── README.md
```

---

## Environment

The project uses a Conda environment:

```bash
conda activate mball
```

The local Moneyball package is installed in editable mode:

```bash
python -m pip install -e .
```

This allows notebooks to import shared project code directly:

```python
from moneyball import project_config as cfg
from moneyball import feature_engineering as fe
from moneyball import feature_sets as fs
from moneyball import model_factory as mf
```

---

## Main Lessons From the Cleanup

1. **Feature engineering should preserve meaning.**  
   Counting statistics should not be mechanically divided by `G`; different baseball measures have different natural denominators.

2. **Raw features are not automatically noise.**  
   The best Ridge and ElasticNet models improved substantially when raw totals were retained alongside engineered features.

3. **Model and feature selection interact.**  
   Feature sets were evaluated per model instead of choosing one universal set first.

4. **Regularized linear models fit this dataset very well.**  
   ElasticNet and Ridge materially outperformed RF, HGB, and LightGBM.

5. **More complex features did not automatically help.**  
   KMeans cluster features were tested and rejected.

6. **A nonlinear ensemble was worth testing, but did not win.**  
   HGB and LightGBM added little compared with a Ridge / ElasticNet blend.

7. **Rounding matters for this target.**  
   Whole-win predictions improved both OOF and chronological MAE.

---

## Final Reproducible Path

To reproduce the final result, run the notebooks in order:

```text
00 → 01 → 02 → 03 → 04 → 05
```

The final submission is written to:

```text
outputs/submissions/submission_predict.csv
```