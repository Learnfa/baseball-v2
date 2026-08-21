from pathlib import Path
import pandas as pd


# ---------------------------------------------------------------------
# Dataset schema
# ---------------------------------------------------------------------

TRAIN_FILENAME    = "data_year_team_franchise.csv"
PRED_FILENAME     = "predict_year_team_franchise.csv"
TRAIN_FE_FILENAME = "train_fe.csv"
PRED_FE_FILENAME  = "pred_fe.csv"

ID_COL            = "ID"
YEAR_COL          = "yearID"
TEAM_COL          = "teamID"
FRANCHISE_COL     = "franchID"

TARGET_COL        = "W"
GAMES_COL         = "G"


# ---------------------------------------------------------------------
# Modeling constants
# ---------------------------------------------------------------------

# Baseball-specific Pythagorean exponent selected during EDA.
PYTHAG_EXPONENT = 1.83

# Protect divisions from zero denominators.
EPS = 1e-12


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

def find_project_root(start: Path | None = None) -> Path:
    """
    Locate the Moneyball project root.

    We identify the root by the presence of data/raw rather than relying
    on the notebook's current working directory. This allows notebooks
    to run whether Jupyter was started from the project root or from
    notebooks/.
    """
    start = (start or Path.cwd()).resolve()

    for candidate in [start, *start.parents]:
        if (candidate / "data" / "raw").is_dir():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. "
        "Expected a data/raw directory in the current directory "
        "or one of its parents."
    )


PROJECT_ROOT  = find_project_root()

RAW_DIR        = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR    = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR     = PROJECT_ROOT / "outputs"

# 01_eda
TRAIN_PATH           = RAW_DIR / TRAIN_FILENAME
PRED_PATH            = RAW_DIR / PRED_FILENAME
# 02_feature_engineering
TRAIN_FE_PATH        = PROCESSED_DIR / TRAIN_FE_FILENAME
PRED_FE_PATH         = PROCESSED_DIR / PRED_FE_FILENAME
# 03_model_feature_selection
MODEL_RESULT_PATH    = OUTPUT_DIR / "model_selection" / "03_model_feature_results.csv"
MODEL_SHORTLIST_PATH = OUTPUT_DIR / "model_selection" / "03_model_shortlist.csv"
# 04_cluster_feature
CLUSTER_RESULT_PATH  = OUTPUT_DIR / "cluster_experiment" / "04_cluster_feature_results.csv"
CLUSTER_SUMMARY_PATH = OUTPUT_DIR / "cluster_experiment" / "04_cluster_feature_summary.csv"
# 05_model_training
FINAL_DIR            = OUTPUT_DIR / "final_model"
SUBMISSION_DIR       = OUTPUT_DIR / "submissions"
OOF_PATH             = FINAL_DIR / "05_oof_predictions.csv"
FINAL_RESULT_PATH    = FINAL_DIR / "05_model_results.csv"
RESIDUAL_CORR_PATH   = FINAL_DIR / "05_residual_correlations.csv"
CHRONO_PATH          = FINAL_DIR / "05_chronological_results.csv"
ENSEMBLE_PATH        = FINAL_DIR / "05_ensemble_results.csv"
DECISION_PATH        = FINAL_DIR / "05_final_decision.csv"
SUBMISSION_PATH      = SUBMISSION_DIR / "submission.csv"

def ensure_project_dirs() -> None:
    """Create project-generated directories when they do not yet exist."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Notebook presentation
# ---------------------------------------------------------------------

def configure_notebook() -> None:
    """Apply display settings shared by project notebooks."""
    pd.set_option("display.max_columns", 120)
    pd.set_option("display.width", 180)


def show_project_paths() -> None:
    """Print the active project paths for notebook traceability."""
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Train        : {TRAIN_PATH}")
    print(f"Prediction   : {PRED_PATH}")
    print(f"Processed    : {PROCESSED_DIR}")