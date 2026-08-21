"""Reusable feature-engineering logic for the Moneyball project.

The notebooks keep the analysis narrative, observations, decisions, and audits.
This module keeps the repeatable transformation logic in one place so every
downstream notebook can use the same feature definitions.

Design principles
-----------------
- Preserve raw source counts under ``raw_*`` for controlled experiments.
- Use baseball-appropriate denominators for engineered rates.
- Keep season length explicit when predicting total season wins.
- Keep team/franchise/year identifiers as metadata only; do not construct
  history-derived features from them unless identity reconciliation is solved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moneyball import project_config as cfg


# Raw source statistics deliberately retained without normalization.
# These are experimental source features, not an automatic final model set.
RAW_COUNT_COLS = [
    "R", "RA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "SB",
    "ER", "CG", "SHO", "SV", "IPouts", "HA", "HRA", "BBA",
    "SOA", "E", "DP",
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide aligned Series and return NaN where the denominator is zero.

    Keeping this behavior in one helper prevents silent +/-inf values from
    entering the model tables. The canonical notebook later asserts that
    engineered numeric features contain no missing or infinite values.
    """
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Return traceability columns that must not be used as model features.

    Decision:
        ID, yearID, teamID, and franchID are retained so predictions can be
        traced back to source rows.

    Reasoning:
        Earlier data-quality work found that team/franchise/year combinations
        are not reliable enough for historical feature construction without
        further reconciliation. Therefore these columns are metadata only.
    """
    out = pd.DataFrame(index=df.index)

    out["meta_ID"] = df[cfg.ID_COL]
    out["meta_yearID"] = df[cfg.YEAR_COL]
    out["meta_teamID"] = df[cfg.TEAM_COL]
    out["meta_franchID"] = df[cfg.FRANCHISE_COL]

    return out


def add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return league/era context features.

    Context flags are already normalized indicators, so they are preserved as
    0/1 values rather than divided by games played.
    """
    out = pd.DataFrame(index=df.index)

    # MLB runs per game captures the overall scoring environment for the year.
    out["ctx_mlb_rpg"] = df["mlb_rpg"].astype(float)

    context_flags = [
        c
        for c in df.columns
        if c.startswith("era_")
        or (c.startswith("decade_") and c != "decade_label")
    ]

    for c in context_flags:
        out[f"ctx_{c}"] = df[c].astype(int)

    return out


def add_raw_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve selected source counting statistics exactly as supplied.

    Decision:
        Raw counts are retained under an explicit ``raw_`` prefix.

    Reasoning:
        We do not want feature engineering to irreversibly replace source
        information. Downstream model experiments can compare raw totals
        against cleaner rate/efficiency features as separate feature families.
    """
    out = pd.DataFrame(index=df.index)

    for c in RAW_COUNT_COLS:
        out[f"raw_{c}"] = df[c]

    return out


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interpretable baseball features with appropriate denominators.

    Important:
        This function intentionally does NOT apply a blanket /G transform.
        Different statistics measure different opportunities:

        - runs / games
        - batting / at-bats or approximate plate appearances
        - pitching / innings
        - already-normalized rates remain unchanged
    """
    out = pd.DataFrame(index=df.index)

    # -----------------------------------------------------------------
    # Season length
    # -----------------------------------------------------------------
    # Keep games explicitly because the primary target is total season wins.
    # The model should not have to infer schedule length indirectly from
    # workload statistics such as total innings pitched.
    out["season_G"] = df[cfg.GAMES_COL].astype(float)

    # -----------------------------------------------------------------
    # Core scoring / run-prevention features
    # -----------------------------------------------------------------
    out["core_R_pg"] = safe_divide(df["R"], df[cfg.GAMES_COL])
    out["core_RA_pg"] = safe_divide(df["RA"], df[cfg.GAMES_COL])
    out["core_RDiff_pg"] = safe_divide(
        df["R"] - df["RA"],
        df[cfg.GAMES_COL],
    )

    # Pythagorean expectation uses the same exponent selected in 01 EDA.
    r_pow = df["R"].astype(float).pow(cfg.PYTHAG_EXPONENT)
    ra_pow = df["RA"].astype(float).pow(cfg.PYTHAG_EXPONENT)

    out["core_Pythag_win_pct"] = safe_divide(
        r_pow,
        r_pow + ra_pow,
    )

    # League-context adjustment:
    #   > 1 for scoring above the league run environment
    #   > 1 for allowing more runs than the league environment
    out["core_R_adj"] = safe_divide(out["core_R_pg"], df["mlb_rpg"])
    out["core_RA_adj"] = safe_divide(out["core_RA_pg"], df["mlb_rpg"])
    out["core_RDiff_adj"] = safe_divide(
        out["core_RDiff_pg"],
        df["mlb_rpg"],
    )

    # -----------------------------------------------------------------
    # Batting quality / efficiency
    # -----------------------------------------------------------------
    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    total_bases = (
        singles
        + 2 * df["2B"]
        + 3 * df["3B"]
        + 4 * df["HR"]
    )

    # HBP and sacrifice flies are unavailable, so AB + BB is only an
    # approximation of plate appearances. Feature names make that explicit.
    approx_pa = df["AB"] + df["BB"]

    out["bat_AVG"] = safe_divide(df["H"], df["AB"])
    out["bat_OBP_approx"] = safe_divide(
        df["H"] + df["BB"],
        approx_pa,
    )
    out["bat_SLG"] = safe_divide(total_bases, df["AB"])
    out["bat_OPS_approx"] = out["bat_OBP_approx"] + out["bat_SLG"]
    out["bat_ISO"] = out["bat_SLG"] - out["bat_AVG"]

    # Opportunity-based batting rates. These are intentionally not /G.
    out["bat_HR_per_AB"] = safe_divide(df["HR"], df["AB"])
    out["bat_BB_rate_approx"] = safe_divide(df["BB"], approx_pa)
    out["bat_SO_rate_approx"] = safe_divide(df["SO"], approx_pa)
    out["bat_XBH_per_H"] = safe_divide(
        df["2B"] + df["3B"] + df["HR"],
        df["H"],
    )
    out["bat_SB_per_onbase_approx"] = safe_divide(
        df["SB"],
        df["H"] + df["BB"],
    )

    # -----------------------------------------------------------------
    # Pitching quality / efficiency
    # -----------------------------------------------------------------
    innings = df["IPouts"].astype(float) / 3.0

    # ERA is already normalized by innings; do not divide it by games again.
    out["pitch_ERA"] = df["ERA"].astype(float)

    out["pitch_WHIP"] = safe_divide(
        df["HA"] + df["BBA"],
        innings,
    )
    out["pitch_K_BB"] = safe_divide(df["SOA"], df["BBA"])

    # Per-nine rates are standard workload-adjusted pitching measures.
    out["pitch_K9"] = 9.0 * safe_divide(df["SOA"], innings)
    out["pitch_BB9"] = 9.0 * safe_divide(df["BBA"], innings)
    out["pitch_H9"] = 9.0 * safe_divide(df["HA"], innings)
    out["pitch_HR9"] = 9.0 * safe_divide(df["HRA"], innings)

    # CG/SHO/SV are game-level outcomes; games are the natural denominator.
    out["pitch_CG_pg"] = safe_divide(df["CG"], df[cfg.GAMES_COL])
    out["pitch_SHO_pg"] = safe_divide(df["SHO"], df[cfg.GAMES_COL])
    out["pitch_SV_pg"] = safe_divide(df["SV"], df[cfg.GAMES_COL])

    # -----------------------------------------------------------------
    # Fielding
    # -----------------------------------------------------------------
    # FP is already a rate, so preserve its meaning.
    out["field_FP"] = df["FP"].astype(float)

    # Detailed fielding chances are not supplied. Per-game exposure is the
    # most interpretable denominator available for E and DP.
    out["field_E_pg"] = safe_divide(df["E"], df[cfg.GAMES_COL])
    out["field_DP_pg"] = safe_divide(df["DP"], df[cfg.GAMES_COL])

    return out


def build_feature_table(
    df: pd.DataFrame,
    *,
    is_train: bool,
) -> pd.DataFrame:
    """Build one aligned Moneyball feature table without mutating the input.

    Train receives both target formulations:
    - target_W: total season wins
    - target_W_pct: win percentage

    Prediction receives no target columns.
    """
    parts = [
        add_metadata(df),
        add_context_features(df),
        add_raw_source_features(df),
        add_domain_features(df),
    ]

    out = pd.concat(parts, axis=1)

    if is_train:
        # Keep both targets so downstream modeling can compare formulations
        # without rebuilding the feature table.
        out["target_W"] = df[cfg.TARGET_COL].astype(float)
        out["target_W_pct"] = safe_divide(
            df[cfg.TARGET_COL],
            df[cfg.GAMES_COL],
        )

    return out


__all__ = [
    "RAW_COUNT_COLS",
    "safe_divide",
    "add_metadata",
    "add_context_features",
    "add_raw_source_features",
    "add_domain_features",
    "build_feature_table",
]
