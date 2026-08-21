"""Reusable feature-family and feature-set definitions for Moneyball models.

This module centralizes *which columns belong together* so model notebooks do
not redefine feature combinations independently.

Why this lives outside the notebooks
------------------------------------
03, the cluster experiment, and the final-model notebook all need to refer to
the same feature families. If each notebook rebuilds those lists separately,
the project can quietly drift into testing different inputs under the same
feature-set name.

The feature engineering itself remains in ``feature_engineering.py``.
This module only groups already-created columns into model-ready sets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


# ---------------------------------------------------------------------
# Feature-family definitions
# ---------------------------------------------------------------------
#
# These prefixes are created by 02_feature_engineering.
# Keep the names stable because downstream notebooks will refer to the
# family names rather than individual columns wherever possible.

FEATURE_FAMILY_PREFIXES: dict[str, str] = {
    "season": "season_",
    "core": "core_",
    "bat": "bat_",
    "pitch": "pitch_",
    "field": "field_",
    "ctx": "ctx_",
    "raw": "raw_",
}


# ---------------------------------------------------------------------
# Candidate feature sets used for the broad model benchmark
# ---------------------------------------------------------------------
#
# These are deliberately meaningful combinations rather than every possible
# subset of columns. The aim is to compare baseball feature families while
# keeping the search interpretable.
#
# Human note:
# ``season`` is included in the domain sets because the primary project target
# is total season wins. A model predicting total W should know how many games
# were played instead of inferring season length indirectly from workload
# statistics.

DEFAULT_FEATURE_SET_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "core": (
        "season",
        "core",
    ),
    "core_bat": (
        "season",
        "core",
        "bat",
    ),
    "core_pitch": (
        "season",
        "core",
        "pitch",
    ),
    "core_bat_pitch": (
        "season",
        "core",
        "bat",
        "pitch",
    ),
    "core_bat_pitch_field": (
        "season",
        "core",
        "bat",
        "pitch",
        "field",
    ),
    "all_domain": (
        "season",
        "core",
        "bat",
        "pitch",
        "field",
        "ctx",
    ),
    "raw": (
        "season",
        "raw",
    ),
    "all_domain_raw": (
        "season",
        "core",
        "bat",
        "pitch",
        "field",
        "ctx",
        "raw",
    ),
}


def get_feature_families(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return columns grouped by their 02 feature-engineering prefixes.

    Parameters
    ----------
    df:
        A feature-engineered Moneyball DataFrame, normally ``train_fe`` or
        ``pred_fe``.

    Returns
    -------
    dict[str, list[str]]
        Mapping such as ``{"core": [...], "bat": [...], ...}``.

    Notes
    -----
    Metadata (``meta_*``) and target (``target_*``) columns are intentionally
    excluded because they are not model feature families.
    """
    families: dict[str, list[str]] = {}

    for family_name, prefix in FEATURE_FAMILY_PREFIXES.items():
        families[family_name] = [
            col for col in df.columns if col.startswith(prefix)
        ]

    return families


def combine_families(
    families: Mapping[str, Sequence[str]],
    *family_names: str,
) -> list[str]:
    """Combine named feature families while preserving column order.

    Duplicate columns are removed defensively. In the current design families
    should not overlap, but preserving this check makes future extensions safer.
    """
    features: list[str] = []

    for family_name in family_names:
        if family_name not in families:
            raise KeyError(
                f"Unknown feature family: {family_name!r}. "
                f"Available families: {sorted(families)}"
            )

        for column in families[family_name]:
            if column not in features:
                features.append(column)

    return features


def get_candidate_feature_sets(
    df: pd.DataFrame,
    definitions: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[str]]:
    """Build named model feature sets from a feature-engineered DataFrame.

    By default this returns the eight feature sets used in the broad
    model × feature benchmark in 03.

    A different definition mapping can be supplied later for focused
    experiments without changing the global defaults.
    """
    families = get_feature_families(df)

    definitions = definitions or DEFAULT_FEATURE_SET_DEFINITIONS

    feature_sets: dict[str, list[str]] = {}

    for set_name, family_names in definitions.items():
        feature_sets[set_name] = combine_families(
            families,
            *family_names,
        )

    return feature_sets


def validate_feature_sets(
    train_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
) -> None:
    """Validate feature sets before they are used by model notebooks.

    Checks:
    - no duplicate columns inside a feature set;
    - every feature exists in both train and prediction data;
    - metadata and target columns cannot leak into model features.

    Raises
    ------
    ValueError
        If any validation rule fails.
    """
    errors: list[str] = []

    for set_name, features in feature_sets.items():
        features = list(features)

        if len(features) != len(set(features)):
            errors.append(f"{set_name}: contains duplicate columns")

        missing_train = sorted(set(features) - set(train_df.columns))
        if missing_train:
            errors.append(
                f"{set_name}: missing from train: {missing_train}"
            )

        missing_pred = sorted(set(features) - set(pred_df.columns))
        if missing_pred:
            errors.append(
                f"{set_name}: missing from prediction: {missing_pred}"
            )

        forbidden = [
            column
            for column in features
            if column.startswith("meta_") or column.startswith("target_")
        ]
        if forbidden:
            errors.append(
                f"{set_name}: contains metadata/target columns: {forbidden}"
            )

    if errors:
        message = "Feature-set validation failed:\n- " + "\n- ".join(errors)
        raise ValueError(message)


__all__ = [
    "FEATURE_FAMILY_PREFIXES",
    "DEFAULT_FEATURE_SET_DEFINITIONS",
    "get_feature_families",
    "combine_families",
    "get_candidate_feature_sets",
    "validate_feature_sets",
]