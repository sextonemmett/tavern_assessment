"""
Utility functions for the Data Science skills assessment.
This file contains helper functions that candidates can use during the assessment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple, Optional
import requests
import os
import hashlib
import json
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from collections import Counter
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Skeleton functions for candidates to implement

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build group-level modeling features for the assessment.

    The output is aggregated to the `video_id × partisanship` grain (one row per
    treatment video and partisanship class), and includes:
    - `control_trump_approval`: mean approval in the control group for that partisanship
    - `treatment_trump_approval`: mean approval among treated participants for that video/partisanship
    - `average_treatment_effect`: treatment minus control
    - `increased_trump_approval`: 1 if treatment effect > 0 else 0
    - `avg_persuadability`: mean `persuadability_score` for that video/partisanship (treated only)
    - `maxdiff_mean`: video-level maxdiff score (treated videos)
    - `embedding_0..embedding_383`: SentenceTransformer embedding of the video transcript text

    Data cleaning performed (per assessment notes):
    - Recode `trump_approval == 11` to 1
    - Drop `video_id == 55` (missing text/maxdiff metadata)

    Notes:
    - Embeddings are computed from `text` via `get_embeddings()` and cached on disk.
    - This function prints progress messages to stdout.

    Args:
        df: Raw participant-level dataframe for Experiment 1, containing both treated
            and control rows plus per-video metadata. Expected columns include
            `treated`, `trump_approval`, `video_id`, `partisanship`,
            `persuadability_score`, `text`, and `maxdiff_mean`.
            
    Returns:
        DataFrame at `video_id × partisanship` grain with target + modeling features.

    Raises:
        ValueError: If `df` is empty or contains no treated rows.
    """
    if df is None or df.empty:
        raise ValueError("compute_features received an empty dataframe.")
    
    print(f"[compute_features] start: input shape={df.shape}")
    working_df = df.copy()
    
    # Fix mis-coded approval values.
    invalid_trump_values = working_df["trump_approval"] == 11
    if invalid_trump_values.any():
        print(f"Corrected {invalid_trump_values.sum()} trump_approval values from 11 to 1.")
        working_df.loc[invalid_trump_values, "trump_approval"] = 1
    else:
        print("[compute_features] trump_approval: no 11 values found")
    
    # Drop video 55 rows where metadata is entirely missing.
    missing_video_mask = working_df["video_id"].eq(55)
    if missing_video_mask.any():
        print(f"Dropping video 55 rows due to missing text/maxdiff/sample_size ({missing_video_mask.sum()} rows).")
        working_df = working_df.loc[~missing_video_mask].copy()
    else:
        print("[compute_features] video 55: no rows found")
    
    working_df["video_id"] = working_df["video_id"].astype("Int64")
    
    print("[compute_features] computing control means by partisanship")
    control_df = working_df[working_df["treated"] == 0]
    control_means = (
        control_df.groupby("partisanship")["trump_approval"]
        .mean()
        .rename("control_trump_approval")
        .reset_index()
    )
    
    treatment_df = working_df[working_df["treated"] == 1].copy()
    if treatment_df.empty:
        raise ValueError("No treated rows were found in the dataframe.")
    print(f"[compute_features] treated rows={len(treatment_df):,}, control rows={len(control_df):,}")
    
    group_cols = ["video_id", "partisanship"]
    
    print("[compute_features] aggregating to video_id × partisanship")
    agg_specs = {
        "treatment_trump_approval": ("trump_approval", "mean"),
        "avg_persuadability": ("persuadability_score", "mean"),
    }
    
    grouped = treatment_df.groupby(group_cols).agg(**agg_specs).reset_index()
    grouped = grouped.merge(control_means, on="partisanship", how="left")
    grouped["average_treatment_effect"] = (
        grouped["treatment_trump_approval"] - grouped["control_trump_approval"]
    )
    grouped["increased_trump_approval"] = (grouped["average_treatment_effect"] > 0).astype(int)
    print(f"[compute_features] grouped shape={grouped.shape}, positives={int(grouped['increased_trump_approval'].sum())}")
    
    print("[compute_features] building per-video metadata + text embeddings")
    video_meta = (
        treatment_df[["video_id", "text", "maxdiff_mean"]]
        .drop_duplicates(subset=["video_id"])
        .reset_index(drop=True)
    )
    video_meta["text"] = video_meta["text"].fillna("")
    embeddings = get_embeddings(video_meta["text"].astype(str).tolist())
    embedding_cols = [f"embedding_{i}" for i in range(embeddings.shape[1])]
    embedding_frame = pd.DataFrame(embeddings, columns=embedding_cols)
    video_meta = pd.concat([video_meta.reset_index(drop=True), embedding_frame], axis=1)
    
    print("[compute_features] merging aggregated outcomes with video metadata/features")
    features_df = grouped.merge(video_meta, on="video_id", how="left")
    
    # Preserve a consistent column order so downstream modeling is predictable.
    ordered_cols = [
        "video_id",
        "partisanship",
        "text",
        "maxdiff_mean",
        "avg_persuadability",
        "treatment_trump_approval",
        "control_trump_approval",
        "average_treatment_effect",
        "increased_trump_approval",
    ]
    ordered_cols += embedding_cols
    remaining_cols = [col for col in features_df.columns if col not in ordered_cols]
    features_df = features_df[ordered_cols + remaining_cols]
    
    print(f"[compute_features] done: output shape={features_df.shape}")
    return features_df


def compute_text_embedding_pca(
    texts: Union[pd.Series, List[str], pd.DataFrame],
    n_components: int = 5
) -> pd.DataFrame:
    """
    Compute PCA scores from SentenceTransformer embeddings of text.

    This helper is useful for exploratory feature creation, but note that it fits PCA
    on *all provided rows*. For leakage-safe modeling, prefer fitting PCA inside the
    cross-validation pipeline (as done in `predict_increased_trump_approval`).

    Args:
        texts: Either a Series/List of text strings or a DataFrame with `video_id` and `text`.
        n_components: Number of PCA components to compute.
        
    Returns:
        DataFrame containing PCA score columns (`text_pca_1..text_pca_k`) and, when
        the input is a DataFrame, a `video_id` column for alignment.

    Raises:
        ValueError: If `n_components <= 0` or DataFrame input is missing required columns.
    """
    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")
    
    print(f"[compute_text_embedding_pca] start: n_components={n_components}")
    video_ids: Optional[pd.Series] = None
    if isinstance(texts, pd.DataFrame):
        if "text" not in texts.columns or "video_id" not in texts.columns:
            raise ValueError("DataFrame input must include 'video_id' and 'text' columns.")
        video_ids = texts["video_id"].reset_index(drop=True)
        text_series = texts["text"].reset_index(drop=True)
    else:
        text_series = pd.Series(texts)
    
    text_series = text_series.fillna("").astype(str)
    print(f"[compute_text_embedding_pca] embedding {len(text_series):,} texts")
    embeddings = get_embeddings(text_series.tolist())
    n_rows, n_dims = embeddings.shape
    
    component_cols = [f"text_pca_{i+1}" for i in range(n_components)]
    component_df = pd.DataFrame(
        np.zeros((n_rows, n_components)),
        index=text_series.index,
        columns=component_cols,
    )
    
    if n_rows == 0:
        return _attach_video_id(component_df, video_ids)
    
    effective_components = min(n_components, n_rows, n_dims)
    if effective_components == 0:
        return _attach_video_id(component_df, video_ids)
    
    print(f"[compute_text_embedding_pca] running PCA with {effective_components} components on dim={n_dims}")
    pca = PCA(n_components=effective_components)
    transformed = pca.fit_transform(embeddings)
    for idx in range(effective_components):
        component_df.iloc[:, idx] = transformed[:, idx]
    
    print("[compute_text_embedding_pca] done")
    return _attach_video_id(component_df, video_ids)


def _attach_video_id(component_df: pd.DataFrame, video_ids: Optional[pd.Series]) -> pd.DataFrame:
    """
    Insert a `video_id` column when available.
    """
    if video_ids is None:
        return component_df.reset_index(drop=True)
    result = component_df.copy()
    result.insert(0, "video_id", video_ids.values)
    return result


def predict_increased_trump_approval(df: pd.DataFrame) -> np.ndarray:
    """
    Train/evaluate an XGBoost classifier to predict `increased_trump_approval`.

    Workflow:
    1. Calls `compute_features(df)` to create a `video_id × partisanship` dataset.
    2. Restricts the modeling feature set to variables computable from the assessment's
       allowed inputs: `video_id`, `partisanship`, `persuadability_score`, `text`,
       and `maxdiff_mean`:
       - `partisanship` (categorical, one-hot encoded)
       - `avg_persuadability` and `maxdiff_mean` (numeric)
       - `embedding_*` reduced to 5 dimensions via PCA fit inside CV
    3. Evaluates with leakage-safe cross-validation:
       - Outer CV: `StratifiedGroupKFold` grouping by `video_id` and stratifying on
         `partisanship × target` to preserve subgroup class balance
       - Inner CV (per outer fold): Bayesian hyperparameter tuning via `BayesSearchCV`
         using the same grouped/stratified split strategy

    Output/side effects:
    - Prints progress, best hyperparameters per fold, fold AUCs, mean/std AUC,
      plus per-partisanship AUC on out-of-fold predictions.

    Args:
        df: Raw participant-level dataframe.
            
    Returns:
        NumPy array `[mean_auc, std_auc]` across outer folds.

    Raises:
        ValueError: If required columns are missing, features contain missing values,
            or a fold contains only one target class.
    """
    print("[predict_increased_trump_approval] start")
    # Step 1: Generate features
    features_df = compute_features(df)
    print(f"[predict_increased_trump_approval] compute_features done: shape={features_df.shape}")
    
    target_col = "increased_trump_approval"
    if target_col not in features_df.columns:
        raise ValueError(f"Expected '{target_col}' to be present in compute_features output.")
    
    y = features_df[target_col].astype(int).to_numpy()
    groups = features_df["video_id"].astype(int).to_numpy()
    
    # Stratify by partisanship × target so each fold reflects subgroup base rates.
    stratify_labels = (
        features_df["partisanship"].astype(str) + "__" + features_df[target_col].astype(int).astype(str)
    ).to_numpy()
    
    # Exclude leakage-prone or non-feature columns.
    drop_cols = {
        "video_id",
        "text",
        target_col,
        "average_treatment_effect",
        "treatment_trump_approval",
        "control_trump_approval",
    }
    feature_df = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns])
    print(f"[predict_increased_trump_approval] feature matrix: shape={feature_df.shape}")
    
    categorical_cols = ["partisanship"] if "partisanship" in feature_df.columns else []
    numeric_cols = [c for c in feature_df.columns if c not in categorical_cols]

    if feature_df.isna().any().any():
        na_cols = feature_df.columns[feature_df.isna().any()].tolist()
        raise ValueError(
            f"Missing values detected in feature matrix (columns: {na_cols}). "
            "Either clean upstream in compute_features or re-enable imputation."
        )
    
    embedding_cols = [c for c in feature_df.columns if c.startswith("embedding_")]
    allowed_feature_cols = ["partisanship", "avg_persuadability", "maxdiff_mean"] + embedding_cols
    missing_allowed = [c for c in allowed_feature_cols if c not in feature_df.columns]
    if missing_allowed:
        raise ValueError(f"Expected allowed feature columns missing from compute_features output: {missing_allowed}")

    # Restrict model features to those allowed by the prompt.
    feature_df = feature_df[allowed_feature_cols]
    categorical_cols = ["partisanship"]
    non_embedding_numeric_cols = ["avg_persuadability", "maxdiff_mean"]

    print(
        f"[predict_increased_trump_approval] using allowed features only: "
        f"{len(non_embedding_numeric_cols)} numeric + {len(embedding_cols)} embeddings + partisanship"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", "passthrough", non_embedding_numeric_cols),
            ("embed_pca", PCA(n_components=5, svd_solver="randomized", random_state=42), embedding_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    
    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_aucs: List[float] = []
    oof_proba = np.full(shape=(len(feature_df),), fill_value=np.nan, dtype=float)
    fold_aucs_by_partisanship: Dict[str, List[float]] = {}
    best_params_per_fold: List[Dict[str, object]] = []

    # More-thorough tuning while respecting dataset size (~378 rows) and grouped CV:
    # Bayesian optimization via scikit-optimize (BayesSearchCV).
    bayes_space = {
        "model__max_depth": Integer(2, 6),
        "model__learning_rate": Real(1e-2, 2e-1, prior="log-uniform"),
        "model__n_estimators": Integer(150, 800),
        "model__min_child_weight": Integer(1, 10),
        "model__subsample": Real(0.6, 1.0),
        "model__colsample_bytree": Real(0.6, 1.0),
        "model__gamma": Real(0.0, 5.0),
        "model__reg_alpha": Real(0.0, 1.0),
        "model__reg_lambda": Real(0.5, 5.0, prior="log-uniform"),
    }
    search_iters = 30
    print(f"[predict_increased_trump_approval] starting outer CV: n_splits={cv.get_n_splits()}")
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(feature_df, stratify_labels, groups=groups), start=1):
        print(f"[predict_increased_trump_approval] fold {fold_idx}: split train={len(train_idx):,} test={len(test_idx):,}")
        X_train = feature_df.iloc[train_idx]
        y_train = y[train_idx]
        X_test = feature_df.iloc[test_idx]
        y_test = y[test_idx]

        groups_train = groups[train_idx]
        stratify_train = stratify_labels[train_idx]

        inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
        inner_splits = list(inner_cv.split(X_train, stratify_train, groups=groups_train))
        print(f"[predict_increased_trump_approval] fold {fold_idx}: bayes search start (iters={search_iters}, inner_splits={len(inner_splits)})")

        search = BayesSearchCV(
            estimator=clf,
            search_spaces=bayes_space,
            n_iter=search_iters,
            scoring="roc_auc",
            cv=inner_splits,
            n_jobs=-1,
            refit=True,
            random_state=42,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        best_params_per_fold.append(search.best_params_)
        print(f"Fold {fold_idx} best params: {search.best_params_} (inner AUC: {search.best_score_:.4f})")

        best_clf = search.best_estimator_
        y_proba = best_clf.predict_proba(X_test)[:, 1]
        oof_proba[test_idx] = y_proba
        
        if len(np.unique(y_test)) < 2:
            raise ValueError(
                f"Fold {fold_idx} has only one class in y_test; adjust CV strategy or n_splits."
            )
        auc = roc_auc_score(y_test, y_proba)
        fold_aucs.append(float(auc))
        print(f"Fold {fold_idx} AUC: {auc:.4f}")

        # per-partisanship AUC within this fold.
        for partisanship in sorted(features_df["partisanship"].astype(str).unique().tolist()):
            mask = features_df.iloc[test_idx]["partisanship"].astype(str).eq(partisanship).to_numpy()
            if not mask.any():
                continue
            y_sub = y_test[mask]
            if len(np.unique(y_sub)) < 2:
                continue
            auc_sub = roc_auc_score(y_sub, y_proba[mask])
            fold_aucs_by_partisanship.setdefault(partisanship, []).append(float(auc_sub))
    
    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else 0.0
    print(f"Mean AUC: {mean_auc:.4f} (std: {std_auc:.4f})")

    if best_params_per_fold:
        params_counter = Counter([tuple(sorted(p.items())) for p in best_params_per_fold])
        most_common_params, count = params_counter.most_common(1)[0]
        print(f"Most common best params across folds (n={count}): {dict(most_common_params)}")

    # Per-partisanship AUC on out-of-fold predictions (each row predicted once).
    if np.isnan(oof_proba).any():
        raise ValueError("OOF predictions contain NaNs; CV split may have failed to cover all rows.")
    for partisanship in sorted(features_df["partisanship"].astype(str).unique().tolist()):
        mask = features_df["partisanship"].astype(str).eq(partisanship).to_numpy()
        y_sub = y[mask]
        if len(np.unique(y_sub)) < 2:
            print(f"{partisanship} OOF AUC: undefined (single class).")
            continue
        auc_oof = roc_auc_score(y_sub, oof_proba[mask])
        print(f"{partisanship} OOF AUC: {auc_oof:.4f}")

    # If available, also summarize fold-wise subgroup AUCs.
    for partisanship, aucs in fold_aucs_by_partisanship.items():
        if len(aucs) < 2:
            continue
        print(
            f"{partisanship} fold AUC mean: {float(np.mean(aucs)):.4f} "
            f"(std: {float(np.std(aucs, ddof=1)):.4f}, n_folds: {len(aucs)})"
        )
    
    return np.array([mean_auc, std_auc])
    

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        
    Returns:
        DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)


def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    Compute SentenceTransformer embeddings for a list of texts with on-disk caching.

    - Model is fixed to `paraphrase-MiniLM-L3-v2` (do not change).
    - Embeddings are cached in `data/cached_embeddings` by an MD5 hash of the text.
    - Empty/None/blank strings return a zero vector.

    Notes:
    - The first run may download model artifacts (network access may be required).
    - Subsequent runs reuse cached embeddings on disk.
    
    Args:
        texts: List of text strings to embed.
        
    Returns:
        NumPy array of shape `(len(texts), 384)`.
    """
    # Do not change the model name and cache directory
    model_name = 'paraphrase-MiniLM-L3-v2'
    embedding_dim = 384
    cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cached_embeddings"))
 
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load the model
    model = SentenceTransformer(model_name) 
    
    # Initialize array for all embeddings
    all_embeddings = np.zeros((len(texts), embedding_dim))
    
    # Process each text
    for i, text in enumerate(texts):
        if text is None or not isinstance(text, str) or text.strip() == "":
            # Handle empty or None text
            all_embeddings[i] = np.zeros(embedding_dim)
            continue
            
        # Create a hash of the text to use as a filename
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        cache_file = os.path.join(cache_dir, f"{text_hash}.npy")
        
        if os.path.exists(cache_file):
            # Load from cache
            try:
                embedding = np.load(cache_file)
                all_embeddings[i] = embedding
            except Exception as e:
                print(f"Error loading cached embedding: {e}")
                # Compute embedding if loading fails
                embedding = model.encode([text])[0]
                all_embeddings[i] = embedding
                # Save to cache
                np.save(cache_file, embedding)
        else:
            # Compute embedding
            embedding = model.encode([text])[0]
            all_embeddings[i] = embedding
            # Save to cache
            np.save(cache_file, embedding)
    
    return all_embeddings
