# Methods

This document describes the modeling workflow implemented in `assessment_utilities.py` to predict whether a treatment video increases Trump approval (relative to control) for each partisanship class, using only features computable from the provided inputs.

## 1) Data preparation and cleansing

All preparation happens inside `compute_features(df)` and is designed to output one row per `video_id × partisanship` group.

Steps:

- **Correct miscoded outcomes:** recode `trump_approval == 11` to `1`.
  - Caveat: I treat this as a clear data-entry/encoding error; if it were a distinct category, this would be inappropriate. (The downstream pipeline assumes a binary approval outcome.)

- **Drop a malformed video:** drop all rows with `video_id == 55`.
  - Reason: this video’s transcript / `maxdiff_mean` metadata are missing, so text-based and maxdiff features cannot be computed consistently for it.

- **Type normalization:** cast `video_id` to an integer type to avoid merge / grouping surprises.

## 2) Constructing the target outcome

The assessment specifies computing an average treatment effect (ATE) for each `video_id × partisanship` group as:

`ATE(video_id, partisanship) = mean(trump_approval | treated video_id, partisanship) - mean(trump_approval | control, partisanship)`

Implementation details:

- **Control baseline:** compute `control_trump_approval` as the mean of `trump_approval` among control observations within each partisanship group.
- **Treatment mean:** compute `treatment_trump_approval` as the mean of `trump_approval` among treated observations for each `video_id × partisanship`.
- **ATE:** `average_treatment_effect = treatment_trump_approval - control_trump_approval`.
- **Binary label:** create `increased_trump_approval = 1` if `average_treatment_effect > 0`, else `0`. 

Notes / caveats:

- This aggregation treats the estimated ATE as a point estimate and does not model uncertainty from cell sample sizes. In a production setting, I would consider incorporating standard errors (or using a continuous target and modeling the ATE directly), especially if some `video_id × partisanship` cells are small.
- Using a partisanship-specific control mean matches the prompt and helps remove baseline differences across Democrats / Republicans / Independents.

## 3) Feature engineering

`compute_features` produces features at the same `video_id × partisanship` grain as the target.

### Numeric features

- `avg_persuadability`: mean `persuadability_score` among treated participants within each `video_id × partisanship` group.
  - Note: assuming `persuadability_score` is pre-treatment (as described), this is safe; if it were affected by treatment, it could introduce post-treatment bias.
- `maxdiff_mean`: video-level MaxDiff score (merged from video metadata).

### Text features

- **SentenceTransformer embeddings:** compute a 384-d embedding from each video transcript using a fixed model name (`paraphrase-MiniLM-L3-v2`), with on-disk caching keyed by an MD5 hash of the text.
  - Empty / blank transcripts map to a zero vector.
  - Caveat: embedding caching assumes the transcript text is stable; if transcripts change, cached vectors should be invalidated.

### Final modeling feature set

In `predict_increased_trump_approval`, the feature set is explicitly restricted to:
- `partisanship` (categorical)
- `avg_persuadability`, `maxdiff_mean` (numeric)
- `embedding_*` (text embeddings)

This adheres to the constraint that features be computable from: `video_id`, `partisanship`, `persuadability_score`, transcript `text`, and `maxdiff_mean`.

## 4) Data splitting strategy

Evaluation is designed to reflect the out-of-sample goal: generalizing to unseen videos.

- **Outer cross-validation:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`.
  - **Groups:** `video_id`, so the same video cannot appear in both train and test within a fold (prevents leakage through text / maxdiff features).
  - **Stratification label:** `partisanship × increased_trump_approval`, to better preserve subgroup class balance across folds.

This implements a grouped, stratified CV consistent with the prompt’s requirement to use stratified K-fold and report AUC mean/std across folds.

## 5) Model selection and hyperparameter tuning

### Model class

- **Classifier:** `XGBClassifier` (XGBoost) with:
  - `objective="binary:logistic"`
  - `eval_metric="auc"`
  - `reg_lambda=1.0`
  - `tree_method="hist"`
  - `random_state=42`, `n_jobs=-1`

### Preprocessing pipeline

A single `sklearn` `Pipeline` is fit within each fold:

- `partisanship`: one-hot encoded (`OneHotEncoder(handle_unknown="ignore")`)
- `avg_persuadability`, `maxdiff_mean`: passed through as numeric
- `embedding_*`: reduced via `PCA(n_components=5, svd_solver="randomized", random_state=42)`

Important nuance: PCA is fit inside the CV pipeline, so the embedding dimensionality reduction is leakage-safe (no information from the test fold influences the PCA projection).

### Hyperparameter tuning

Within each outer fold, hyperparameters are tuned via Bayesian optimization:

- **Search method:** `BayesSearchCV`
- **Inner CV:** `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)` using the same grouping (`video_id`) and stratification (`partisanship × label`) logic as the outer fold.
- **Iterations:** 30
- **Search space:** `max_depth`, `learning_rate`, `n_estimators`, `min_child_weight`, `subsample`, `colsample_bytree`, `gamma`, `reg_alpha`, `reg_lambda`.

Caveats:

- Nested CV tuning is more compute-intensive but helps avoid optimistic bias from tuning on the same folds used for evaluation.
- The search space is intentionally constrained given the relatively small number of `video_id × partisanship` rows; very large trees or overly aggressive boosting can overfit quickly.

## 6) Evaluation methodology

Primary metric:

- **AUC (ROC AUC)** computed on each outer-fold test split, then summarized as mean and standard deviation across folds.

Additional diagnostics (printed, not required by the prompt):

- **Out-of-fold (OOF) predictions:** store predicted probabilities for each row from its held-out fold.
- **Subgroup performance:** compute OOF AUC separately for each partisanship group when both classes are present; also summarize fold-wise subgroup AUCs where defined.

Notes / caveats:

- AUC can be undefined in a subgroup if a fold (or subgroup subset) contains only one class. The code skips such subgroup AUC calculations rather than forcing a value.
- If missing values appear in the final feature matrix, the function raises an error rather than silently imputing. This is deliberate to surface upstream data issues early; if needed, it can be extended with imputation, but I prefer explicit handling in an assessment setting.

