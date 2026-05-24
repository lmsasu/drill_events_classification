# SHAP Analysis Report — GRU

**Source run:** `results_gru_2026-05-23_09-33-40`
**Explainer backend:** gradient
**Background samples (training set):** 100
**Explained samples (test set):** 200

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| `epochs` | 100 |
| `seed` | 42 |
| `deterministic` | True |
| `benchmark` | False |
| `early_stopping_patience` | 8 |
| `early_stopping_min_delta` | 0.0 |
| `grad_clip_norm` | 1.0 |
| `batch_size` | 64 |
| `dropout` | 0.33037859822247173 |
| `hidden_size` | 256 |
| `learning_rate` | 0.000359283375708533 |
| `lookback_window` | 50 |
| `num_layers` | 4 |

## Model Performance on Test Set

| Metric | Value |
|--------|-------|
| accuracy | 0.9411 |
| auc | 0.9937 |
| f1 | 0.9397 |
| loss | 0.1876 |
| precision | 0.9406 |
| recall | 0.9411 |

## Global Feature Importance (Top 3)

Mean absolute SHAP value aggregated across all explained test samples, all time steps in the lookback window, and all output classes.

| Rank | Feature | Mean \|SHAP\| |
|------|---------|--------------|
| 1 | `Speed` | 0.056150 |
| 2 | `Heading` | 0.017882 |
| 3 | `Dist` | 0.006115 |

## Output Files

| File | Description |
|------|-------------|
| `shap_summary.png` | Beeswarm plot — distribution of SHAP values per feature across test samples; dot colour encodes feature value (red = high, blue = low). |
| `shap_bar.png` | Bar chart — mean \|SHAP\| per feature, global importance ranking. |
| `shap_heatmap.png` | Heatmap — mean \|SHAP\| per time step × feature, revealing which features matter most at which position in the lookback window. |

## Interpretation Notes

- SHAP values are computed per output class using the explainer's class-wise decomposition. Importance scores reported here are averaged in absolute value across all classes to give a class-agnostic feature ranking.
- The heatmap rows correspond to consecutive time steps inside the lookback window of length **50**. Row `t-1` is the most recent observation immediately before the predicted point; row `t-N` is the oldest.
- Features concentrated near `t-1` (bottom rows of the heatmap) are primarily driven by recent dynamics; features spread uniformly across rows carry persistent long-range information.
- The beeswarm plot preserves the sign of SHAP values: a cluster of red dots shifted right means high feature values push the model toward a particular class.
