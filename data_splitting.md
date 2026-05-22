# Preventing Data Leakage in Sliding-Window Time Series Classification

## 1. Problem Setting

Let $\mathbf{s} = (s_1, s_2, \ldots, s_N)$ be a univariate or multivariate time series of length $N$, where each observation $s_t \in \mathbb{R}^F$ carries $F$ features and an associated class label $c_t \in \{1, \ldots, C\}$.

A standard preprocessing step for sequence classification is the construction of overlapping **sliding windows**. Given a lookback window of length $W$, sample $i$ is defined as:

$$
\mathbf{x}_i = (s_{i-W},\, s_{i-W+1},\, \ldots,\, s_{i-1}) \in \mathbb{R}^{W \times F}, \quad y_i = c_i, \quad i \in \{W+1, \ldots, N\}
$$

This yields a dataset of $M = N - W$ labelled windows. Consecutive windows are **highly overlapping**: windows $\mathbf{x}_i$ and $\mathbf{x}_{i+1}$ share $W - 1$ out of $W$ raw time steps.

---

## 2. Data Leakage Under Random Splitting

A common but incorrect approach is to first construct all $M$ windows and then apply a random permutation to partition them into training, validation, and test subsets:

$$
\{(\mathbf{x}_i, y_i)\}_{i=1}^{M} \xrightarrow{\text{random shuffle}} \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}
$$

This introduces two forms of data leakage.

**Overlap leakage.** Because adjacent windows share $W - 1$ time steps, a window $\mathbf{x}_i$ assigned to $\mathcal{D}_{\text{train}}$ and the window $\mathbf{x}_{i+1}$ assigned to $\mathcal{D}_{\text{val}}$ share the raw observations $s_{i-W+1}, \ldots, s_{i-1}$. The model is therefore evaluated on inputs that are nearly identical to those it was trained on, producing an optimistic and unreliable estimate of generalisation performance.

**Temporal leakage.** Under a random assignment, future observations can appear in $\mathcal{D}_{\text{train}}$ while earlier observations appear in $\mathcal{D}_{\text{val}}$ or $\mathcal{D}_{\text{test}}$. This violates the causal structure of the data, since at inference time the model will only ever observe past context.

---

## 3. Temporal Splitting

The correct strategy is to split the **raw time series** chronologically before constructing any windows. Let $r_{\text{val}}, r_{\text{test}} \in (0, 1)$ denote the desired fractions for validation and test, and define the cut points:

$$
n_{\text{test}} = \lfloor N \cdot r_{\text{test}} \rfloor, \quad
n_{\text{val}}  = \lfloor N \cdot r_{\text{val}}  \rfloor, \quad
n_{\text{train}} = N - n_{\text{val}} - n_{\text{test}}
$$

The raw series is partitioned into three chronological segments:

$$
\underbrace{s_1, \ldots, s_{n_{\text{train}}}}_{\text{train segment}}
\;\Big|\;
\underbrace{s_{n_{\text{train}}+1}, \ldots, s_{n_{\text{train}}+n_{\text{val}}}}_{\text{val segment}}
\;\Big|\;
\underbrace{s_{n_{\text{train}}+n_{\text{val}}+1}, \ldots, s_N}_{\text{test segment}}
$$

Windows and labels are then constructed **independently within each segment**. For the training set, window endpoints are restricted to $[W+1,\, n_{\text{train}}]$:

$$
\mathcal{D}_{\text{train}} = \{(\mathbf{x}_i, y_i) : i \in \{W+1, \ldots, n_{\text{train}}\}\}
$$

For validation and test, the first window requires $W$ steps of historical context. Rather than discarding those boundary samples, the tail of the preceding segment is used as **read-only context**: it supplies the lookback but contributes no labelled endpoints to the previous partition.

$$
\mathcal{D}_{\text{val}} = \{(\mathbf{x}_i, y_i) : i \in \{n_{\text{train}}+1, \ldots, n_{\text{train}}+n_{\text{val}}\}\}
$$

$$
\mathcal{D}_{\text{test}} = \{(\mathbf{x}_i, y_i) : i \in \{n_{\text{train}}+n_{\text{val}}+1, \ldots, N\}\}
$$

where each $\mathbf{x}_i$ is constructed from the global raw series, so the lookback of the first validation window naturally spans $s_{n_{\text{train}}-W+1}, \ldots, s_{n_{\text{train}}}$.

This construction guarantees:

1. **No endpoint overlap.** The index sets of labelled endpoints are disjoint by construction, so no raw observation is used as a label in more than one split.
2. **Causal ordering.** Every training endpoint precedes every validation endpoint, which in turn precedes every test endpoint. The model is never exposed to future information during training.
3. **Minimal boundary waste.** Only the lookback context (at most $W$ raw steps at each boundary) is shared between adjacent segments; no labelled samples are discarded.

---

## 4. Practical Considerations

**Class distribution.** Unlike random splitting, temporal splitting does not support stratification. The class distribution may therefore differ across splits, particularly when the recorded process undergoes concept drift over time. Reporting per-class metrics (precision, recall, F1) for each split is advisable to detect such drift.

**Lookback window size.** The number of labelled samples lost at each segment boundary is zero under this scheme, but the effective size of each split is reduced by $W$ raw steps of context. For typical values of $W \ll n_{\text{val}},\, n_{\text{test}}$, this reduction is negligible.

**Hyperparameter search.** When conducting random hyperparameter search, $W$ itself is a tunable parameter. Each candidate value of $W$ induces a different set of split boundaries and a different number of windows per split. All metrics reported during search are computed on $\mathcal{D}_{\text{val}}$, whose boundaries shift with $W$; this is expected and does not constitute leakage provided the test set is held out until final evaluation.
