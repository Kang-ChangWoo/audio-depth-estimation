# Ambisonic Coefficient Predictability via Temporal Compression

## 1. Goal

This experiment evaluates whether **Ambisonic representations can be predicted from binaural input**, independent of depth prediction.

Unlike previous experiments that measure *utility for depth*, this experiment focuses on:

> How well different compressed Ambisonic representations can be inferred from binaural observations.

---

## 2. Core Idea

We compress FOA signals ((4 \times 3000)) into lower-dimensional representations using:

* **RMS-based summary**
* **Eigen-based summary (dominant mode)**

and vary temporal resolution:

```text
N = 1, 8, 100 bins
```

This allows us to analyze:

```text
Predictability vs Temporal granularity vs Representation type
```

---

## 3. Representation Design

Given FOA IR:

[
A \in \mathbb{R}^{4 \times T_w}
]

for each temporal window, we compute:

---

### (A) RMS Summary

[
r = \sqrt{\frac{1}{T_w} \sum_t A(t)^2}
]

Shape:

```text
[4] → (W, X, Y, Z)
```

Interpretation:

* Channel-wise independent energy
* No cross-channel interaction

---

### (B) Eigen Summary (Top-1 Mode)

[
R = \frac{1}{T_w} A A^\top
]

[
R \approx \lambda_1 q_1 q_1^\top
]

[
z = \sqrt{\lambda_1} q_1
]

Shape:

```text
[4]
```

Interpretation:

* Dominant acoustic mode
* Captures cross-channel structure
* Encodes directional interaction

---

## 4. Temporal Binning

We divide the full signal into (N) bins:

```text
N = 1   → global summary
N = 8   → interpretable temporal structure
N = 100 → fine-grained temporal resolution
```

Final target shapes:

```text
RMS:
[B, N, 4]

Eigen:
[B, N, 4]
```

---

## 5. Model Setup

```text
Binaural input
→ Encoder (same backbone as depth model)
→ Prediction head
→ FOA compressed target
```

Important:

* Same architecture as depth model
* Same dataset and split
* Only target is changed (depth → FOA representation)

---

## 6. Loss Functions

### Coefficient Loss

```math
L_{coef} = \| \hat{A} - A \|_1
```

### Direction Loss (for eigen)

```math
L_{dir} = 1 - \cos(\hat{z}, z)
```

### Final Loss

```math
L = L_{coef} + \lambda L_{dir}
```

Recommended:

```text
λ = 0.1
```

---

## 7. Evaluation Metrics

Evaluate per-bin and per-channel:

### Basic metrics

```text
L1 / L2 error
Cosine similarity
Correlation
```

### Structure-aware metrics

```text
Energy map reconstruction error
Directional similarity
```

---

## 8. Experimental Matrix

```text
RMS-1
RMS-8
RMS-100

Eigen-1
Eigen-8
Eigen-100
```

---

## 9. Key Questions

This experiment answers:

```text
1. Is FOA structure predictable from binaural input?
2. Does predictability depend on temporal resolution?
3. Is cross-channel structure (Eigen) harder to predict than channel-wise energy (RMS)?
4. How much temporal detail can be recovered?
```

---

## 10. Expected Behavior

```text
RMS > Eigen in predictability (easier task)

1-bin > 8-bin > 100-bin (in prediction accuracy)

Eigen-8 may provide best trade-off between structure and predictability
```

---

## 11. Interpretation

We interpret results as:

```text
High predictability:
→ binaural contains sufficient cues for that representation

Low predictability:
→ information is either absent or too ambiguous to recover
```

Important:

```text
Predictability ≠ usefulness for depth
```

This experiment must be combined with utility analysis.

---

## 12. Key Takeaway

This experiment evaluates:

> How much of Ambisonic structure is recoverable from binaural observations under different levels of temporal compression and representation.

It provides the foundation for deciding:

```text
Which Ambisonic representation can be realistically used as an intermediate prediction target.
```
