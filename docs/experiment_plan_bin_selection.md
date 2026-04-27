# Acoustic Temporal Bin Analysis (Research Plan)

## 1. Goal

We aim to identify **which temporal components (bins) and coefficients in acoustic signals contribute to depth estimation**, under a realistic setting where **binaural cues are always available**.

The core questions are:

* Which temporal windows contain depth-relevant information?
* Are these cues localized (specific bins) or distributed (combinational)?
* Do different bins contribute differently across **depth ranges** (near/mid/far)?
* Do they affect different **spatial regions** (ERP directions)?

---

## 2. Experimental Principle

We treat **binaural input as the primary cue**, and analyze:

> What additional information ambisonic temporal bins provide **on top of binaural signals**

This avoids unrealistic settings and ensures practical relevance.

---

## 3. Model Setup

### Input

* Binaural signal (always used)
* Ambisonic representation divided into **K temporal bins (e.g., K=8)**

### Architecture

```text
Binaural Encoder → f_b

Ambisonic bins → Gated → Ambisonic Encoder → f_a

Fusion: concat(f_b, f_a)
→ Depth head (classification or regression)
```

### Gating Mechanism

For temporal bins:

```python
g = sigmoid(gate)  # shape: [K]
x = x * g.view(1, K, 1, 1, 1)
```

Optional (stronger analysis):

```python
g = sigmoid(gate)  # shape: [K, C] (per coefficient)
```

### Loss

```python
L = L_depth + λ * mean(g)
```

* `L_depth`: depth-bin classification (preferred) or regression
* `λ`: sparsity weight (encourages selective bin usage)

---

## 4. Core Experiments (Single Training Run)

Train **one model only** with gating.

Then evaluate multiple configurations:

### Baselines

```text
1. Binaural only
2. Binaural + all bins
```

### Gated variants

```text
3. Binaural + learned-selected bins
4. Binaural + top-k bins (by gate)
5. Binaural + random k bins
6. Binaural + all except selected bins
```

---

## 5. Temporal Bin Analysis

### (A) Gate Inspection

Analyze:

```text
g[0], g[1], ..., g[K-1]
```

Interpretation:

* High value → frequently used bin
* Low value → likely redundant

⚠️ Gate alone is NOT sufficient evidence

---

### (B) Drop-one-bin Ablation (Critical)

```text
All bins
All - bin0
All - bin1
...
All - binK
```

Measure performance drop.

This reveals **actual importance**, not just learned preference.

---

### (C) Single-bin Test

```text
bin0 only
bin1 only
...
```

Used to distinguish:

```text
independent vs combinational contribution
```

---

## 6. Depth-Range Analysis

Divide depth into bins:

```text
Near: 0–2m
Mid: 2–5m
Far: 5m+
```

Evaluate:

```text
AbsRel
RMSE
Depth-bin accuracy
```

Goal:

* Identify which temporal bins contribute to which depth ranges

Expected pattern:

```text
Early bins → near depth
Late bins → weak or diffuse contribution
```

---

## 7. Spatial (ERP) Region Analysis

Divide ERP into regions:

```text
Front / Back / Left / Right
(Optional: Up / Down)
```

Evaluate metrics per region.

---

### Spatial Drop-Bin Map

For bin (k):

[
\Delta E_k(u) = E_{\text{w/o bin }k}(u) - E_{\text{all}}(u)
]

Visualize:

```text
importance map per bin
```

This shows **where each temporal bin matters spatially**.

---

## 8. Coefficient-Level Extension (Optional)

Extend gate to:

```python
gate ∈ ℝ^{K × C}
```

Analyze:

```text
which bin + which coefficient (W, X, Y, Z) contributes
```

Expected insight:

```text
Directional channels (X,Y,Z) > W for geometry
```

---

## 9. Additional Validation (Important)

### (A) Random Control

Compare:

```text
Learned bin selection vs random selection
```

Must show:

```text
learned > random
```

---

### (B) Energy Normalization

Normalize bin magnitude:

```python
x_bin = x_bin / ||x_bin||
```

Purpose:

* Remove loudness bias
* Test if structure (not energy) is responsible

---

## 10. Interpretation Framework

A bin is considered meaningful if:

```text
1. Gate value is high
2. Removing it degrades performance
3. It outperforms random selection
4. Its effect is consistent across scenes
```

---

## 11. Expected Outcome

Strong results should show:

```text
- Early temporal bins dominate near-field depth prediction
- Contribution is spatially localized (e.g., around reflecting surfaces)
- Some bins are weak individually but critical in combination
- Directional coefficients and cross-channel structure matter
```

---

## 12. Key Takeaway

This framework allows us to answer:

> Where (time), how (channel interaction), and where in space (ERP) acoustic information contributes to depth estimation.

Most importantly:

> It isolates **complementary information beyond binaural cues**, which is critical for validating ambisonic representations.
