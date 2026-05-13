# E20260429 — Round 3 (exp917, exp930–963) 모델 입출력 및 코드 description

본 라운드는 round-2의 `RangeDepthHead`(softmax distribution) + `HazardRangeDepthHead`(첫-히트 렌더링) 위에 (a) **R4 hazard rescue** 4 종 새 aux loss + smooth ramp, (b) **R5 RangeDepthHead 메인화**: 9 종 posterior representative + soft-quantile train loss + Spherical-SH aux + cylindrical bin-axis, (c) **per-metric 4-best ckpt** 저장, (d) **eval-time `--range-eval-mode`** 후처리 override를 도입한다. **Round-2 IO 문서(`logs/round1_bin-based/round2/E20260428-exp907-920-IO_and_code_description.md`)와의 차이점만 부각**하고 동일 부분은 짧게 참조한다.

코드 snapshot은 같은 폴더 위 `../code_snapshot/round3_frozen/`. 라운드-2의 코드 스냅샷은 `../code_snapshot/`(round1_frozen은 별도)에 보관됨.

---

## 1. 데이터 입력 — round-2와 동일

`SoundSpacesDataset.__getitem__` → 3-tuple `(audio, gt_depth, waveform)`. 텐서 shape 및 단위는 round-1/2 IO 문서 §1 그대로:

| 텐서 | 형태 | 단위 |
|---|---|---|
| `audio` | `(B, 2, 256, 512)` | binaural log-magnitude STFT |
| `gt_depth` | `(B, 1, 256, 512)` | ERP radial depth, normalised `[0,1] = real / max_depth (=10m)` |
| `waveform` | `(B, 2, 5648)` | binaural raw audio @ 16 kHz |

설정은 `config/echorange.yaml` (`logs/round1_bin-based/code_snapshot/round3_frozen/echorange.yaml`)에서:

- round-2와 동일: `lambda_silog: 0.5`, `lambda_berhu: 1.0`, `lambda_range_nll: 1.0`
- **R5 신규 default-off 플래그** (모두 0 또는 'radial'/'default' default):
  - `lambda_soft_quantile: 0.0`, `soft_quantile_q: 0.5`, `soft_quantile_tau: 0.05`
  - `lambda_spherical_sh: 0.0`, `spherical_sh_order: 2`, `spherical_sh_log_depth: true`
  - `range_bin_axis: 'radial'`, `cyl_min_axis_factor: 0.15`
  - `range_output_mode: 'expectation'`, `range_output_quantile: 0.5`, `range_output_temperature: 1.0`

R4 hazard 셀(exp930–934)은 round-2의 hazard hyperparams(`hazard_warmup_epochs`, `lambda_hit`, `lambda_free`, `hazard_bias_init`)에 더해:
- `hazard_aux_mode: 'raw_hit' | 'event_nll' | 'survival' | 'soft_hit'` (R4 신규)
- `hazard_far_thresh: 9.8` (rendered loss 마스킹)
- `hazard_event_sigma_bins: 1.0`, `hazard_survival_tau_bins: 1.0`, `hazard_soft_hit_target: 0.75`
- `hazard_log_delta: null`(default — 자동 bin-width 추출)

---

## 2. 모델 (`models/bin_based/echorange.py` — `EchoRangeDepth`)

### 2.1 head 토글 — round-2와 동일 (3 옵션)

```python
out = model(audio_spec, audio_wave)
```

`cfg.model.depth_head_type` ∈ `{'scalar', 'range', 'hazard'}` 동일. R5에서 RangeDepthHead가 메인 디자인이며 hazard는 closure 1셀(exp957)만 잔존.

### 2.2 forward 출력 — round-2와 동일 키 set

`{'pred_depth', 'range_logits', 'range_prob', 'range_entropy', 'range_bins'}` (range head)
`{'pred_depth', 'range_logits', 'range_bins', 'hazard_alpha', 'hazard_weights', 'hazard_bg_weight'}` (hazard head)

### 2.3 RangeDepthHead 변경 (R5)

`RangeDepthHead.__init__` 시그니처 확장:

```python
class RangeDepthHead(nn.Module):
    def __init__(self, in_channels, num_bins=32, r_min=0.1, r_max=10.0,
                 spacing="log", max_depth=10.0,
                 output_mode="expectation",   # round-2: {expectation, median}
                                              # R5 추가: {map, quantile,
                                              #          temperature_expectation}
                 output_quantile=0.5,         # R5 신규 (mode='quantile')
                 output_temperature=1.0,      # R5 신규 (mode='temperature_expectation')
                 axis="radial"):              # R5 신규: {radial, horizontal, z}
```

- `expectation`: `Σ p_j · r_j` (round-1/2 default)
- `median`: cumulative prob ≥ 0.5인 첫 bin 직접 선택 (round-2 exp907_TESTmedian)
- `map` (R5 신규): `argmax_j p_j` — 최빈 bin
- `quantile q ∈ {0.25, 0.35, ..., 0.75}` (R5 신규): cumulative prob ≥ q의 bin
- `temperature_expectation T ∈ {0.5, 0.75, 1.0, 1.5}` (R5 신규): softmax(logits/T)의 expectation

### 2.4 RangeDepthHead 신규 함수 (R5)

```python
def range_point_estimate(logits, range_bins, mode, q=0.5, temperature=1.0):
    """Eval-time decoder. 9 모드 모두 forward 수행 가능."""

def soft_quantile_depth(logits, range_bins, q=0.5, tau=0.05):
    """Differentiable soft quantile. Hard quantile은 gradient를 끊어
    train-time에 못 쓴다. softmax-weighted bin aggregation으로 grad 보존:
        F_j = cumsum(softmax(logits))
        weight_j = softmax((q - F_j) / tau).abs()  # bell-shape near q
        depth = Σ weight_j · range_bins[j] / Σ weight_j
    """
```

### 2.5 HazardRangeDepthHead — round-2와 동일

`models/bin_based/range_head.py:265-340` 그대로. R4의 4 aux mode는 head 자체가 아니라 *train_step의 loss 분기* 변경.

### 2.6 신규 모듈: `models/bin_based/spherical_loss.py`

```python
def make_erp_grid(H, W):
    """ERP pixel center → unit-sphere direction (x, y, z). cos(lat) area weight 동봉."""

def _real_sh_basis(L_max, dirs):
    """Real-valued SH basis Y_l^m(θ, φ) for l ∈ [0, L_max], m ∈ [-l, +l].
    Returns (L_max+1)² coefficients."""

def spherical_sh_coeffs(scalar_field, L=2, area_weight=cos_lat):
    """ERP scalar field → SH coefficient projection (L=2 → 9 coeffs, L=3 → 16, L=4 → 25)."""

def spherical_sh_loss(pred_depth_m, gt_depth_m, L=2, use_log_depth=True,
                      smooth_l1_beta=0.5):
    """Smooth-L1 between SH coefficients of pred vs gt depth fields on the
    ERP sphere. log_depth=True applies log1p before SH for far-pixel
    sensitivity. cos(lat) area weighting integrated."""
```

L=2 (9 coeffs)이 anchor; L=3, L=4는 R5B sweep에서 시도하나 게인 없음(§ Round 3 results analysis §4.3).

---

## 3. Train step — `_train_step_echorange` (`train.py:141..516`)

`code_snapshot/round3_frozen/_train_step_echorange.py` verbatim 발췌. round-2의 scalar/range/hazard 3-branch 구조를 그대로 두고 다음 4 가지 R5 신규 + R4 신규 분기를 끼움:

### 3.1 Hazard branch (R4 신규 4 aux mode)

```python
elif head_type == 'hazard':
    # ── Smooth aux-weight ramp (R4 신규, round-3 jump 대체) ─────────
    ramp_progress = min(1.0, float(epoch) / float(ramp_epochs))
    lam_aux  = ramp_progress * lambda_hit_target
    lam_free = ramp_progress * lambda_free_target

    aux_mode = cfg.model.hazard_aux_mode  # raw_hit | event_nll | survival | soft_hit

    if aux_mode == 'raw_hit':
        primary = hazard_supervision_loss(use_hit=True)['hit']
    elif aux_mode == 'event_nll':
        primary = rendered_event_nll(logits, gt_for_nll, range_bins,
                                     sigma_bins=1.0, far_thresh=9.8)
    elif aux_mode == 'survival':
        primary = survival_loss(logits, gt_for_nll, range_bins,
                                tau_bins=1.0, far_thresh=9.8)
    elif aux_mode == 'soft_hit':
        primary = soft_hit_bce_loss(logits, gt_for_nll, range_bins,
                                    soft_target=0.75, far_thresh=9.8)

    head_loss = lam_aux * primary + lam_free * hazard_free_loss(...)
```

R3의 jump warmup (`epoch ≤ ramp ? warm : target`)을 round-4가 smooth ramp(`progress = min(1, ep/ramp)`)로 교체. R4 §4.1.B의 frac>.99 dump가 직접 보여주는 saturation 회피 효과.

### 3.2 Range branch + soft-quantile aux loss (R5A 신규)

```python
if head_type == 'range':
    nll = soft_range_nll_loss(logits, gt_for_nll, range_bins,
                              valid_mask=far_valid, sigma=sigma, weights=pix_w)
    head_loss = lam_nll * nll

    # ── R5 신규: differentiable soft-quantile aux loss ──────────────
    if cfg.model.lambda_soft_quantile > 0.0:
        pred_q_axis = soft_quantile_depth(logits, range_bins,
                                          q=sq_q, tau=sq_tau)
        # bin-axis → radial projection if cylindrical
        # depth_norm scaling
        q_loss = BerHuLoss()(pred_q_norm, gt_q_norm) + SILogLoss()(...)
        head_loss = head_loss + lam_sq * q_loss
```

R5A sweep은 `(q ∈ {0.45, 0.50, 0.55}) × (τ ∈ {0.03, 0.05}) × (λ_sq ∈ {0.25, 0.50})`.

### 3.3 Cylindrical bin-axis (R5D, exp936–956 옵션)

```python
range_bin_axis = cfg.model.range_bin_axis  # radial | horizontal | z
cyl_min_factor = cfg.model.cyl_min_axis_factor  # default 0.15

if range_bin_axis == 'horizontal':
    f = torch.cos(lat).clamp(min=cyl_min_factor)   # ρ_xy = D · cos(lat)
elif range_bin_axis == 'z':
    f = torch.sin(lat).abs().clamp(min=cyl_min_factor)  # |z| = D · |sin(lat)|
# else radial: f=1 (no projection)

# GT는 NLL 계산 전 bin-axis로 투영
gt_for_nll = gt_for_nll * axis_factor_nll
# Polar 픽셀(|lat|>81°, factor 클램프 floor)은 마스크 처리
polar_mask_nll = (axis_factor_nll > cyl_min_factor + 1e-6)
far_valid = far_valid & polar_mask_nll

# pred_depth는 bin-axis로 출력되므로 BerHu/SILog 전 radial로 역투영
pred_radial = pred_depth / axis_factor_orig
```

R5D(exp954–956) 미실행 — n2 batch에서 슬롯 부족으로 보류.

### 3.4 Spherical-SH aux loss (R5B 신규)

```python
# train step 후미, range·hazard 공통
if cfg.model.lambda_spherical_sh > 0.0:
    sh_l = spherical_sh_loss(
        pred_depth_m=pred_radial,
        gt_depth_m=gt_radial,
        L=cfg.model.spherical_sh_order,
        use_log_depth=cfg.model.spherical_sh_log_depth,
    )
    loss = loss + lam_sh * sh_l
```

R5B sweep은 `L ∈ {2, 3} × λ_SH ∈ {0.02, 0.05, 0.10} × log_depth ∈ {true, false}`. exp946 (L=2, λ=0.10, logd=true)이 R5 best.

### 3.5 R5C combo

`lambda_soft_quantile > 0` AND `lambda_spherical_sh > 0` 동시 활성. exp950–951 2셀.

---

## 4. Validation pass — `_val_metrics` + `_hazard_diagnostics`

round-2와 동일. hazard 셀 val 첫 batch에서 `_hazard_diagnostics(logits, weights, alpha, gt)` dump:
- `alpha`: mean / p10 / p50 / p90 / p95 / p99
- `frac_alpha`: > {0.5, 0.9, 0.95, 0.99}
- `bg_weight`: p10 / p50 / p90
- `entropy_weights`: p10 / p50 / p90
- `argmax_bin_hist`: 32 → folded 16-bucket histogram

R4의 [haz] dump가 round-3 분석문서 §4.1.B에서 직접 인용된 출처.

---

## 5. Checkpoint — R5 신규 4-best 저장 (`train.py:1270..1488`)

매 val epoch마다 다음 4개를 분리 저장:

```python
score = 0.7 * vm['rmse'] + 0.3 * vm['abs_rel']

if score < best_score:
    best_score = score
    torch.save(payload, ckpt_dir / 'best_score.pth')
    torch.save(payload, ckpt_dir / 'best_model.pth')   # legacy 별명

if vm['rmse'] < best_rmse:
    torch.save(..., ckpt_dir / 'best_rmse.pth')

if vm['abs_rel'] < best_abs_rel:
    torch.save(..., ckpt_dir / 'best_absrel.pth')

if vm['delta1'] > best_delta1:
    torch.save(..., ckpt_dir / 'best_delta1.pth')
```

`best_model.pth`는 `best_score.pth`의 동일 사본 — 기존 test 스크립트 호환성 보장. R4 셀(exp917, 930–934)은 R5 코드 적용 전(round-2 종료 직후)의 단일 best_score만 보유.

---

## 6. Test — `test.py` + `utils/test_utils.py` (R5 신규 후처리)

### 6.1 `--checkpoint-tag {score, absrel, rmse, delta1}` (default `score`)

자동으로 `best_<tag>.pth` 선택. `score`는 legacy `best_model.pth` alias.

### 6.2 `--range-eval-mode` 후처리 override (R5 신규)

```python
_RANGE_EVAL_PRESETS = {
    'expectation':  ('expectation',             0.5,  1.0),
    'map':          ('map',                     0.5,  1.0),
    'q25/q35/.../q75': ('quantile',             q,    1.0),
    'temp05/temp075/temp15': ('temperature_expectation', 0.5, T),
}

def _override_range_pred_depth(raw_out, eval_mode):
    """Re-decode pred_depth from range_logits using chosen representative.
    range head only; hazard head ignores."""
    if eval_mode == 'default' or 'range_logits' not in raw_out:
        return raw_out['pred_depth']
    pred, _ = range_point_estimate(
        logits=raw_out['range_logits'],
        range_bins=raw_out['range_bins'],
        mode=mode, q=q, temperature=T)
    return pred
```

분포-head 학습된 ckpt는 그대로 두고 추론 시 9 종 representative로 재디코딩 가능. round-2 exp907_TESTmedian의 generalization. test_only swap이라 학습 비용 0.

### 6.3 Bin-axis → radial projection (R5D 신규)

```python
def _project_pred_to_radial(pred_depth, cfg, device):
    """cylindrical mode일 때 pred를 radial로 역투영. depth_norm scaling 전."""
```

### 6.4 평가 흐름

```python
elif echorange:
    raw_out = model(audio, waveform)
    head_type = cfg.model.depth_head_type
    eval_mode = cfg.model.range_eval_mode  # default | expectation | map | qXX | tempXX

    if head_type == 'range' and eval_mode != 'default':
        depth_pred = _override_range_pred_depth(raw_out, eval_mode)
    else:
        depth_pred = raw_out['pred_depth']

    if head_type in ('range', 'hazard'):
        depth_pred = _project_pred_to_radial(depth_pred, cfg, device)

    # depth_norm scaling, compute_errors(depth_pred, depthgt)
```

---

## 7. Bulk launchers

| Script | 셀 범위 | bs | epochs | 시간/셀 |
|---|---|---:|---:|---:|
| `scripts/n2_bulk_0428_r2.sh` (R4 hazard rescue) | exp917, 930–934 | 48 DP | 20 | ≈ 2.5h |
| `scripts/n2_bulk_0429_r3.sh` (R5 main) | exp936–957 | 48 DP | 20 | ≈ 2.5h |
| `scripts/n9_bulk_0429_r3.sh` (R5 paired baseline + low-bin) | exp958–967 | 32 single-GPU | 40 (R40) / 20 (R20) | ≈ 5h / 2.5h |

n2 환경(`HAZ_BASE`):
```
--depth-head-type hazard --range-num-bins 32 --range-bin-spacing log
--range-min-depth 0.1 --range-max-depth 10.0 --hazard-bias-init -4.6
--hazard-warmup-epochs 3 --hazard-far-thresh 9.8 --lambda-berhu 1.0
--lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask
```

n2 환경(`R_BASE`):
```
--depth-head-type range --range-num-bins 32 --range-bin-spacing log
--range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14
--range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0
--lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask
```

n9 환경(`R_BASE_BS32`): `R_BASE`와 동일 + bs=32, single-GPU.

---

## 8. 부록 A — Historical baselines (Round 0~2) 모든 수치

### A.1 Round 0 — 외부 echodiffusion 본가

| 실험 | lr | bs | ABS_REL | RMSE | δ1 |
|---|---|---:|---:|---:|---:|
| **exp11 (round-0 SOTA)** | 1e-4 | 32 | **0.4300** | **1.1060** | 0.4876 |
| exp363 | 5e-4 | 48 | 0.4482 | 1.2198 | 0.4936 |
| exp13 | 1e-4 | 16 | 0.4504 | 1.1134 | 0.4930 |
| exp362 | 1e-4 | 32 | 0.4557 | 1.2292 | 0.4923 |

### A.2 Round 1 — bin-based 도입 (`round1/`)

| 실험 | head | bs | ABS_REL | RMSE | δ1 |
|---|---|---:|---:|---:|---:|
| exp900 | scalar | 64 DP | 0.5229 | 1.2284 | 0.4854 |
| exp901–905 | range, bin/sigma sweep | 16 | 0.4888–0.5489 | 1.24–1.28 | 0.49–0.50 |
| exp904 (anchor) | range | 16 | 0.4888 | 1.2459 | 0.4988 |
| **exp906 (round-1 SOTA)** | range, ERP fix on | 16 | **0.4814** | **1.2532** | **0.5079** |

### A.3 Round 2 — bs=32 + ERP-ablation + 첫 hazard (`round2/`)

| 셀 | head | bs | ABS_REL | RMSE | δ1 |
|---|---|---:|---:|---:|---:|
| **exp907 (round-2 distribution best)** | range, full ERP fix | 32 | 0.4705 | 1.2269 | 0.5029 |
| **exp907_*_TESTmedian** | range(median, inf-only) | 32 | **0.4202** | 1.2765 | **0.5129** |
| exp908 | range, cos-lat only | 32 | 0.4946 | 1.2135 | 0.5064 |
| exp909 | range, far-mask only | 32 | 0.5113 | 1.2250 | 0.4969 |
| exp910 (6/20 ep) | range(median train+inf) | 32 | 0.4626 | 1.2266 | 0.5237 |
| exp911 | scalar (val only) | 32 | (val 0.4198) | (val 1.3863) | (val 0.5297) |
| **exp912 (same-env scalar key)** | scalar | 32 | **0.4349** | **1.2432** | **0.4831** |
| exp913 (hazard full) | hazard | 48 | 0.4130 | 1.5662 | 0.3016 |
| exp914 (hazard no-free) | hazard | 48 | 0.4266 | 1.5894 | 0.2531 |
| exp915 (hazard no-hit) | hazard | 48 | 0.4388 | 1.2473 | 0.4878 |
| exp916 (hazard depth-only) | hazard | 48 | 0.4777 | 1.2214 | 0.4963 |
| exp919 (λ_NLL=0.3 audit) | range | 48 | 0.4877 | 1.2225 | 0.5020 |
| exp920 (echodiff seed-2) | scalar | 48 | 0.4884 | 1.2212 | 0.4890 |

> Round-2 noise floor (exp912 ↔ exp920 비교 측정): test ABS_REL ±0.05, RMSE ±0.02, δ1 ±0.006.

---

## 9. 부록 B — Round 3 (n9_0427_test) 70 entries 풀덤프

> **포맷**: `exp_id | tag | best_epoch | ABS_REL RMSE δ1 δ2 δ3 Log10 MAE`. tag = score | absrel | delta1 | rmse (R5 4-best ckpt).

### B.1 R3·R4 hazard rescue (n2 bs=48 DP, 20 ep)

```
exp917 | score | ep20 | 0.4291 1.6095 0.2612 0.4965 0.6829 0.2462 1.0769
exp930 | score | ep02 | 0.4067 1.6270 0.3011 0.5329 0.7019 0.2305 1.0561
exp931 | score | ep12 | 0.4984 1.2169 0.4962 0.7122 0.8281 0.1584 0.7948
exp932 | score | ep08 | 0.5010 1.2233 0.4861 0.7047 0.8252 0.1593 0.7949
exp933 | score | ep08 | 0.4733 1.2202 0.4959 0.7063 0.8282 0.1563 0.7853
exp934 | score | ep12 | 0.4522 1.2293 0.4988 0.7158 0.8343 0.1548 0.7794
```

### B.2 R5A Soft-Quantile sweep (exp936–943, n2 bs=48 DP, 20 ep)

```
exp936 | score    | ep06 | 0.4726 1.2158 0.4957 0.7112 0.8322 0.1559 0.7839     # q=0.50 τ=0.05 λ=0.25 (anchor)
exp936 | delta1   | ep18 | 0.4903 1.2425 0.5021 0.7133 0.8266 0.1591 0.8013

exp937 | score    | ep06 | 0.5150 1.2175 0.4920 0.7039 0.8236 0.1599 0.7988     # q=0.45 τ=0.05 λ=0.25
exp937 | absrel   | ep20 | 0.5027 1.2559 0.5027 0.7117 0.8244 0.1604 0.8090
exp937 | delta1   | ep20 | 0.5027 1.2559 0.5027 0.7117 0.8244 0.1604 0.8090

exp938 | score    | ep12 | 0.4817 1.2304 0.5070 0.7151 0.8306 0.1566 0.7879     # q=0.55 τ=0.05 λ=0.25
exp938 | delta1   | ep18 | 0.5091 1.2502 0.5130 0.7172 0.8269 0.1587 0.8033

exp939 | score    | ep08 | 0.4585 1.2303 0.4982 0.7113 0.8331 0.1559 0.7853     # q=0.50 τ=0.03 λ=0.25
exp939 | rmse     | ep12 | 0.5183 1.2212 0.5023 0.7131 0.8282 0.1584 0.8005
exp939 | delta1   | ep18 | 0.5151 1.2640 0.5038 0.7048 0.8194 0.1615 0.8191

exp940 | score    | ep08 | 0.5230 1.2170 0.4992 0.7142 0.8288 0.1585 0.7966     # q=0.45 τ=0.03 λ=0.25
exp940 | rmse     | ep06 | 0.5504 1.2201 0.4867 0.7052 0.8208 0.1619 0.8100
exp940 | absrel   | ep10 | 0.4983 1.2213 0.5043 0.7181 0.8309 0.1567 0.7883
exp940 | delta1   | ep20 | 0.5426 1.2751 0.4968 0.7001 0.8161 0.1648 0.8324

exp941 | score    | ep10 | 0.5050 1.2039 0.4978 0.7126 0.8298 0.1572 0.7880     # q=0.50 τ=0.05 λ=0.50
exp941 | absrel   | ep20 | 0.5016 1.2658 0.5133 0.7127 0.8225 0.1606 0.8108
exp941 | delta1   | ep20 | 0.5016 1.2658 0.5133 0.7127 0.8225 0.1606 0.8108

exp942 | score    | ep10 | 0.5239 1.2030 0.4923 0.7108 0.8266 0.1589 0.7919     # q=0.45 τ=0.05 λ=0.50
exp942 | absrel   | ep20 | 0.4622 1.2473 0.5058 0.7140 0.8287 0.1582 0.7958
exp942 | delta1   | ep14 | 0.4953 1.2157 0.5079 0.7170 0.8316 0.1558 0.7850

exp943 | score    | ep06 | 0.4879 1.2149 0.4852 0.7066 0.8292 0.1584 0.7941     # q=0.50 τ=0.03 λ=0.50
exp943 | absrel   | ep08 | 0.4791 1.2094 0.5069 0.7167 0.8340 0.1547 0.7779
exp943 | delta1   | ep14 | 0.5186 1.2477 0.5072 0.7143 0.8268 0.1600 0.8094
```

### B.3 R5B Spherical-SH sweep (exp944–949, n2 bs=48 DP, 20 ep)

```
exp944 | score    | ep08 | 0.4928 1.2048 0.5014 0.7197 0.8339 0.1549 0.7770     # L=2 λ=0.02 logd=true
exp944 | absrel   | ep16 | 0.4917 1.2379 0.5129 0.7143 0.8250 0.1583 0.7973
exp944 | delta1   | ep18 | 0.4985 1.2547 0.5137 0.7157 0.8249 0.1590 0.8033

exp945 | score    | ep06 | 0.5222 1.2178 0.4995 0.7077 0.8260 0.1587 0.7964     # L=2 λ=0.05 logd=true
exp945 | absrel   | ep10 | 0.4805 1.2232 0.4999 0.7134 0.8290 0.1567 0.7840
exp945 | delta1   | ep14 | 0.4839 1.2458 0.5061 0.7145 0.8273 0.1582 0.7998

exp946 | score    | ep08 | 0.4413 1.2208 0.5019 0.7150 0.8320 0.1542 0.7755     # ★ L=2 λ=0.10 logd=true (R5 best, 28셀 중 유일 7-metric all-win vs exp958)
exp946 | rmse     | ep10 | 0.5405 1.2272 0.5038 0.7115 0.8240 0.1598 0.8045
exp946 | delta1   | ep20 | 0.5320 1.2683 0.5048 0.7075 0.8199 0.1621 0.8225

exp947 | score    | ep08 | 0.5082 1.2327 0.4953 0.7055 0.8248 0.1596 0.8048     # L=3 λ=0.02 logd=true
exp947 | absrel   | ep12 | 0.4647 1.2573 0.5061 0.7141 0.8281 0.1581 0.7999
exp947 | delta1   | ep16 | 0.5233 1.2706 0.5007 0.7077 0.8221 0.1619 0.8200

exp948 | score    | ep10 | 0.4834 1.2053 0.5008 0.7167 0.8322 0.1552 0.7787     # L=3 λ=0.05 logd=true
exp948 | absrel   | ep08 | 0.4401 1.2327 0.4940 0.7077 0.8307 0.1561 0.7825
exp948 | delta1   | ep14 | 0.5532 1.2290 0.5057 0.7062 0.8200 0.1609 0.8068

exp949 | score    | ep10 | 0.4622 1.2200 0.5079 0.7190 0.8334 0.1545 0.7793     # L=2 λ=0.02 logd=false
exp949 | rmse     | ep14 | 0.5187 1.2225 0.5055 0.7158 0.8280 0.1579 0.7917
exp949 | delta1   | ep14 | 0.5187 1.2225 0.5055 0.7158 0.8280 0.1579 0.7917
```

### B.4 R5C Combo (sq + SH) (exp950–951, n2 bs=48 DP, 20 ep)

```
exp950 | score    | ep08 | 0.5158 1.2141 0.4978 0.7077 0.8265 0.1586 0.7925     # sq(q0.5/τ0.05/λ0.25) + SH(L2/λ0.02)
exp950 | delta1   | ep16 | 0.5207 1.2493 0.5089 0.7118 0.8242 0.1595 0.8091

exp951 | score    | ep10 | 0.5003 1.2158 0.4971 0.7117 0.8281 0.1576 0.7925     # sq(q0.5/τ0.05/λ0.25) + SH(L2/λ0.05)
exp951 | rmse     | ep12 | 0.5039 1.2268 0.5067 0.7138 0.8281 0.1578 0.7949
exp951 | delta1   | ep16 | 0.4914 1.2595 0.5015 0.7121 0.8245 0.1602 0.8092
```

### B.5 R5F 40-epoch baseline (exp958–961, n9 bs=32 single-GPU, 40 ep)

```
exp958 | score    | ep16 | 0.4463 1.2611 0.4983 0.7067 0.8272 0.1593 0.7965     # echodiffusion 본가 scalar 40ep
exp958 | rmse     | ep12 | 0.5564 1.2685 0.4688 0.6928 0.8166 0.1711 0.8610
exp958 | delta1   | ep24 | 0.4989 1.2780 0.4964 0.7015 0.8184 0.1627 0.8187

exp959 | score    | ep12 | 0.4873 1.2557 0.4860 0.7014 0.8257 0.1621 0.8126     # echorange-scalar 40ep
exp959 | absrel   | ep26 | 0.4707 1.2797 0.4909 0.7001 0.8181 0.1630 0.8139
exp959 | delta1   | ep30 | 0.4807 1.2813 0.4959 0.7040 0.8198 0.1625 0.8147

exp960 | score    | ep06 | 0.4951 1.2160 0.4962 0.7126 0.8303 0.1570 0.7903     # range expectation 40ep
exp960 | rmse     | ep08 | 0.5166 1.2277 0.5037 0.7098 0.8263 0.1594 0.7984
exp960 | delta1   | ep30 | 0.5198 1.2825 0.5086 0.7120 0.8223 0.1615 0.8221

exp961 | score    | ep10 | 0.4520 1.2444 0.5199 0.7231 0.8319 0.1562 0.7850     # ★ range median 40ep (28셀 중 δ1 챔피언)
exp961 | rmse     | ep08 | 0.5301 1.2259 0.5192 0.7213 0.8307 0.1573 0.7888
exp961 | absrel   | ep12 | 0.4593 1.2607 0.5163 0.7171 0.8263 0.1601 0.7967
exp961 | delta1   | ep10 | 0.4520 1.2444 0.5199 0.7231 0.8319 0.1562 0.7850
```

### B.6 R5G·R5H Br=20 저-비닝 (exp962–963, n9 bs=32, 20 ep)

```
exp962 | score    | ep10 | 0.4686 1.2218 0.5034 0.7151 0.8321 0.1556 0.7838     # R20 sq anchor
exp962 | absrel   | ep06 | 0.4921 1.2342 0.4865 0.7059 0.8266 0.1591 0.7940
exp962 | rmse     | ep12 | 0.5086 1.2172 0.5062 0.7133 0.8276 0.1574 0.7914
exp962 | delta1   | ep14 | 0.5000 1.2265 0.5051 0.7150 0.8285 0.1578 0.7919

exp963 | score    | ep10 | 0.5221 1.2267 0.5010 0.7119 0.8261 0.1595 0.7996     # R20 SH (L=2/λ=0.02)
exp963 | rmse     | ep08 | 0.5855 1.2262 0.4791 0.6940 0.8138 0.1656 0.8268
exp963 | absrel   | ep14 | 0.5110 1.2324 0.5089 0.7118 0.8245 0.1589 0.7999
exp963 | delta1   | ep20 | 0.5152 1.2570 0.5089 0.7133 0.8238 0.1601 0.8113
```

---

## 10. 부록 C — Metric별 28셀 챔피언 (best_score 기준)

| Metric | Best value | Best 셀 | Round-0 SOTA | 격차 |
|---|---:|---|---:|---|
| ABS_REL ↓ (정상 셀만) | **0.4413** | exp946 (R5B SH high-λ) | 0.4300 | +0.011 손해 |
| RMSE ↓ | **1.2030** | exp942 (R5A sq λ=0.5) | 1.1060 | +0.097 손해 |
| δ1 ↑ | **0.5199** | exp961 (R40 median) | 0.4876 | +0.032 우위 |
| δ2 ↑ | **0.7231** | exp961 | — | (R0 미기재) |
| δ3 ↑ | **0.8343** | exp934 (R4 soft_hit) | — | — |
| Log10 ↓ | **0.1542** | exp946 | — | — |
| MAE ↓ | **0.7755** | exp946 | — | — |

**핵심**: ABS_REL과 RMSE의 round-0 SOTA(0.4300, 1.1060)는 어떤 셀도 깨지 못함. δ1·δ2·δ3·Log10·MAE는 깸. ABS_REL과 RMSE를 동시에 baseline 대비 의미있게(둘 다 0.01 이상) 좋아진 셀: **0건** (어떤 baseline 비교에서도). 자세한 분석은 같은 폴더 `E20260429-exp917-963-Round3_results_analysis.md` §5 참조.

---

## 11. 산출 ckpt

```
checkpoints/echorange_soundspaces_BS48_Lr0.0001_AdamW_exp{917,930-934,936-951}_*_bs48_r{2,3}/
  ├── best_model.pth       (= best_score.pth alias, R5 셀만; R4 셀은 단일 best_model.pth)
  ├── best_score.pth       (R5 셀만, 4-best 도입 후)
  ├── best_rmse.pth        (R5 셀만)
  ├── best_absrel.pth      (R5 셀만)
  └── best_delta1.pth      (R5 셀만)

checkpoints/echodiffusion_soundspaces_BS32_Lr0.0001_AdamW_exp958_R5F_S40_echodiff_bs32_r3_ep40/
checkpoints/echorange_soundspaces_BS32_Lr0.0001_AdamW_exp{959-963}_*_bs32_r3*/
```
