# E20260428 — Round 2 (exp907–920) 모델 입출력 및 코드 description

본 라운드는 round-1의 분포(`range`) head 위에 (a) hazard 첫-히트 렌더링 head를 추가하고, (b) `_train_step_echorange`에 hazard branch / smooth ramp / aux-mode dispatch / ablation flag를 끼워넣고, (c) val pass에 `_hazard_diagnostics` 1-shot dump를 추가했다. **Round-1 IO 문서(`logs/round1_bin-based/round1/E20260428-exp900-906-IO_and_code_description.md`)와의 차이점만 부각**하고 동일 부분은 짧게 참조한다.

코드 snapshot은 같은 폴더 `code_snapshot/`. 라운드-1 snapshot은 `logs/round1_bin-based/round1/code_snapshot/`에 별도 보관(round-1 시점 frozen).

---

## 1. 데이터 입력 — round-1과 동일

`SoundSpacesDataset.__getitem__` → 3-tuple `(audio, gt_depth, waveform)`:

| 텐서 | 형태 | 단위 |
|------|------|------|
| `audio` | `(B, 2, 256, 512)` | binaural log-magnitude STFT |
| `gt_depth` | `(B, 1, 256, 512)` | ERP radial depth, normalised `[0,1] = real/10m`, GT > 10 m clamped to 1.0 |
| `waveform` | `(B, 2, 5648)` | binaural raw audio @ 16 kHz |

설정은 `config/echorange.yaml`에서 round-1과 동일. round-2의 변경:
- `lambda_silog: 0.5` (round-1 audit fix; train.py가 `cfg.model.lambda_silog`를 읽음)
- `depth_head_type: 'scalar'`(default) | `'range'` | **`'hazard'`** (신규)
- hazard hyperparams (`hazard_bias_init`, `hazard_warmup_epochs`, `lambda_hit{,_warmup}`, `lambda_free{,_warmup}`)

---

## 2. 모델 (`models/bin_based/echorange.py` — `EchoRangeDepth`)

### 2.1 head 토글

라운드-1과 같은 forward 시그니처 + `'hazard'` 옵션이 추가됨:

```python
out = model(audio_spec, audio_wave)   # 시그니처 동일
```

`cfg.model.depth_head_type`에 따라 `self.depth_head` 인스턴스가 분기:

```python
if depth_head_type == 'scalar':
    head = nn.Sequential(Conv2d(C, 1, 1), nn.Sigmoid())   # × max_depth
elif depth_head_type == 'range':
    head = RangeDepthHead(C, num_bins, r_min, r_max, spacing,
                          output_mode='expectation' | 'median')
elif depth_head_type == 'hazard':
    head = HazardRangeDepthHead(C, num_bins, r_min, r_max,
                                spacing, max_depth, bias_init=-4.6)
```

### 2.2 forward 출력 (dict, 세 head 공통 + 추가 키)

```python
{
    # 공통 (모든 head)
    'pred_depth':       (B, 1, H, W),       # metres

    # range head only (round-1과 동일)
    'range_logits':     (B, Br, h, w),
    'range_prob':       (B, Br, h, w),
    'range_entropy':    (B, 1, H, W),
    'range_bins':       (Br,),

    # hazard head only (round-2 신규)
    'range_logits':     (B, Br, h, w),       # raw logits (sigmoid 전)
    'range_bins':       (Br,),
    'hazard_alpha':     (B, Br, h, w),       # σ(logits), [eps, 1-eps] 클램프
    'hazard_weights':   (B, Br, h, w),       # w_j = T_{j-1}·α_j
    'hazard_bg_weight': (B, 1, h, w),        # w_bg = ∏(1-α_j)
}
```

### 2.3 HazardRangeDepthHead 핵심 (`models/bin_based/range_head.py:265-340`)

```python
class HazardRangeDepthHead(nn.Module):
    def __init__(self, in_channels, num_bins=32, r_min=0.1, r_max=10.0,
                 spacing="log", max_depth=10.0, bias_init=-4.6, eps=1e-8):
        super().__init__()
        self.register_buffer("range_bins",
            _make_range_bins(num_bins, r_min, r_max, spacing))
        self.logit_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_bins, 3, 1, 1),
        )
        nn.init.constant_(self.logit_conv[-1].bias, bias_init)
        self.max_depth, self.eps = float(max_depth), float(eps)

    def forward(self, feat):
        logits = self.logit_conv(feat)                                # (B, R, H, W)
        alpha  = torch.sigmoid(logits).clamp(min=self.eps, max=1.0-self.eps)
        T_inc  = torch.cumprod(1.0 - alpha, dim=1)                    # ∏_{k≤j}
        T_pre  = torch.cat([torch.ones_like(T_inc[:, :1]),
                            T_inc[:, :-1]], dim=1)                    # ∏_{k<j}
        weights   = T_pre * alpha                                     # w_j
        bg_weight = T_inc[:, -1:]                                     # w_bg

        bins = self.range_bins.view(1, -1, 1, 1)
        pred_depth = ((weights * bins).sum(dim=1, keepdim=True)
                      + bg_weight * self.max_depth)                   # metres
        return {
            "pred_depth":       pred_depth,
            "range_logits":     logits,
            "range_bins":       self.range_bins,
            "hazard_alpha":     alpha,
            "hazard_weights":   weights,
            "hazard_bg_weight": bg_weight,
        }
```

수치 안정성:
- α ∈ [eps, 1−eps] 클램프 — cumprod에서 0/1 underflow/overflow 방지.
- bias_init=−4.6 → α_init ≈ 0.01: 학습 시작 시 “거의 free space”.
- pred_depth ∈ [r_min · w_1, max_depth]에 항상 머무름 (Σw + w_bg = 1).

---

## 3. Loss (`train.py` → `_train_step_echorange`, code_snapshot 참조)

### 3.1 head 분기 (round-2 모습)

```
head_type:
  scalar  →  loss = BerHu(pred_norm, gt_norm) + 0.5·SILog(pred_norm, gt_norm)
                                                  ↑ silog 가중치 (echorange.yaml의 lambda_silog)

  range   →  head_loss = λ_NLL · soft_range_nll_loss(logits, gt_m, range_bins,
                                                     valid_mask=far_valid,
                                                     sigma=σ, weights=cos_lat)
             total     = head_loss + λ_BerHu·BerHu + λ_SILog·SILog [+ λ_ent·entropy]

  hazard  →  ramp_progress = min(1, epoch / hazard_warmup_epochs)   # smooth ramp (round-4)
             λ_aux  = ramp_progress · λ_hit
             λ_free = ramp_progress · λ_free
             primary = (raw_hit | event_nll | survival | soft_hit) loss   # aux_mode dispatch
             head_loss = λ_aux · primary + λ_free · hazard_free_loss(...)
             total     = head_loss + λ_BerHu·BerHu + λ_SILog·SILog
```

> **Round-2 ↔ round-1의 핵심 train-step 차이**:
>
> 1. range head 분기는 보존 (round-1 동일).
> 2. **hazard 분기 신규 추가** — 로직은 위 §3.1.
> 3. `lambda_berhu`/`lambda_silog`가 cfg에서 직접 읽힘 (round-1은 yaml의 `train.w_berhu`/`w_silog`만 보고 model.* 값과 mismatch가 있었음 — round-2 audit fix).
> 4. ERP 보정(`erp_far_mask`, `erp_cos_lat_weight`)이 range *와* hazard 양쪽에 공통 적용.
> 5. round-2 학습 도중 hazard discontinuous warmup 점프(λ_hit 0.3 → 0.5)가 saturation을 일으킨 게 진단됐고, **round-4 (`n2_bulk_0428_r2.sh`)에서 smooth ramp로 교체**된 코드가 이 snapshot의 base. round-2의 cell들은 *원래 jump 코드*로 학습됐음을 유의.

### 3.2 hazard 보조 손실 4종 (`models/bin_based/range_head.py`)

라운드 2의 cell들은 raw_hit만 사용했지만, snapshot에는 round-4 도입의 4종 모두 들어있음. 각각:

```python
# (1) hazard_supervision_loss — round-3 raw α-BCE.
#   L_hit  = mean over (pixel, hit_bin) of  BCEwithLogits(logits, target=1)
#   L_free = mean over (pixel, free_bin) of BCEwithLogits(logits, target=0)
# 이 형태가 라운드-2 cell exp913–917에 사용된 것.
def hazard_supervision_loss(logits, target_depth, range_bins,
                            log_delta=None, valid_mask=None,
                            weights=None, use_hit=True, use_free=True): ...

# (2) rendered_event_nll — round-4 main candidate.
#   per-pixel q_j = exp(-((log_bin_j - log D)/σ)^2/2), 정규화 → soft Gaussian target
#   loss = -Σ_j q_j · log(w_j + eps)  (rendered first-hit weight w_j 직접 supervise)
def rendered_event_nll(logits, target_depth, range_bins,
                       valid_mask=None, weights=None,
                       sigma_bins=1.0, far_thresh=9.8): ...

# (3) survival_loss — ordinal-style.
#   target S_j = sigmoid((log D - log r_j) / τ_bins · log(r_max/r_min)/Br)
#   loss = BCE(log(T_j), S_j) where T_j = ∏_{k<j}(1-α_k) (cumulative survival)
def survival_loss(logits, target_depth, range_bins,
                  valid_mask=None, weights=None,
                  tau_bins=1.0, far_thresh=9.8): ...

# (4) soft_hit_bce_loss — saturation-guarded raw α-BCE.
#   L = BCE(α_hit, target=soft_target=0.75) on hit bins
def soft_hit_bce_loss(logits, target_depth, range_bins,
                      log_delta=None, soft_target=0.75,
                      valid_mask=None, weights=None,
                      far_thresh=9.8): ...

# Free loss (모든 모드 공유, survival에서는 자동 skip)
def hazard_free_loss(logits, target_depth, range_bins,
                     log_delta=None, valid_mask=None,
                     weights=None, far_thresh=9.8):
    """BCE(α_free, target=0) on bins closer than D − half_log_bin."""
```

### 3.3 ERP 보정 (round-1과 동일, `_train_step_echorange` 안)

```python
# erp_far_mask: GT >= range_max_depth 픽셀 제외 (last-bin saturation 방지)
far_valid = (gt_for_nll > 0) & (gt_for_nll < range_max_depth)

# erp_cos_lat_weight: row 별 cos(lat) 가중치 (polar oversampling 보정)
H = gt_for_nll.shape[-2]
lat   = (π/2) - π * (arange(H) + 0.5) / H
pix_w = cos(lat).clamp(min=1e-3)        # (H,)
```

range 분기는 `soft_range_nll_loss`에 이 둘을 전달.
hazard 분기는 hit/free/event/soft_hit에 동일 전달 (`valid_mask=far_valid`, `weights=pix_w`).

---

## 4. Val pass 변경 (`train.py:990-1017`)

```python
elif echorange:
    audio, gtdepth, waveform = batch
    out = model(audio, waveform)
    depth_pred = out["pred_depth"]

    # 라운드-2 신규: 첫 val 배치에서 hazard 진단 dump (1회/val pass)
    if (bi == 0
            and getattr(cfg.model, 'depth_head_type', 'scalar') == 'hazard'
            and 'hazard_alpha' in out):
        _hazard_diagnostics(out, gtdepth, cfg)

    # range/hazard pred 정규화 (라운드-1의 eval bug fix 그대로)
    if cfg.dataset.depth_norm and cfg.model.depth_head_type in ('range', 'hazard'):
        depth_pred = depth_pred / cfg.dataset.max_depth

    pred_for_crit = depth_pred.clamp(min=1e-6)
    lv = criterion(pred_for_crit, gtdepth)
    ...
```

`_hazard_diagnostics`는 1-shot dump:
- `alpha p50/p90/p95/p99` quantile, `frac>{0.5, 0.9, 0.95, 0.99}` (saturation 진단).
- `bg_weight p10/p50/p90` (background mass).
- `argmax_bin_hist` (R bin → 12 fold-bucket %, 분포 collapse 진단).
- `range_entropy`가 있으면 `ent p10/p50/p90`.
- depth bin-sliced ABS_REL/RMSE — `[0.1,1) [1,3) [3,6) [6,9.8) [9.8,10)`.

> 주의: round-2 cell들은 *진단 dump 도입 전에* 학습됐기 때문에 train logs에 `[haz]` 라인이 없음. **round-4 cell부터** 진단 출력이 살아남.

> Val pass의 score 정의(`best_score`)는 round-1과 동일: `0.5·RMSE + ABS_REL`.

---

## 5. Test pass 변경 (`utils/test_utils.py:179-194`)

```python
elif echorange:
    audio, depthgt, waveform = batch
    raw_out = model(audio, waveform)
    depth_pred = raw_out["pred_depth"]

    # 라운드-1 eval bug fix를 hazard에도 확장
    if (cfg.dataset.depth_norm
            and cfg.model.depth_head_type in ('range', 'hazard')):
        depth_pred = depth_pred / cfg.dataset.max_depth
```

이후의 metric path는 round-1과 동일:
```
gt_map *= max_depth                    # GT [0,1] → metres
pred_map *= max_depth                  # pred [0,1] → metres
pred_map = clip(pred_map, 1e-3, max_depth)
errors.append(compute_errors(gt_map, pred_map))   # ABS_REL/RMSE/Δ1.../MAE
```

---

## 6. CLI 인자 — round-2 신규 (test/train 양쪽)

`train.py` 마지막의 ArgumentParser에 round-2 추가 (round-1 인자 + 다음):

```
--depth-head-type {scalar,range,hazard}     # 'hazard' 신규
# range/hazard 공통
--range-num-bins, --range-bin-spacing, --range-min-depth, --range-max-depth

# range only
--range-soft-label-sigma, --range-output-mode {expectation,median}
--lambda-range-nll

# 공통 loss weights
--lambda-berhu, --lambda-silog, --lambda-entropy-smooth

# hazard only — 라운드-2 도입
--hazard-bias-init                          # σ(bias) ≈ 초기 α_j
--hazard-warmup-epochs                      # smooth ramp 길이 (round-4 기본)
--lambda-hit, --lambda-hit-warmup
--lambda-free, --lambda-free-warmup

# hazard ablation flags — 라운드-2
--disable-hit-loss
--disable-free-loss
--hazard-depth-only

# hazard aux mode (round-4 도입, snapshot 기준)
--hazard-aux-mode {raw_hit,event_nll,survival,soft_hit}
--hazard-event-sigma-bins
--hazard-survival-tau-bins
--hazard-soft-hit-target
--hazard-far-thresh
--hazard-log-delta

# ERP 보정 (round-1과 동일)
--erp-cos-lat-weight, --erp-far-mask
```

`test.py`는 round-1과 동일하게 위 인자들을 모두 받아 `cfg`에 override 후 모델 빌드. **range/hazard head가 동일 ckpt를 다른 output_mode로 평가하려면 `--range-output-mode median` 식으로 override 가능** — 라운드-2의 `exp907_*_TESTmedian` 셀이 이 방식.

---

## 7. Checkpoint 형식

라운드-1과 동일:

```python
{
    'epoch': int,
    'state_dict': model.state_dict(),       # range / hazard head이면 range_bins buffer 포함
    'optimizer': optimizer.state_dict(),
    'best_score': float,
}
```

Inference 주의:
- `range_bins` buffer는 학습 시 hyperparam(`range_min_depth`, `range_max_depth`, `range_num_bins`, `range_bin_spacing`)으로 만들어짐. **로드 시 cfg에 동일 값 필수**.
- hazard ckpt는 `hazard_alpha`/`weights`/`bg_weight`가 *output*이라 buffer로 저장되지 않음. forward에서 매번 재계산.
- bias_init는 *bias 텐서에 들어감*이므로 ckpt에 자동 보존.

---

## 8. 코드 snapshot 위치

라운드-1과 라운드-2 통합 snapshot: `logs/round1_bin-based/code_snapshot/` (round1/round2 두 라운드의 상위 폴더 직속). 라운드-1 시점의 frozen 코드(softmax range head only, no hazard)는 `logs/round1_bin-based/code_snapshot/round1_frozen/`에 archival 보관.

unified `code_snapshot/`의 내용:

| 파일 | 출처 | round-1 ↔ round-2 차이 |
|------|------|----------------------|
| `echorange.py` | `models/bin_based/echorange.py` | hazard head 빌드 분기 추가 |
| `range_head.py` | `models/bin_based/range_head.py` | `HazardRangeDepthHead`, `hazard_supervision_loss`, `rendered_event_nll`, `survival_loss`, `soft_hit_bce_loss`, `hazard_free_loss` 신규 |
| `echorange.yaml` | `config/echorange.yaml` | `lambda_silog: 0.5` 정렬, hazard 기본값 추가 |
| `echodiffusion.yaml` | `config/echodiffusion.yaml` | round-1 snapshot에 없던 echodiff 본가 cfg (exp912/920 비교용) |
| `_train_step_echorange.py` | `train.py:54-386` | hazard 분기 + smooth ramp + aux mode dispatch + `_hazard_diagnostics` 함수 |
| `_val_pass_echorange.py` | `train.py:988-1050` | hazard 진단 dump 호출, 정규화 path를 hazard에도 확장 |
| `_test_utils_evaluate_echorange_branch.py` | `utils/test_utils.py:170-225` | range/hazard 공통 정규화 |
| `n9_bulk_0427.sh` | `scripts/n9_bulk_0427.sh` | round-2 launcher (n9 server, bs=32, exp907–912) |
| `n9_bulk_0427_re.sh` | `scripts/n9_bulk_0427_re.sh` | exp910 재학습 launcher (broadcast bug fix 후) |
| `n2_bulk_0428.sh` | `scripts/n2_bulk_0428.sh` | round-3 launcher (n2 server, DP, bs=48, exp913–920) |

> snapshot은 **2026-04-28 시점의 정지 상태**이며 round-4 변경(smooth ramp, aux_mode dispatch)이 이미 들어가 있음. round-2의 *원본* train-step (jump warmup) 재현이 필요하면 `_train_step_echorange.py`에서 ramp_progress 계산을 다음으로 교체:
>
> ```python
> if epoch <= ramp_ep:
>     lam_aux  = lam_hit_warm
>     lam_free = lam_free_warm
> else:
>     lam_aux  = lam_aux_target
>     lam_free = lam_free_target
> ```

---

## 9. 한눈에 보는 round-2 데이터 흐름 (hazard 추가)

```
batch ──► encoder ──► decoder ──► feat (B, 192, h, w)
                                          │
       ┌──────────────┬──────────────────┴──────────────────┐
       ▼              ▼                                      ▼
  scalar head     RangeDepthHead                       HazardRangeDepthHead
  Conv→σ×10       Conv→softmax                         Conv→σ→cumprod→render
  pred (m)        pred = Σp·r (m)                       pred = Σw·r + w_bg·max (m)
                  + range_entropy                       + α, weights, bg_weight

학습:
  분포(range)  : λ_NLL·soft_NLL(logits, gt_m, bins) + λ_b·BerHu + λ_s·SILog
  hazard       : λ_aux(ramp)·primary + λ_free(ramp)·free  + λ_b·BerHu + λ_s·SILog
                 primary ∈ {raw_hit BCE, event_nll, survival, soft_hit}

평가:
  pred_depth → /max_depth (range/hazard) → ×max_depth → clip → compute_errors
              + (hazard val일 때 1-shot) _hazard_diagnostics(α/bg/argmax/sliced)
```

---

## 10. 이전 라운드와의 호환성 / migration 노트

- **Round-1 (exp900–906) cell을 본 코드로 재학습**: yaml의 `lambda_silog=0.5`로 silog 가중치가 바뀌었으니, round-1 결과를 *정확히* 재현하려면 `--lambda-silog 1.0`을 명시적으로 넘겨야 함.
- **Round-2 raw_hit jump 재현**: snapshot의 `_train_step_echorange.py`는 smooth ramp가 적용된 round-4 base. raw_hit jump를 그대로 재현하려면 `--hazard-aux-mode raw_hit`로 + 위 §8의 코드 변경 필요. 또는 cell exp930 (`n2_bulk_0428_r2.sh:282`)이 ramp 적용 상태로 raw_hit를 5 epoch만 돌려 실패 signature를 재확인.
- **Hazard ckpt + range_output_mode**: hazard 출력에는 `range_output_mode`가 의미 없음 (renderer가 mode 합성). Median switch(test_only)은 *softmax range head ckpt*에만 적용 가능.
