# E20260428 — exp900–906 모델 입출력 및 코드 description

이번 sweep에서 사용된 핵심 모델 코드와 입출력 텐서의 의미. 코드 snapshot은
`code_snapshot/` 하위에 동봉.

---

## 1. 데이터 입력 (`data/dataset.py` → `SoundSpacesDataset.__getitem__`)

batch 1개당 3-tuple `(audio, gt_depth, waveform)`:

| 텐서 | 형태 | 단위 / 의미 |
|------|------|--------------|
| `audio` | `(2, 256, 512)` | binaural spectrogram. 2-channel (좌/우 마이크), STFT log-magnitude. n_fft=512, hop=160, win=400 후 `(256, 512)`로 nearest-interpolate |
| `gt_depth` | `(1, 256, 512)` | ERP radial depth map. 픽셀 = (latitude, longitude) 방향 ray 거리. `depth_norm=true`라 [0, 1] 정규화 (실제 깊이 / `max_depth=10`). `>10 m`은 1.0으로 클램프 |
| `waveform` | `(2, 5648)` | binaural raw waveform. 5648 sample (≈0.353 s @ 16 kHz). 패딩 또는 truncate로 길이 통일 |

DataLoader에서 batch dim 추가 → 모델 입력은 각각 첫 차원에 B가 붙음:
- `audio`: `(B, 2, 256, 512)`
- `gt_depth`: `(B, 1, 256, 512)`
- `waveform`: `(B, 2, 5648)`

설정 출처 (`config/echorange.yaml`):
```yaml
dataset:
  input_type: echoes
  audio_format: spectrogram
  depth_type: erp
  depth_norm: true
  images_size: [256, 512]
  min_depth: 0.01
  max_depth: 10.0
  use_waveform: true
  waveform_len: 5648
```

---

## 2. 모델 (`models/bin_based/echorange.py` — `EchoRangeDepth`)

EchoDiffusion encoder + decoder를 그대로 쓰고, **출력 head만 scalar 또는
range로 토글**. 두 head는 같은 forward 시그니처를 공유하는 dict를 반환해
downstream 코드가 head 분기를 안 봐도 되게 설계.

### Forward 시그니처

```python
out = model(audio_spec, audio_wave)
```

| 입력 | 형태 | 비고 |
|------|------|-------|
| `audio_spec` | `(B, 2, H_in, W_in)` | 모델 내부에서 `(B, 2, 128, 128)`로 bilinear-resize 후 encoder |
| `audio_wave` | `(B, 2, T)` | encoder가 spec과 함께 활용 |

### Forward 출력 (dict)

```python
{
    'pred_depth':     (B, 1, H, W),          # 항상 — 단일 깊이 추정 (metres)
    'range_logits':   (B, Br, h, w),         # range head only — bin별 raw logit
    'range_prob':     (B, Br, h, w),         # range head only — softmax(logits)
    'range_entropy':  (B, 1, H, W),          # range head only — pixel별 -Σp log p
    'range_bins':     (Br,),                 # range head only — bin centre (metres)
}
```

- `H, W`는 입력 spec과 동일 `(256, 512)`. `h, w`는 decoder 내부 해상도 (보통 `H/4, W/4` 정도).
- **`pred_depth`는 두 head 모두 metres**. `[0.1, 10.0]` 범위에 포함.
  - scalar head: `pred = sigmoid(self.scalar_head(feat)) * self.max_depth` → metres.
  - range head: `pred = Σ_j p_j · r_j`, `r_j ∈ [0.1, 10]` → metres.
- val/test 평가 시에는 `depth_norm=true` 가정 하에 `pred_map *= max_depth`가 일률적으로 호출되므로, range head의 `pred_depth`는 metric path 진입 직전 `/= max_depth`로 [0, 1]로 한 번 normalize함 (eval 버그 fix). scalar head는 학습 dynamics 상 수치적으로 [0, 1]에 머물러서 그대로 통과.

### scalar vs range head 구조

- **scalar head**: `Conv2d(C, 1, 1) → sigmoid → × max_depth` (`echorange.py:101`)
- **range head** (`models/bin_based/range_head.py`):
  - `Conv2d(C, Br, 1)` 로 logit 출력
  - `softmax(logits, dim=1)` → 분포 `p ∈ [0,1]^Br`
  - `pred_depth = Σ_j p_j · range_bins_j` (expectation) 또는 CDF 기반 median
  - `range_bins`은 `[range_min_depth, range_max_depth] = [0.1, 10.0]`을 log spacing으로 Br등분한 bin centre. `nn.Module`의 `register_buffer`로 저장돼 checkpoint에 같이 들어감.

---

## 3. Loss (`train.py` → `_train_step_echorange`)

range head 모드의 학습 loss:

```
L = λ_NLL · soft_range_NLL(logits, gt_for_nll, range_bins, σ, weights, valid_mask)
  + λ_BerHu · BerHu(pred_norm, gt_norm)
  + λ_SILog · SILog(pred_norm, gt_norm)
  [+ λ_ent  · mean(range_entropy)]      # 옵션, 이번 sweep은 0
```

- `λ_NLL = λ_BerHu = λ_SILog = 1.0` (이번 sweep 고정).
- `pred_norm = pred_depth / max_depth ∈ [0.01, 1.0]`, `gt_norm = gt_depth ∈ [0, 1]` (둘 다 normalised).
- soft_range_NLL의 GT는 `gt_for_nll = gt_depth * max_depth` (다시 metres로 복원, range_bins와 동일 단위).

### `soft_range_nll_loss` (`models/bin_based/range_head.py`)

```python
def soft_range_nll_loss(logits, target_depth, range_bins,
                         valid_mask=None, sigma=0.08,
                         eps=1e-8, weights=None):
```

각 픽셀 GT 깊이 `D`에 대해 log-space Gaussian soft label:

```
log q_j = -(log r_j - log D)^2 / (2 σ^2),  normalize via logsumexp over j
ce_pixel = -Σ_j q_j · log p_j
```

- `valid_mask=None` 또는 (B,1,H,W) bool. NaN/Inf와 ≤0 픽셀은 자동 제외.
  이번 sweep의 `--erp-far-mask`는 `(D > 0) & (D < range_max_depth)` 로 마스킹.
- `sigma`: 이번 sweep에서 `log-bin-spacing = ln(10/0.1) / Br`로 설정 (ratio≈1.0).
- `weights`: per-pixel float weight `(B,1,H,W)` 또는 `(H,)`. 이번 sweep의
  `--erp-cos-lat-weight`는 `cos(latitude_per_row)`로 (H,) 텐서 만들어 전달 →
  ERP 폴라 oversampling 보정.

### ERP 보정 (옵션 두 종류, 이번 sweep exp906에서 둘 다 켬)

```python
# in _train_step_echorange:
far_valid = None
if cfg.model.erp_far_mask:
    far_valid = (gt_for_nll > 0) & (gt_for_nll < range_max_depth)

pix_w = None
if cfg.model.erp_cos_lat_weight:
    H = gt_for_nll.shape[-2]
    lat = (π/2) - π * (arange(H) + 0.5) / H
    pix_w = cos(lat).clamp(min=1e-3)        # (H,)

nll = soft_range_nll_loss(
    logits, gt_for_nll, range_bins,
    valid_mask=far_valid, sigma=sigma, weights=pix_w
)
```

BerHu / SILog에는 ERP 보정 안 들어감 (NLL term에만). 다음 sweep에서 이것도
공유할지 결정해야 함.

---

## 4. 평가 (`utils/test_utils.py` → `evaluate`)

```python
elif echorange:
    audio, depthgt, waveform = batch
    raw_out = model(audio, waveform)
    depth_pred = raw_out["pred_depth"]
    # range head는 metres라 metric path 진입 전 한 번 정규화
    if cfg.dataset.depth_norm and cfg.model.depth_head_type == 'range':
        depth_pred = depth_pred / max_depth
    ...
    # 통합 metric path (scalar/range 공용)
    if cfg.dataset.depth_norm:
        gt_map *= max_depth
        pred_map *= max_depth
    pred_map = clip(pred_map, 1e-3, max_depth)
    errors.append(compute_errors(gt_map, pred_map))
```

`compute_errors` (`utils/metrics.py`)가 ABS_REL, RMSE, Delta1/2/3, Log10, MAE
를 계산해서 `(N, 7)` 배열 반환.

---

## 5. Checkpoint 형식

`best_model.pth` 저장 시 dict:

```python
{
    'epoch': int,
    'state_dict': model.state_dict(),       # range head 모드면 range_bins buffer 포함
    'optimizer': optimizer.state_dict(),
    'best_score': float,
    ...
}
```

Inference 시 주의:
- `range_bins` buffer는 학습 시 `range_min_depth`, `range_max_depth`,
  `range_num_bins`, `range_bin_spacing`으로 만들어졌으므로, **체크포인트 로드
  시 동일 값을 cfg에 줘야 buffer shape이 맞음.** 이게 `test.py`의 range CLI
  override가 필요한 이유 (이전엔 `parse_known_args`로 silently drop돼서 yaml
  default로 64-bin head를 만들고 32-bin checkpoint를 로드하다가 size mismatch
  로 실패하는 버그가 있었음).

---

## 6. 코드 snapshot 위치

> **2026-04-28 통합 노트**: 본 round-1 시점의 *frozen* snapshot은 `logs/round1_bin-based/code_snapshot/round1_frozen/`로 이동했고, round1/round2 통합본은 `logs/round1_bin-based/code_snapshot/`(현재/superset)에 있음. 본 표는 round-1 시점의 frozen 사본 내용을 가리킨다 (`code_snapshot/round1_frozen/`):

| 파일 | 출처 | 비고 |
|------|------|------|
| `echorange.py` | `models/bin_based/echorange.py` | 메인 모델 클래스 (scalar/range head 토글) |
| `range_head.py` | `models/bin_based/range_head.py` | RangeDepthHead + soft_range_nll_loss |
| `echorange.yaml` | `config/echorange.yaml` | 데이터 / 모델 / 학습 hyperparam |
| `_train_step_echorange.py` | `train.py` 함수 추출 | echorange 학습 루프 1-step |
| `_val_pass_echorange.py` | `train.py` val 루프 추출 | val 시 range pred 정규화 fix 포함 |
| `_test_utils_evaluate_echorange_branch.py` | `utils/test_utils.py` 추출 | test 시 동일 정규화 fix 포함 |

**주의**: snapshot은 2026-04-28 시점의 정지 상태. 이후 본 코드베이스가 바뀌면
실제 코드와 달라질 수 있음. 재현이 필요하면 이 폴더의 snapshot을 기준으로.

---

## 7. 한눈에 보는 모델 데이터 흐름

```
        ┌──────────────────┐
batch → │ audio (B,2,256,  │
        │        512)      │
        │ gt    (B,1,256,  │
        │        512) [0,1]│
        │ wave  (B,2,5648) │
        └────────┬─────────┘
                 ▼
        ┌────────────────────┐
        │ EcoDepthEncoder    │  spec→128×128 resize 후 진입
        │ + Decoder (192ch)  │  → feat (B, 192, h, w)
        └────────┬───────────┘
                 │
        ┌────────┴──────────────────┐
        ▼                           ▼
  scalar head                  range head
  Conv→sig×10                  Conv(→Br) → softmax
  pred_depth                   pred_depth = Σp·r (metres)
  (B,1,256,512)                + logits, prob, entropy
  metres                        range_bins (Br,)
```

학습:
```
pred_depth → /max_depth → BerHu+SILog (vs gt_norm)
range_logits → soft_NLL (vs gt_metres, optional cos-lat weight, far-mask)
              ↓
            backward / AdamW lr=1e-4
```

평가:
```
pred_depth → (range head면 /max_depth) → ×max_depth → clip [1e-3, 10]
                                                       ↓
                                              compute_errors(gt, pred)
                                              → ABS_REL/RMSE/Delta1...
```
