# Audio→ERP-depth 모델 — Hazard head 실패 진단 + 다음 디자인 자문 요청

> **Audience**: 외부 LLM (ChatGPT-5.5 등) consultant.
> **Mode**: 자유 사고실험 환영. 기존 결과·코드를 보고 **(1) 진단 검증, (2) 다음
> 라운드 셀 디자인 추천**을 부탁드립니다. 상충하는 권장이 있을 때는 *왜*를
> 설명해 주세요. 본 문서는 self-contained — 외부 코드베이스 액세스 없이도
> 답변할 수 있게 작성되었습니다.

---

## 0. 빠른 요약 (TL;DR)

- **태스크**: binaural spectrogram + waveform → ERP radial depth (256×512, 0.1–10 m).
- **현재 SOTA on this codebase (radial depth)**: original `echodiffusion` exp11
  ABS_REL **0.4300**, RMSE **1.1060**, δ1 **0.4876** (40 epoch, bs=32).
- **분포 head 확장 (softmax range, exp906)**: ABS_REL 0.4814, RMSE 1.2532, δ1
  0.5079 — δ1만 echodiff 능가. RMSE는 12 % 상대 악화. **expectation blur**가
  의심.
- **첫 시도 hazard head (방금 학습)**: λ_hit > 0인 모든 셀에서 RMSE 1.9+ 폭발.
  L_hit이 sigmoid α를 1로 saturate시켜 grad vanish가 발생한 것으로 진단함.
  L_hit을 끈 셀(L_free + BerHu/SILog only)은 RMSE 1.42로 정상 범위.
- **자문 요청**:
  1. 위 진단이 맞나? 다른 가설이 있나?
  2. Hazard 방향을 살리려면 어떤 셀을 다음에 돌려야 하나?
  3. 또는 hazard를 접고 다른 분포 head 디자인 (Bernoulli mixture, ordinal,
     focal-softmax 등)으로 갈아타야 하나?

---

## 1. 시스템 / 데이터 / 메트릭 컨텍스트

### 1.1 입출력

```
batch = (audio, gt_depth, waveform)
  audio:    (B, 2, 256, 512)  binaural log-magnitude STFT (n_fft=512, hop=160)
  gt_depth: (B, 1, 256, 512)  ERP radial depth, normalized to [0,1] = real/10m
                              GT > 10m clamped to 1.0 (sky/far saturation)
  waveform: (B, 2, 5648)      binaural raw audio @ 16 kHz (≈0.353 s)
```

### 1.2 모델 (echorange = echodiffusion + 교체 가능 head)

```
audio_spec (B,2,256,512) ─┐
                          ├─ EcoDepthEncoder (CIDE/wav2vec2 cross-attn)
waveform   (B,2,5648)  ───┘                       │
                                                  ▼
                          Decoder ─── feat (B, 192, 256, 256)
                                            │
                          ┌─────────────────┴──────────────────┐
                          ▼                                    ▼
                  scalar / softmax-range / **hazard** head   →  pred_depth (B,1,H,W)
```

- 모델 크기: 132.6 M params (CIDE + wav2vec2 frozen 포함).
- Encoder는 spec을 (128, 128)로 bilinear-resize 후 진입 (echodiffusion 본가
  코드 그대로). Decoder가 2× upsample → feat 해상도 (256, 256).
- 모든 head는 동일한 3-conv 스택 (`Conv→ReLU→Conv`)으로 구현되어 capacity
  parity 유지. scalar head 출력 1ch, range/hazard head 출력 32ch (=Br).

### 1.3 데이터셋

- **SoundSpaces (Matterport3D scenes)**: 23 560 train / 2 951 val / 2 951 test.
  scene-disjoint split. 단일 source → 단일 receiver의 binaural / FOA RIR.
- ERP radial depth (`erp_depth_radial/`)를 사용 — 각 ERP 픽셀 = (lat, lon)
  방향 ray 거리. `erp_depth/`(planar Z)는 별도 존재하나 본 라운드에서 미사용.

### 1.4 Eval metric (전부 반환)

`compute_errors(gt, pred)` returns 7 scalars per sample:
- ABS_REL = mean(|pred - gt| / gt)        # **report 주 지표**
- RMSE    = sqrt(mean((pred - gt)^2))      # 거리 단위 (m)
- Delta1  = % pixels with max(pred/gt, gt/pred) < 1.25
- Delta2/3, Log10, MAE

clip rule: pred to [1e-3, 10], gt to [0, ∞).

---

## 2. 라운드별 진행 (radial depth, 모두 동일 데이터셋)

### 2.1 Round 0 — original echodiffusion (baseline, 별도 코드)

| exp | LR | BS | Epochs | ABS_REL | RMSE | δ1 |
|-----|---:|---:|---:|---:|---:|---:|
| exp11  (best) | 1e-4 | 32 | 40 | **0.4300** | **1.1060** | 0.4876 |
| exp363       | 5e-4 | 48 | 40 | 0.4482 | 1.2198 | 0.4936 |
| exp13        | 1e-4 | 16 | 40 | 0.4504 | 1.1134 | 0.4930 |
| exp362       | 1e-4 | 32 | 40 | 0.4557 | 1.2292 | 0.4923 |

→ **기준선: ABS_REL 0.43 / RMSE 1.11 / δ1 0.49**.

### 2.2 Round 1 — softmax range head sweep (n9_bulk, bs=16, 20 epoch)

```
Loss = λ_NLL · soft_range_NLL(logits, gt_metres, log_bins, σ)
     + λ_BerHu · BerHu(pred_depth/max, gt_norm)
     + λ_SILog · SILog(pred_depth/max, gt_norm)
pred_depth = Σ_j p_j · r_j        # expectation
λ_NLL = λ_BerHu = λ_SILog = 1.0
```

| exp | bin | σ | extra | ABS_REL | RMSE | δ1 |
|----:|----:|---:|---|---:|---:|---:|
| 900 (scalar baseline) | – | – | bs=64 DP | 0.5229 | 1.2284 | 0.4854 |
| 901 | 4  | 1.15 | – | 0.5233 | 1.2393 | 0.4848 |
| 902 | 8  | 0.58 | – | 0.5489 | 1.2849 | 0.4853 |
| 903 | 16 | 0.29 | – | 0.5321 | 1.2632 | 0.4955 |
| 904 | 32 | 0.14 | – (anchor) | 0.4888 | 1.2459 | 0.4988 |
| 905 | 32 | 0.30 | wider σ | 0.5106 | 1.2621 | 0.4899 |
| **906** | **32** | **0.14** | **cos-lat + far-mask** | **0.4814** | 1.2532 | **0.5079** |

**Round 1 결론**:
- bin=32 + ERP-aware loss가 sweet spot.
- δ1에서 echodiff 능가, **RMSE는 0.14 m 더 큼 (~12 % 상대)**.
- 진단: expectation `Σ p_j · r_j`가 multi-modal 분포(반사 모호 등)를 평균내서
  pred를 GT 표면에서 끌어내림 → RMSE 손해.

### 2.3 Round 2 — bs=32 sweep + audit (n9_bulk_0427, 진행 중·일부 학습 미완)

이 라운드는 본 자문 이슈와 직접 관련 없으니 생략. 핵심은 bs=16→bs=32
이동이 echodiff와 격차를 충분히 좁히지 못함을 확인.

### 2.4 Round 3 — A-Hazard head (n2_bulk_0428.sh, **현재 sweep**)

**디자인 의도**: expectation blur 제거. 각 픽셀에서 first-hit (가장 가까운
표면 명중)을 differentiable rendering으로 commit하면 multi-modal 분포의
한 mode를 잡을 수 있고, BerHu/SILog와 정렬된 단일-mode pred를 얻을 수 있음.

#### 핵심 수식

```
α_j = sigmoid(logits_j)                  ∈ (0,1)   # per-bin hazard
T_j = ∏_{k<j} (1 - α_k)                  # transmittance, T_0 = 1
w_j = T_j · α_j                           # first-hit weight at bin j
w_bg = ∏_j (1 - α_j)                      # no-hit / background prob.
pred_depth = Σ_j w_j · r_j + w_bg · max_depth      # rendered depth (m)

# Σ_j w_j + w_bg = 1   (보장됨, telescoping)
```

#### Hit / Free 보조 손실

```
log_delta = log(r_{j+1} / r_j) / 2 = log(10/0.1) / (2·31) ≈ 0.073   # log-spacing 절반 폭

For each valid pixel with GT D:
  hit_mask  = { j : |log r_j - log D| ≤ log_delta }     (보통 1-2개 bin)
  free_mask = { j : log r_j - log D <  -log_delta }     (D보다 가까운 bin들)
  behind    = otherwise → 무감독

L_hit  = mean over (pixel, hit_bin) of  BCEwithLogits(logits, target=1)
L_free = mean over (pixel, free_bin) of BCEwithLogits(logits, target=0)
```

ERP cos-lat weight + far-mask는 `valid_mask` / `weights`로 NLL 시에 적용
(round1 exp906과 동일).

#### Total loss + warmup

```
warmup (epoch ≤ 3):   L = L_depth + 0.3 · L_hit + 0.05 · L_free
post-warmup:           L = L_depth + 0.5 · L_hit + 0.10 · L_free
L_depth = 1.0 · BerHu(pred/max, gt_norm) + 0.5 · SILog(pred/max, gt_norm)
```

`hazard_bias_init = -4.6` → α_init ≈ sigmoid(-4.6) = 0.01 → "처음에는 거의
free space" 상태에서 시작.

#### Round 3 실험 셋업

- bs = 48, lr = 1e-4, 20 epoch, AdamW.
- 2-GPU DataParallel × 4 worker 병렬 (8-GPU 노드).
- Bin grid: 32 log-spaced bins, [0.1, 10] m.
- ERP cos-lat + far-mask on (round1 exp906 동일).
- silog 가중치 0.5 (round 2 audit fix).

---

## 3. Round 3 실측 결과 (현재까지)

### 3.1 Val metric (epoch 2 best, epoch 4 전후)

| exp | L_hit | L_free | Val RMSE @ best | Val ABS_REL @ best | δ1 @ best |
|---|---|---|---:|---:|---:|
| **exp913** full   | ✓ (0.3→0.5) | ✓ (0.05→0.1) | **1.93** | 0.44 | 0.24 |
| **exp914** no-free | ✓ (0.3→0.5) | ✗            | **1.89** | 0.41 | 0.28 |
| exp915 no-hit     | ✗ | ✓ (0.05→0.1) | 1.42  | 0.53 | 0.49 |
| exp916 depth-only | ✗ | ✗ | 1.43  | 0.45 | 0.46 |

비교 기준:
- 같은 셋업의 softmax range head best (round1 exp906): **RMSE 1.25, ABS_REL 0.48, δ1 0.51**.
- 원조 echodiffusion best: **RMSE 1.11, ABS_REL 0.43, δ1 0.49**.

→ **L_hit-on cells (913, 914) → RMSE 1.9 (목표 대비 +0.7 m, 60 % 악화)**.
→ **L_hit-off cells (915, 916) → RMSE 1.42 — softmax 베이스라인과 비슷**.

### 3.2 Train loss 추이 (D = BerHu + 0.5·SILog)

```
exp913 (full):     0.60 → 0.49 → 0.47 → 0.52 → 0.51    ← epoch 4에서 jump (warmup 끝)
exp914 (no-free):  0.64 → 0.52 → 0.50 → 0.56 → 0.55    ← 마찬가지
exp915 (no-hit):   0.45 → 0.35 → 0.33 → 0.32 → 0.31    ← 단조 감소
exp916 (depth-only):0.45 → 0.35 → 0.33 → 0.32 → 0.31   ← 단조 감소
```

→ **smoking gun**: warmup 종료 (epoch 3→4)에서 λ_hit 0.3 → 0.5로 점프하는
   순간 L_hit-on 셀들의 D가 0.47→0.52로 *튀어 오름* (악화). L_hit-off 셀은
   같은 epoch에서 0.33→0.32로 매끄럽게 감소.

### 3.3 진단 (저자 제안 — 검증 부탁드립니다)

L_hit = `BCE(logits, target=1)`은 hit-bin α를 1로 끌어올리려 함.

1. 워밍업 직후: λ_hit=0.3, L_hit=4.6 → 가중 contrib 1.4 (BerHu+SILog ≈ 1.1보다 큼).
   옵티마이저가 L_hit 줄이는 데 우선 → α_hit이 0.5–0.9까지 빠르게 상승.
2. Epoch 4: λ_hit 0.5로 점프. α_hit이 0.95+로 saturate.
3. Sigmoid 미분 `∂α/∂logit = α(1−α)`. α=0.99에서 0.0099, α=0.999에서 0.001
   → **gradient vanish**.
4. BerHu/SILog가 "이 픽셀의 pred_depth가 잘못됐다"는 신호를 내도, hit-bin α를
   움직일 grad가 0에 가까움 → **renderer가 잘못된 첫 commit에서 stuck**.
5. 그 결과 train D plateau, val RMSE 폭발.

대조로 L_hit-off 셀(exp915, 916)은 BerHu/SILog만으로 renderer가 부드럽게
끌려가 정상 학습. exp915는 L_free만 있으니 free bin α도 0으로 눌려서
첫 hit이 자연스럽게 형성됨.

---

## 4. 핵심 코드 (verbatim, 원본은 `models/bin_based/range_head.py`)

### 4.1 Hazard head forward

```python
class HazardRangeDepthHead(nn.Module):
    def __init__(self, in_channels, num_bins=32, r_min=0.1, r_max=10.0,
                 spacing="log", max_depth=10.0, bias_init=-4.6, eps=1e-8):
        super().__init__()
        # log-spaced bin centres in metres
        self.register_buffer("range_bins",
            torch.exp(torch.linspace(math.log(r_min), math.log(r_max), num_bins)))
        # 3-conv stack mirroring scalar head's capacity
        self.logit_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_bins, 3, 1, 1),
        )
        nn.init.constant_(self.logit_conv[-1].bias, bias_init)   # → α_init ≈ 0.01
        self.max_depth, self.eps = float(max_depth), float(eps)
        self.num_bins = int(num_bins)

    def forward(self, feat):
        logits = self.logit_conv(feat)                           # (B, R, H, W)
        # CRITICAL: α clamped away from {0, 1} for cumprod / log stability
        alpha = torch.sigmoid(logits).clamp(min=self.eps,
                                            max=1.0 - self.eps)

        T_inclusive = torch.cumprod(1.0 - alpha, dim=1)          # ∏_{k≤j}(1-α_k)
        T_prefix    = torch.cat(
            [torch.ones_like(T_inclusive[:, :1]), T_inclusive[:, :-1]],
            dim=1)                                               # ∏_{k<j}(1-α_k)
        weights   = T_prefix * alpha                             # w_j
        bg_weight = T_inclusive[:, -1:]                          # w_bg

        bins = self.range_bins.view(1, -1, 1, 1)
        pred_depth = (weights * bins).sum(dim=1, keepdim=True) \
                   + bg_weight * self.max_depth                  # m
        return {"pred_depth": pred_depth, "range_logits": logits, ...}
```

### 4.2 Hazard supervision loss

```python
def hazard_supervision_loss(logits, target_depth, range_bins,
                            log_delta=None, valid_mask=None,
                            weights=None, use_hit=True, use_free=True,
                            eps=1e-8):
    target = target_depth                              # (B, 1, H, W) metres
    if log_delta is None:
        ratio = (range_bins[1] / range_bins[0]).item()
        log_delta = math.log(ratio) * 0.5              # half-bin in log space

    log_target = torch.log(target.clamp(min=eps))
    log_bins   = torch.log(range_bins.view(1, -1, 1, 1).clamp(min=eps))
    diff = log_bins - log_target                       # (B, R, H, W)

    valid = torch.isfinite(target) & (target > 0)
    if valid_mask is not None:
        valid = valid & valid_mask.bool()

    free_mask = (diff < -log_delta) & valid            # bins closer than D
    hit_mask  = (diff.abs() <= log_delta) & valid      # ±half-bin around D
    # behind: not supervised

    losses = {}
    if use_hit:
        bce = F.binary_cross_entropy_with_logits(
                logits, torch.ones_like(logits), reduction='none')
        m = hit_mask.float() * (weights or 1.0)
        losses['hit'] = (bce * m).sum() / m.sum().clamp(min=eps)
    if use_free:
        bce = F.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits), reduction='none')
        m = free_mask.float() * (weights or 1.0)
        losses['free'] = (bce * m).sum() / m.sum().clamp(min=eps)
    return losses
```

### 4.3 Train step (요약)

```python
out = model(audio, waveform)                # dict with 'pred_depth', 'range_logits', ...
gt_m = gt * max_depth                       # gt is normalized; lift to metres
gt_for_nll = F.interpolate(gt_m, size=logits.shape[-2:], mode='nearest')

# Per-cell warmup-aware weights:
λ_hit  = 0.3 if epoch ≤ 3 else 0.5
λ_free = 0.05 if epoch ≤ 3 else 0.10

haz = hazard_supervision_loss(logits, gt_for_nll, range_bins, ...)
head_loss = λ_hit * haz['hit'] + λ_free * haz['free']

pred_norm = (out['pred_depth'] / max_depth).clamp(min=1e-6)
berhu = BerHuLoss()(pred_norm, gt)          # gt already normalized
silog = SILogLoss()(pred_norm, gt)

total = head_loss + 1.0 * berhu + 0.5 * silog
total.backward()
```

---

## 5. 다음 라운드 후보 (저자 5종 — 검토·우선순위 부탁드립니다)

저자가 "L_hit이 sigmoid saturate를 일으켜 stuck" 가설을 받아들이고 떠올린
후보들. 각각 1-cell 비용 (≈ 2 시간 학습) 정도. 어느 것을 우선해야 할지,
또는 다른 디자인이 더 약속적인지 자문 부탁드립니다.

### 후보 A — λ_hit 대폭 축소

```
λ_hit_warmup = 0.05 (was 0.3),  λ_hit = 0.10 (was 0.5)
```
가설: L_hit을 약한 hint로만 두고 BerHu/SILog가 주도. saturation 진입 자체를
방지. 단점: hit bin 학습 자체가 약해질 위험.

### 후보 B — Hit target을 0.9 (logit ≈ 2.2) 로 캡

```python
target_hit = 0.9 * torch.ones_like(logits)
# BCE_with_logits 대신 BCE on probabilities, or smooth-target version
```
가설: "α를 1로 만들지 말고 ~0.9에서 멈춰라"라는 신호. sigmoid 포화 직전에서
gradient finite 유지.

### 후보 C — α를 forward에서 [eps, 0.95] hard-clamp

```python
alpha = torch.sigmoid(logits).clamp(min=eps, max=0.95)
```
가설: 모든 픽셀에서 ∂α/∂logit ≥ 0.95·0.05 = 0.0475 보장. α=1 saturation
물리적으로 불가능하게 막음. 단점: gradient가 clamp 경계에서 0이 되긴 함
(passthrough straight-through 안 쓰면), 하지만 적어도 미만 영역에서 부드러움.

### 후보 D — L_hit을 (1−α)^γ focal weight로 가중

```python
focal = (1 - alpha.detach()) ** gamma   # γ ≈ 2
bce = bce * focal
```
가설: 이미 α가 높은 픽셀에서는 자동으로 grad weight가 줄어 saturation 회피.
단점: focal hyperparam 추가, behavior가 γ에 민감.

### 후보 E — L_hit을 워밍업에만 쓰고 post-warmup 0

```
warmup λ_hit = 0.3, post-warmup λ_hit = 0
λ_free unchanged (0.05 → 0.10)
```
가설: 초기에 hit bin α를 적당히 올려놓기만 하고, 그 다음은 BerHu/SILog가
fine-tune. exp915 결과(L_hit 처음부터 0)와 합치면 "warmup hit이 도움이
되긴 하나?"라는 질문도 동시 답.

### 후보 F (대안 방향) — Hazard 접고 다른 분포 head

- **Bernoulli mixture / depth-anything-style**: per-pixel K개 mode (μ_k, σ_k, π_k)로
  posterior 학습, mode 중 max-π를 pred로.
- **Ordinal regression head** (DORN-style): bin을 cumulative classification으로
  학습, expectation 대신 "P(D > r_j) = 0.5"의 r_j를 pred로.
- **Soft median head**: 같은 logits에서 expectation 대신 학습 시간에도 median
  사용 (round1에서 inference-only median 시도는 round2 exp910에 별도 진행 중).

---

## 6. 자문 요청 (구체적 질문)

다음 5개에 대해 의견 부탁드립니다. 각각 *한 단락 정도*면 충분합니다.

### Q1 — 진단 검증
저자의 진단 (L_hit이 sigmoid α를 saturate시켜 grad vanish가 stuck을 만든다)이
충분한가요? 다른 더 그럴듯한 가설은 없나요?
- 가능한 alt 가설:
  - hit window 폭(log_delta=0.073)이 너무 좁아 노이즈가 큰 supervision
  - L_hit gradient가 logits를 너무 빠르게 키워서 cumprod에서 T_prefix가
    near-zero → renderer 출력이 단일 bin으로 collapse → BerHu/SILog gradient도
    그 bin에 pile-up
  - far_mask 경계에서 (D ≈ 10) hit_mask가 사실상 last bin만 잡음 → last bin α
    독주 → bg_weight = 0 → renderer 출력 = bins[-1] = 10 m fixed
- 위 중 어느 것이 dominant라고 보시나요?

### Q2 — 후보 A–E 중 우선순위
구현 cost가 동일하다면 (위 5개 모두 1줄–수줄 변경), 어느 순서로 시도하시겠나요?
"가장 정보량이 많은 1셀"을 먼저 돌리고 싶습니다.

### Q3 — Hazard 디자인 자체의 적합성
오디오 → ERP depth 태스크에서 hazard rendering이 정말 적합한 prior인가요?
NeRF 계열의 hazard rendering은 **multi-view consistency 가정** 위에서
설계됐는데, 우리 setup은 single-source single-receiver의 ill-posed
inverse problem이라 first-hit prior가 오히려 noise를 굳힐 위험은 없나요?

### Q4 — bin 그리드 설계
현재 32 log-bins on [0.1, 10]. 셀 폭이 멀어질수록 넓어짐 (마지막 bin 폭 ≈
1.4 m). hit window log_delta=0.073이 → 마지막 bin에서 ±10 cm 정도지만
실제 GT 분포에서 그 영역이 saturated mass spike (sky/far). 다른 bin 그리드
(linear, hybrid, 64 bins, 비대칭 spacing)가 hazard에 더 잘 맞을지 의견 있으신가요?

### Q5 — 저자가 놓치고 있는 게 있나
실험 디자인, 진단, fix 후보 어디든 — "이 사람들이 이걸 모르는 듯하다",
"이걸 검증 안 했네" 같은 게 있다면 지적 부탁드립니다. 특히:
- DataParallel BatchNorm 같은 분산학습 idiosyncrasy가 cumprod 같은 비선형
  연산에서 어떤 영향을 줄지
- α가 hard-clamp 경계에 닿으면서 logits만 무한정 커지는 vacuous loss 시나리오
- SILog가 (pred / gt)의 log-variance를 패널티하는데 hazard renderer의 단일-mode
  commit이 SILog gradient에 어떻게 상호작용하는지

---

## 7. 참고: 디렉터리 / 산물 위치 (외부 답변자에게는 무관, 메모용)

```
logs/round1_bin-based/round1/                 round 1 결과 + code snapshot
logs/n9_0427_train/exp9{07,08,11,12}*.log     round 2 실측 (진행 중·일부 미완)
logs/n9_0427_train/exp91{3..7}*bs48.log       round 3 실측 (현재 sweep)
models/bin_based/range_head.py                Softmax + Hazard head 코드
train.py                                       _train_step_echorange (hazard 분기)
scripts/n2_bulk_0428.sh                       round 3 sweep launcher
```

---

## 8. 추가 컨텍스트: noise floor

- echodiffusion family HP-induced std on RMSE: **0.0336** (single-cell 비교
  noise). 0.05 RMSE 이상 차이가 나야 의미 있음.
- δ1은 분포 head가 echodiff 능가하는 유일한 metric (round1 exp906: 0.5079 vs
  echodiff 0.4876). 분포-기반 head가 confidence 기반 metric에 유리한 듯.
- ABS_REL은 분포 head가 약한 metric. expectation blur가 평균적으로 GT보다
  낮게 측정되는 경향이 ABS_REL 정의(|diff|/gt)에 불리함.

답변에 사용하실 만한 모든 컨텍스트는 위에 다 포함되었습니다. 답변은 한국어
또는 영어 어느 쪽이든 좋습니다. 길이 제한 없음.
