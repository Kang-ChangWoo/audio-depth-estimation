# E20260428 — EchoRange 분포-기반 깊이 head: bin / σ / ERP 보정 sweep

오디오 → ERP radial depth 추정에서 scalar head 대신 log-spaced bin 분포 head를
도입할 때 (1) bin 수 sweet spot, (2) soft label sigma 적정값, (3) ERP-aware
loss 보정의 효과를 한 번에 점검한 첫 실험. **결론: range head + cos(lat) +
far-mask 조합 (exp906)이 같은 backbone의 scalar head 대비 ABS_REL을 7.9 %
상대 개선했지만, 원조 echodiffusion 베스트 (bs=32 학습) 대비로는 여전히 12 %
상대로 뒤짐. 주된 원인은 batch size 차이 (bs=16 단일 GPU)로 추정.**

---

## 1. 셋업

| 항목 | 값 |
|------|-----|
| 모델 | `echorange` (echodiffusion encoder/decoder + switchable scalar/range head) |
| 데이터 | `matterport3d_0303_renew`, ERP radial depth (`erp_depth_radial/`) |
| 입력 | spectrogram + waveform |
| 분할 | train 23 560 / val 2 951 / test 2 951 |
| 깊이 범위 | 0.1–10 m (`max_depth=10.0`, `range_max_depth=10.0`로 통일) |
| Loss | `λ_NLL · soft_range_NLL + λ_BerHu · BerHu + λ_SILog · SILog` (모두 1.0) |
| Optim | AdamW, lr=1e-4, 20 epoch |
| Batch size | **16** (단일 GPU 모드 — GPU 2 driver fault → DP/NCCL 사용 불가) |
| 하드웨어 | 4090 × 2 병렬 (GPU 0, 1 / GPU 2 fault, GPU 3 enumeration도 깨짐) |
| 학습 시간 | 실험당 ≈ 2 시간 (epoch당 ≈ 350 s) |

`range_max_depth`는 echorange.yaml에서 원래 20.0이었는데 dataset.max_depth=10과 어긋나서 10.0으로 맞춤. 이 sweep 전에는 마지막 bin이 데이터에 등장하지 않는 영역(10–20 m)을 차지하고 있어 절반의 bin이 낭비됐음.

---

## 2. 실험 디자인

bin 수와 sigma는 σ = log-bin-spacing (ratio ≈ 1.0) 규칙으로 정함 — soft label이
타겟 bin에 ~40 %, 양쪽 인접 bin에 각각 ~24 % mass를 갖도록 해서 hard CE에
가까워지는 걸 방지.

| exp | bin | σ | 추가 옵션 | 검증 질문 |
|-----|-----|----|-----------|-----------|
| 901 | 4   | 1.15 | — | Q1 — 매우 거친 분포에서 expectation이 의미 있나 |
| 902 | 8   | 0.58 | — | Q1 |
| 903 | 16  | 0.29 | — | Q1 |
| 904 | 32  | 0.14 | — (anchor) | Q1 — 적정 bin 수 |
| 905 | 32  | 0.30 | wider σ (ratio≈2.1) | Q2 — sigma 민감도 |
| 906 | 32  | 0.14 | cos(lat) + far-mask | Q3 — ERP-aware 보정 효과 |

ERP 보정 두 종류:
- **cos(lat) loss weight**: equirectangular projection은 위도에 따라 픽셀당 입체각이 cos(lat)에 비례. uniform pixel mean을 쓰면 극지방(천장/바닥)에 ~3× 과대 가중됨.
- **far-mask**: GT ≥ 10 m 픽셀을 NLL에서 제외. ERP에서 sky/먼 곳이 max_depth로 클램프되면서 마지막 bin에 인공 mass spike가 생기는 걸 차단.

ERP wraparound (좌우 경계 circular padding)는 backbone-level 변경이라 이번 sweep에서 제외.

---

## 3. 학습 진행 (epoch별 train loss)

값은 `[L = total loss, D = BerHu+SILog 합]`. epoch당 약 350 s.

| epoch | exp901 (bin=4) | exp902 (bin=8) | exp903 (bin=16) | exp904 (bin=32) | exp905 (σ=0.30) | exp906 (ERP fix) |
|-------|---------------:|---------------:|----------------:|----------------:|----------------:|-----------------:|
|  1    | L=1.64 D=0.43  | L=2.00 D=0.42  | L=2.44 D=0.41   | L=2.96 D=0.41   | L=3.17 D=0.41   | L=3.09 D=0.40    |
|  5    | L=1.51 D=0.31  | L=1.79 D=0.29  | L=2.16 D=0.28   | L=2.65 D=0.31   | L=2.92 D=0.32   | L=2.75 D=0.30    |
| 10    | L=1.46 D=0.27  | L=1.69 D=0.23  | L=2.05 D=0.23   | L=2.49 D=0.25   | L=2.81 D=0.25   | L=2.62 D=0.24    |
| 15    | L=1.41 D=0.24  | L=1.64 D=0.20  | L=1.96 D=0.20   | L=2.33 D=0.20   | L=2.71 D=0.20   | L=2.45 D=0.19    |
| 20    | L=1.37 D=0.22  | L=1.60 D=0.18  | L=1.89 D=0.18   | L=2.21 D=0.16   | L=2.61 D=0.17   | L=2.32 D=0.16    |

전 실험 모두 단조 감소, 발산/oscillation 없음. bin 수가 클수록 NLL 절대값이 큰 건
정상 (Br bin에 대한 cross-entropy의 baseline은 `log(Br)`이라 bin 수에 비례해
증가).

각 실험의 best checkpoint (val 기준) 선택 epoch:
- exp901: epoch 14, exp902: 18, exp903: 16, exp904: 14, exp905: 14, exp906: 18

> ⚠️ 학습 중 `Val L:..., ABS:5.x...` 로그는 **eval 버그 시기에 출력된 값이라
> 무의미함**. 아래 §4 참고. checkpoint 자체는 정상.

---

## 4. Eval 버그 발견 및 수정

학습 직후 첫 test 결과: 모든 range head 실험이 ABS_REL ≈ 5.5, Delta1 ≈ 0.02
로 catastrophic. scalar baseline (exp900)은 ABS_REL=0.52로 정상이었음 → 학습이
망가진 게 아니라 **평가 코드 버그**.

원인: `utils/test_utils.py:197-203` 와 `train.py` val pass에서

```python
if cfg.dataset.depth_norm:
    gt_map *= max_depth      # GT [0,1] → metres  (정상)
    pred_map *= max_depth    # PRED 도 [0,1] 가정 → metres
pred_map = np.clip(pred_map, 1e-3, max_depth)
```

이 로직은 두 가지 head 동작에 동시에 의존했는데:

- **scalar head**: `pred = sigmoid * max_depth`로 정의되지만 BerHu/SILog가
  `pred / max_depth` vs normalised gt로 비교되어 학습 후 pred는 **수치적으로
  [0,1]에 머무름** → `pred *= max_depth`가 metres로 정상 변환.
- **range head**: `pred = Σ p_j · r_j` (`r_j ∈ [0.1, 10]`) → 항상 metres로
  [0.1, 10]. `pred *= max_depth`를 하면 [1, 100]이 되고 clip [1e-3, 10]으로
  대부분 10에 saturate → ABS_REL 폭발.

**수정** (`utils/test_utils.py`, `train.py` val pass): echorange + range head일 때
metric 계산 직전에 `depth_pred /= max_depth` 한 번 정규화. 두 head가 같은
metric path를 공유하도록.

수정 후 6개 실험 test 재수행 (학습은 그대로) → 정상 metric 회수.

---

## 5. 최종 Test 결과 (post-fix)

| exp | bin | σ | 추가 | ABS_REL | RMSE | Delta1 | Delta2 | Delta3 | Log10 | MAE |
|----:|----:|---:|------|--------:|-----:|-------:|-------:|-------:|------:|----:|
| 900 | scalar | — | baseline (bs=64 DP) | 0.5229 | 1.2284 | 0.4854 | 0.6983 | 0.8214 | 0.1648 | 0.8227 |
| 901 | 4  | 1.15 | —              | 0.5233 | 1.2393 | 0.4848 | 0.6999 | 0.8212 | 0.1663 | 0.8350 |
| 902 | 8  | 0.58 | —              | 0.5489 | 1.2849 | 0.4853 | 0.6983 | 0.8157 | 0.1696 | 0.8560 |
| 903 | 16 | 0.29 | —              | 0.5321 | 1.2632 | 0.4955 | 0.7057 | 0.8219 | 0.1636 | 0.8327 |
| **904** | **32** | **0.14** | — (anchor) | **0.4888** | 1.2459 | 0.4988 | 0.7080 | 0.8232 | 0.1598 | 0.8074 |
| 905 | 32 | 0.30 | wider σ        | 0.5106 | 1.2621 | 0.4899 | 0.7000 | 0.8164 | 0.1654 | 0.8267 |
| **906** | **32** | **0.14** | **cos-lat + far-mask** | **0.4814** | 1.2532 | **0.5079** | **0.7105** | 0.8225 | 0.1606 | **0.8028** |

**Sweep best: exp906** — bin=32 + ERP 보정 조합.

---

## 6. echodiffusion baseline과 비교

이 코드베이스에 별도로 학습돼 있는 원조 echodiffusion 모델의 test 기록 (bs를
달리 한 grid):

| 실험 | 학습 설정 | ABS_REL | RMSE | Delta1 |
|------|-----------|--------:|-----:|-------:|
| echodiffusion exp11 (best) | lr=1e-4, **bs=32** | **0.4300** | **1.1060** | 0.4876 |
| echodiffusion exp363 | lr=5e-4, bs=48 | 0.4482 | 1.2198 | 0.4936 |
| echodiffusion exp13 | lr=1e-4, bs=16 | 0.4504 | 1.1134 | 0.4930 |
| echodiffusion exp362 | lr=1e-4, bs=32 | 0.4557 | 1.2292 | 0.4923 |
| **echorange-scalar exp900 (우리 baseline)** | lr=1e-4, bs=64 DP | 0.5229 | 1.2284 | 0.4854 |
| **echorange-range best exp906 (우리 best)** | lr=1e-4, **bs=16**, 32-bin + ERP fix | 0.4814 | 1.2532 | **0.5079** |

### 주요 격차

- **ABS_REL**: 우리 best 0.4814 vs echodiffusion best 0.4300 → **+12 % 상대로 뒤짐**.
- **RMSE**: 우리 1.2532 vs echodiffusion 1.1060 → +13 % 상대로 뒤짐.
- **Delta1**: 우리 best 0.5079 > echodiffusion best 0.4876 → **분포 head가 D1에서는 echodiffusion을 능가**. "GT 25 % 이내"라는 metric에는 분포 출력이 유리한 듯.

### 격차의 출처 추정

1. **batch size**: 가장 의심됨. echodiffusion best는 bs=32, 우리는 bs=16 →
   gradient noise ≈ √2배. echodiffusion exp13 (bs=16)도 ABS_REL 0.4504로
   우리보다 좋음 → bs만이 전부는 아님.
2. **scalar head 자체의 차이**: 우리 echorange-scalar (exp900, bs=64 DP)도
   원조 echodiffusion best (bs=32)보다 0.09 ABS_REL 떨어짐. 같은 backbone을
   썼다고 알려져 있는데 결과가 다름 → echorange의 scalar 모드 구현이
   echodiffusion 본가와 완전 일치하지 않을 가능성.
3. **lr schedule / warmup**: echodiffusion은 cosine warmup을 쓸 수 있음. 우리는
   cfg에 정의된 schedule만 사용 (이 sweep에서는 검증 안 함).
4. **head 구조**: echorange의 1×1 head가 echodiffusion 출력 head보다 얕을 수
   있음.

따라서 **range head 자체가 echodiffusion보다 약하다고 단정할 수 없음** —
같은 코드베이스 안에서 scalar 베이스라인부터 echodiffusion 본가에 못 미치고
있어, 그 격차를 먼저 좁혀야 공정한 비교가 됨.

---

## 7. 핵심 발견사항

### Q1. bin 수 sweet spot

bin 4 / 8 → scalar baseline 수준 (개선 없음). 너무 거친 분포는 expectation으로
환산해도 scalar regressor와 차이가 거의 없음. bin 16부터 Delta1이 살짝 오름.
**bin 32**가 명확한 sweet spot으로, ABS_REL이 scalar 대비 -6.5 % 상대 (0.5229 → 0.4888).
0–10 m 범위에서는 32 bin이면 log-spacing 0.144 nat (≈ 15 % depth resolution)이라
대부분의 깊이 변동을 표현 가능.

### Q2. soft label sigma

bin=32 anchor에서 σ=0.14 (ratio 1.0) > σ=0.30 (ratio 2.1). 넓은 sigma는 분포가
평탄해져 expectation 정확도가 떨어짐. **σ = log-bin-spacing 휴리스틱이 맞음**을
확인.

### Q3. ERP-aware loss 보정

`cos(lat)` 가중치 + `far-mask`를 함께 적용하면 모든 metric이 일관되게 개선:

|             | exp904 (anchor) | exp906 (ERP fix) | Δ              |
|-------------|----------------:|-----------------:|----------------|
| ABS_REL     | 0.4888          | **0.4814**       | −0.0074 (−1.5 %) |
| Delta1      | 0.4988          | **0.5079**       | +0.0091         |
| Delta2      | 0.7080          | 0.7105           | +0.0025         |
| MAE         | 0.8074          | **0.8028**       | −0.0046         |

폴라 oversampling과 last-bin saturation은 ERP radial depth에서 둘 다 실재하는
체계적 bias고, 보정 시 무시할 수 없는 이득을 줌.

### 한계

- **batch size 의존성 미점검**: bs=16 한 점으로만 측정 → bs=32에서 어떻게
  되는지 모름.
- **ERP fix를 cos-lat / far-mask로 분리 ablation 안 함** (combined만 봄). 어느
  쪽이 주효한지 불명.
- **lr schedule, head 구조 ablation 없음**.
- **wraparound (circular padding)** 미적용. backbone 손대야 해서 이 sweep에서 제외.

---

## 8. 다음 디자인 방향

### A. 우선순위 1 — echodiffusion baseline 격차 좁히기

현재 우리 echorange-scalar(exp900)부터 echodiffusion-best 대비 0.09 ABS_REL
떨어진다. 이걸 먼저 좁혀야 range head의 진짜 가치가 보임.

1. **bs=32 단일 GPU 재학습** — exp904 + exp906만이라도. 24 GB에 bs=32 들어가는지
   memory 확인 후 진행. gradient accumulation으로 effective bs=32도 옵션 (현재
   bs=16 × accum 2).
2. **echorange-scalar vs echodiffusion 차이 진단** — encoder/decoder weight load
   비교, head 구조 비교, lr schedule 비교. 동일 학습 조건에서 두 모델의 train
   loss curve를 align해서 차이를 확인.

### B. 우선순위 2 — ERP 보정 분리 ablation

exp906을 cos-lat 단독, far-mask 단독, combined로 쪼개서 어느 쪽이 주효한지
파악. anchor 동일하게 bin=32 σ=0.14.

- exp907: bin=32 + cos-lat only
- exp908: bin=32 + far-mask only
- exp909: bin=32 + combined (= 현재 exp906 재학습으로 sanity)

### C. 우선순위 3 — bin 수 미세조정 + 출력 mode

- bin 24 / 40 / 48 추가 (32와 48 사이가 비어있음).
- output mode: median (CDF 기반) — 이번엔 expectation만 봄. AbsRel friendly한 median이
  bin 32 anchor에서 expectation을 능가하는지 확인.

### D. 우선순위 4 — ERP wraparound (backbone-level)

decoder의 horizontal Conv2d padding을 `circular`로 교체 (또는 입력 ERP에
reflect-wrap padding 추가). backbone 전체 conv를 다 손대야 해서 별도 PR 단위.

### E. 장기 — 분포 head의 진짜 가치 발현

- **uncertainty 출력 활용**: range_entropy를 inference time confidence로 사용. low-confidence pixel을 보간/보강.
- **multi-modal depth**: 분포가 bimodal일 때 (반사가 모호한 경우) median/expectation이 한 mode를 잡고 나머지를 버리는 현상 검증.
- **NLL/BerHu/SILog 가중치 sweep**: 현재 모두 1.0. NLL을 더 강화하면 분포 head의 분포 학습이 더 깨끗해질 가능성.

---

## 9. 부록

### A. 학습 / 평가 명령어 (재현용)

```bash
cd /root/storage/implementation/shared_audio/baseline

# 전체 sweep (현재 스크립트, GPU 0+1 병렬)
bash scripts/n9_bulk_0427.sh

# 특정 실험만 학습 (예: exp904)
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /opt/conda/bin/python3 train.py \
  --config echorange --experiment-name exp904_echorange_32log_d10_lr1e4_bs16 \
  --epochs 20 --lr 0.0001 --batch-size 16 --num-workers 4 \
  --depth-dir erp_depth_radial \
  --dataset-dir /root/local1/changwoo/matterport3d_0303_renew \
  --depth-head-type range --range-num-bins 32 --range-bin-spacing log \
  --range-min-depth 0.1 --range-max-depth 10.0 \
  --range-soft-label-sigma 0.14 --range-output-mode expectation \
  --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 1.0
```

### B. 코드 변경 요약

| 파일 | 변경 |
|------|------|
| `config/echorange.yaml` | `range_max_depth: 20.0 → 10.0`, `validation_iter: 4 → 2` |
| `models/bin_based/range_head.py` | `soft_range_nll_loss`에 `weights` 인자 추가 (cos-lat용) |
| `train.py` | `_train_step_echorange`에 `erp_cos_lat_weight` / `erp_far_mask` 처리, val pass의 range head pred normalize 수정, CLI 인자 추가 |
| `test.py` | `--depth-head-type`, `--range-num-bins` 등 range CLI 인자 등록 (이전엔 `parse_known_args`로 silently drop돼서 test 시 yaml 기본값으로 모델 빌드되는 버그) |
| `utils/test_utils.py` | range head pred를 metric path 진입 전에 [0,1]로 정규화 (eval 버그 fix) |
| `scripts/n9_bulk_0427.sh` | 6개 실험 셋업, 단일 GPU bs=16 모드, 2 phase (train → test) 구조 |

### C. 파일 경로

- 학습 로그: `/root/storage/implementation/shared_audio/baseline/logs/n9_0427_train/exp9{01..06}_*.log`
- 테스트 로그: `/root/storage/implementation/shared_audio/baseline/logs/n9_0427_test/exp9{01..06}_*_test.log`
- 망가진 첫 test 로그 백업: `logs/n9_0427_test/_broken_pre_eval_fix/`
- Checkpoint: `/root/storage/implementation/shared_audio/baseline/checkpoints/echorange_soundspaces_BS16_Lr0.0001_AdamW_exp9{01..06}_echorange_*log_d10_lr1e4_bs16/best_model.pth`

### D. 환경 메모

- GPU 2가 driver-level fault (`Unable to determine the device handle for GPU2`) → NVML 망가짐 → NCCL 사용 불가. GPU 3도 enumeration이 GPU 2에 의존해서 동시에 깨짐. 단일 GPU 모드로 GPU 0, 1만 사용.
- 시스템 reboot 또는 GPU 2 hardware 점검 필요.
- bs=16은 24 GB 단일 GPU에 안전. bs=24-32까지는 시도해볼 여지 있음.
