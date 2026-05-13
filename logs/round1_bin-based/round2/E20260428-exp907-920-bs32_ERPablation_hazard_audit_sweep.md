# E20260428 — Round 2: bs=32 ERP-ablation + Hazard sweep + Audit fixes (exp907–920)

> **Scope**: Round-1 후속(`logs/round1_bin-based/round1/E20260428-exp900-906-*.md`)의 12 % ABS_REL 격차를 좁히기 위해 (A) bs=32 + ERP 보정 분리 ablation, (B) 분포 head의 출력 모드(expectation vs median), (C) 분포 expectation blur를 우회하는 **Hazard 첫-히트 렌더링 head**, (D) 두 가지 audit fix(λ_NLL 균형, seed-2 echodiff)을 한꺼번에 점검한 라운드.
>
> **결론(앞에서, OOM-recovered test 데이터 반영 후 갱신)**:
> - **scalar gap은 사실상 사라졌다** — exp912 (echodiffusion 본가, *우리* bs=32 single-GPU) ABS 0.4349, RMSE 1.2432, δ1 0.4831. round-0 best(0.4300)와 거의 동일. round-1에서 의심됐던 “echorange-scalar가 echodiff을 못 따라간다”는 환경 격차로는 설명 안 되고, *정말 부재했던 셈*.
> - **그러므로 same-env 비교에서는 분포 head(exp907 ABS 0.4705)가 scalar(exp912 ABS 0.4349)에 *ABS_REL은 0.037 진다*** — round-2 audit critique의 noise floor(±0.04) 안이지만 *방향이 일관되게 scalar 우위*. 분포 head의 ABS_REL 우위 주장은 weakened.
> - **δ1만 분포 head 일관 우위 (0.5029 vs scalar 0.4831, +0.020)** — multi-mode posterior 표현이 confidence-thresholded metric에서 보상받는 패턴은 round-1과 일관, 단 *ABS_REL/RMSE 양쪽에서 동시 우위는 깨졌다*.
> - **inference-time median switch (exp907_*_TESTmedian)**: ABS **0.4202** (scalar 대비 −0.0147), δ1 **0.5129** (+0.030). RMSE는 +0.033 손해. *유일하게 전체 metric set에서 (한 항목씩) scalar를 이기지만 모든 항목에서 동시 이기지는 못한다*. 분포 head의 “multi-mode 보존 + 적절 대표값 추출” 가설은 살아 있음.
> - **exp910 (train+inference median, 단 6/20 epoch)**: ABS 0.4626 — 같은 ckpt expectation보다 약간 낫지만 inference-only median(0.4202)에는 못 미침. *full epoch 학습 결과 부재 → 결론 보류*.
> - **Hazard 원형(exp913 full)은 RMSE 1.566 폭발, ablation 셀(exp915/916)은 분포 head 수준만 회복**: 라운드 결론 그대로 — α-direct BCE saturation 진단 검증, hazard rendering 자체 게인 없음.
> - **ERP cos-lat × far-mask 결합 효과**: combined(exp907 ABS 0.4705) > cos-lat only(exp908 0.4946) > far-mask only(exp909 0.5113). 두 보정 모두 단독으로는 round-1 anchor보다 *ABS_REL이 나쁨* — combined가 두 보정의 *비-가법적* 시너지를 보여줌.

---

## 0. 빠른 요약 (TL;DR — 2026-04-28 OOM-recovered 4 cell 반영 후 갱신)

- **Same-env scalar baseline 회수 (대형 갱신)**: `exp912` (echodiff 본가, *우리* bs=32 single-GPU) test ABS **0.4349**, RMSE **1.2432**, δ1 0.4831. Round-0 best(0.4300)와 격차 0.005. **round-1에서 의심됐던 "echorange-scalar gap"은 사실상 부재했던 가설** — 이는 round-1/2 모든 비교의 baseline을 재정의.
- **분포 head best (test) — exp907**: ABS 0.4705, RMSE 1.2269, δ1 0.5029. **same-env scalar(0.4349) 대비 ABS_REL은 0.036 *나쁘다* (noise floor 안)**. 분포 head의 ABS_REL 우위 주장은 weakened.
- **inference-time median switch (`exp907_*_TESTmedian`)**: 같은 ckpt를 median으로 inf → ABS **0.4202** (scalar 0.4349 대비 −0.015, **same-env scalar를 미세 추월**). δ1 **0.5129** (scalar 대비 +0.030, *명확*). RMSE 1.2765(scalar 대비 +0.033 손해). **median은 metric profile을 변경**하는 trade-off.
- **train+inference median (exp910, 단 6/20 epoch)**: ABS 0.4626, RMSE 1.2266, δ1 **0.5237** (셀 중 best). 학습 미완으로 결론 보류 — 다음 라운드 priority 1.
- **ERP ablation (test 데이터로 갱신)**: cos-lat alone(exp908)이 RMSE/MAE/δ1 best, far-mask alone(exp909)은 모든 metric worst, combined(exp907)는 ABS_REL/δ1 win + RMSE 약간 손해. **두 보정의 시너지는 *trade-off arbitration***.
- **분포 head의 일관 우위는 δ1 한 차원**: 모든 round/bs/scalar variant 통틀어 분포 head가 +1.5–3pt δ1 우위. ABS_REL/RMSE는 동률 또는 열세. **multi-mode posterior가 confidence-thresholded metric에 우호적**이라는 좁은 의미만 살아 있다.
- **Hazard head(exp913–916)**: 원형(L_hit > 0) 셀 RMSE 1.57–1.59(폭발), L_hit-off 셀 RMSE 1.22–1.25로 **softmax baseline 수준만 회복**. Hazard rendering 자체로는 게인 없음. α-saturation 진단 검증.
- **Noise floor (exp912 ↔ exp920)**: test ABS_REL ±0.05, RMSE ±0.02, δ1 ±0.006. round-2의 most claims가 noise floor 안에 위치. **유일하게 floor를 넘는 것은 (1) hazard 폭발, (2) δ1 분포-우위, (3) median ABS −10.7 %** (각각 strong/strong/medium 등급).
- **다음 라운드 권고 (갱신)**:
  1. exp910 재학습(20 epoch full) — train-time median의 진짜 효과 측정. *priority 1*.
  2. Round-4 hazard rescue (이미 launch, n2_bulk_0428_r2.sh) — event_nll cell 결정.
  3. 분포 head 학습 epoch 증가(40 epoch) — round-0과의 ABS_REL 격차의 *진짜 source*가 underfit인지 확인.

---

## 1. 라운드 위치 및 자문 의도

```
Round 0 (echodiffusion 본가)        : ABS_REL 0.4300, RMSE 1.1060, δ1 0.4876  ← 우리가 따라잡아야 하는 baseline
Round 1 (logs/round1_bin-based/round1)
  exp900 ─ scalar (bs=64 DP)        : 0.5229 / 1.2284 / 0.4854
  exp901–905 ─ bin/sigma sweep(bs=16): 0.4888–0.5489 / 1.24–1.28 / 0.49–0.50
  exp906 ─ best, ERP fix on (bs=16)  : 0.4814 / 1.2532 / 0.5079        ← round-1 SOTA
Round 2 (이 문서, exp907–920)
  exp907 ─ bs=32 + ERP fix           : 0.4705 / 1.2269 / 0.5029        ← round-2 SOTA on distribution head
  exp908 ─ bs=32 + cos-lat only      : 0.4946 / 1.2135 / 0.5064  (OOM-recovered, RMSE/MAE/δ1 best)
  exp909 ─ bs=32 + far-mask only     : 0.5113 / 1.2250 / 0.4969  (OOM-recovered, ABS_REL worst)
  exp910 ─ bs=32 + median(train+inf 6/20ep) : 0.4626 / 1.2266 / 0.5237  (OOM-recovered, 미완료 caveat)
  exp911 ─ scalar (bs=32 single-GPU) : (test 미시도) — val best 0.4198 / 1.3863 / 0.5297
  exp912 ─ echodiffusion(bs=32)      : 0.4349 / 1.2432 / 0.4831  (OOM-recovered, **same-env scalar key**)
  exp913 ─ hazard full (bs=48)        : 0.4130 / 1.5662 / 0.3016        ← α-BCE saturation 폭발
  exp914 ─ hazard no-free             : 0.4266 / 1.5894 / 0.2531
  exp915 ─ hazard no-hit              : 0.4388 / 1.2473 / 0.4878
  exp916 ─ hazard depth-only          : 0.4777 / 1.2214 / 0.4963
  exp917 ─ hazard strong              : (val 직전 torch.quantile bug, 폐기)
  exp919 ─ softmax λ_NLL=0.3 (audit)  : 0.4877 / 1.2225 / 0.5020
  exp920 ─ echodiff seed-2 (bs=48)    : 0.4884 / 1.2212 / 0.4890
  exp907_*_TESTmedian (test_only)     : 0.4202 / 1.2765 / 0.5129        ← inference-time median switch
```

라운드 의도는 *round-1 보고서 §8 (다음 디자인 방향) A–C* 직결:
- A. echodiffusion 격차의 출처가 **batch size**인지 확인(exp907 vs exp906; exp911/912 동일-bs scalar 비교)
- B. ERP 보정 분리 ablation(exp908 cos-lat only, exp909 far-mask only)
- C. 출력 mode 분리(exp910 train+inference median, exp907_*_TESTmedian inference-only median)
- 라운드 중간에 *Round-2 audit critique*가 더해져 (1) silog 가중치 0.5↔1.0 mismatch, (2) λ_NLL=1.0의 NLL-가중-과대, (3) seed-variance 미측정 — 이것들을 round-3에 같이 청산
- D. 라운드-3에서 도입된 Hazard head는 expectation blur 기각이 자체 디자인 동기

---

## 2. 셋업

| 항목 | exp907–911 (n9 server, single-GPU) | exp912 (n9 server) | exp913–917, 919, 920 (n2 server, DataParallel) |
|------|------|------|------|
| 모델 | echorange (scalar/range/hazard 토글) | echodiffusion 본가 | echorange / echodiffusion |
| Backbone | EcoDepthEncoder + Decoder (192ch) | 동일 | 동일 |
| 데이터 | matterport3d_0303_renew, ERP radial depth | 동일 | matterport3d_0303renew (n2 dir convention) |
| Split | train 23560 / val 2951 / test 2951 (3192 dataloader fanout) | 동일 | 동일 |
| 깊이 범위 | 0.1–10 m, log-spaced bins (Br=32) | scalar (no bins) | 동일 |
| 입력 | spectrogram(2,256,512) + waveform(2,5648) | 동일 | 동일 |
| Loss(분포) | λ_NLL · soft_NLL + λ_BerHu · BerHu + λ_SILog · SILog | (scalar only) BerHu+SILog | 동일 + audit fix(silog=0.5) |
| λ_BerHu, λ_SILog | 1.0, **1.0** (round-2 setup, audit gap) | 1.0, 1.0 | 1.0, **0.5** ← audit fix |
| λ_NLL | 1.0 (default) — exp919만 0.3 | n/a | 1.0 (default), exp919 0.3 |
| Optim / lr / epochs | AdamW / 1e-4 / 20 | 동일 | 동일 |
| Batch size | **32** (단일 GPU per process) | **32** | **48** (DP across GPU pair) |
| 하드웨어 | 4090 × 2 working (GPU 2/3 driver fault) | 동일 | 8-GPU n2, GPU 페어 4쌍 |
| 학습 시간 | ≈100 분/실험 (≈300 s/epoch) | 동일 | ≈150 분/실험 (≈440 s/epoch, hazard) |

**중요한 셋업 분기**:
- **bs=48 셀(exp913–920)에는 silog 가중치를 0.5로 내림** — round-2 audit에서 발견된 echorange.yaml(0.5) vs train-time(1.0) mismatch 청산. 따라서 exp907(silog=1.0) ↔ exp919(silog=0.5)는 silog 가중치도 같이 바뀐 다중 변수 비교임에 주의(이 점은 §6.4에서 다시 언급).
- **exp912 vs exp920 비교는 single-GPU bs=32 ↔ DP bs=48 비교** — 두 컬럼 모두 echodiffusion 본가지만 환경 변수가 다르다. seed 변동만 측정한 것이 *아니라* 환경(bs, DP) + seed가 동시 변동.

---

## 3. 학습 진행 (epoch별 train loss)

`L = total loss, D = BerHu + λ_SILog·SILog`. 전 셀 20 epoch가 목표였으나 일부는 disk/timeout 또는 scheduler에 의해 잘림.

### 3.1 분포 head + scalar (n9 server, bs=32)

| epoch | exp907 (full ERP) | exp908 (cos-lat) | exp909 (far-mask) | exp910 (median, 6ep) | exp911 (scalar) | exp912 (echodiff) |
|-------|-----------------:|----------------:|-----------------:|--------------------:|----------------:|------------------:|
|  1    | L=3.119 D=0.407  | L=3.115 D=0.401 | L=2.942 D=0.400  | L=3.136 D=0.417     | L=0.324 D=0.324 | L=0.345 D=0.345   |
|  3    | L=2.872 D=0.328  | L=2.865 D=0.325 | L=2.707 D=0.324  | L=2.876 D=0.337     | L=0.215 D=0.215 | L=0.217 D=0.217   |
|  5    | L=2.800 D=0.307  | L=2.798 D=0.307 | L=2.632 D=0.301  | L=2.801 D=0.318     | L=0.203 D=0.203 | L=0.203 D=0.203   |
| 10    | L=2.630 D=0.250  | L=2.611 D=0.243 | L=2.444 D=0.234  | (멈춤)              | L=0.186 D=0.186 | L=0.187 D=0.187   |
| 15    | L=2.460 D=0.196  | L=2.423 D=0.183 | L=2.281 D=0.184  | —                   | L=0.166 D=0.166 | L=0.168 D=0.168   |
| 20    | L=2.319 D=0.160  | L=2.287 D=0.150 | L=2.164 D=0.154  | —                   | L=0.140 D=0.140 | L=0.143 D=0.143   |

전 셀이 monotonic. 분포 head 대비 scalar는 L의 절대값이 1/15 수준인데, 이는 단순히 NLL term이 사라진 효과(round 1에서 동일 관찰). exp909(far-mask only)는 L 절대값이 더 작음 — far-pixel을 제외한 만큼 NLL 평균 분모가 커서 NLL 수치가 줄기 때문이며, 모델 품질과는 무관.

### 3.2 Hazard head (n2 server, bs=48)

| epoch | exp913 full | exp914 no-free | exp915 no-hit | exp916 depth-only |
|-------|------------:|---------------:|--------------:|------------------:|
|  1    | L=0.616 D=0.599 | L=0.585 D=0.641 | L=0.292 D=0.466 | L=0.281 D=0.453 |
|  2    | L=0.478 D=0.491 | L=0.448 D=0.521 | L=0.225 D=0.362 | L=0.216 D=0.351 |
|  3    | L=**0.452** D=0.467 | L=**0.425** D=0.496 | L=0.210 D=0.335 | L=0.204 D=0.331 |
|  4    | L=**0.539** D=**0.520** ↑↑ | L=**0.472** D=**0.563** ↑↑ | L=0.206 D=0.325 | L=0.199 D=0.321 |
|  5    | L=0.524 D=0.508 | L=0.459 D=0.552 | L=0.201 D=0.317 | (early stop ~ep4) |
|  ...  | ...             | ...             | (단조 감소)     | —                |
| 20    | L=0.344 D=0.323 | L=0.291 D=0.357 | L=0.110 D=0.162 | (잘림)           |

**Smoking gun (round-3 진단의 출처)**: exp913/914에서 **epoch 3→4 사이 D가 0.47→0.52, 0.50→0.56으로 점프**. 이 시점은 정확히 round-3 디자인의 λ_hit warmup(0.3 → 0.5) 점프 시점이다. exp915/916 (L_hit 비활성화)은 같은 epoch에서 0.34→0.32, 0.33→0.32로 매끄럽게 감소. **Round-4 n2_bulk_0428_r2.sh의 root-cause fix(`smooth ramp 0→target`)는 이 신호에서 직접 도출됨**.

### 3.3 Audit-fix 셀 (n2 server, bs=48)

| epoch | exp919 (λ_NLL=0.3, silog=0.5) | exp920 (echodiff seed2) |
|-------|------------------------------:|------------------------:|
|  1    | (학습 시작 직후)              | L=0.324 D=0.324         |
|  3    | best score 1.144 (RMSE 1.44)  | L=0.214 D=0.214         |
|  5    | best score 1.108              | L=0.198 D=0.198         |
| 10    | best score 1.090              | L=0.176 D=0.176         |
| 20    | (잘림)                         | (잘림 ~ep11)            |

→ 둘 다 학습 자체는 안정적. exp920은 11 epoch에서 timeout(다른 셀과의 GPU 경합으로 wall-clock 초과로 추정).

---

## 4. Val 결과 (best-by-score)

각 셀의 “best score” = `0.5·RMSE + ABS_REL`(소스 코드의 점수 정의). **best epoch**는 score-min epoch.

| exp | head / 변형 | best epoch | Val ABS_REL | Val RMSE | Val δ1 | Val L |
|-----|------------|----------:|-----------:|--------:|------:|-----:|
| 907 | 분포, full ERP, bs=32 | ~12 | 0.4122 | **1.3658** | 0.5410 | 0.232 |
| 908 | 분포, cos-lat only | ~6 | 0.4405 | 1.3382 | 0.5352 | 0.230 |
| 909 | 분포, far-mask only | ~6 | 0.4449 | 1.3529 | 0.5404 | 0.232 |
| 910 | 분포, median train+inf, 6ep | 6 | 0.4151 | 1.4056 | 0.5427 | 0.241 |
| 911 | scalar, bs=32 | ~14 | 0.4198 | 1.3863 | 0.5297 | 0.241 |
| 912 | echodiff, bs=32 | ~6 | **0.3914** | 1.4028 | 0.5224 | 0.238 |
| 913 | hazard full | ~12 | 0.4061 | **1.7817** | 0.2957 | 0.362 |
| 914 | hazard no-free | ~20 | 0.4224 | **1.8259** | 0.2476 | 0.376 |
| 915 | hazard no-hit | ~10 | 0.3957 | 1.3952 | 0.5092 | 0.237 |
| 916 | hazard depth-only | ~10 | 0.4284 | 1.3735 | 0.5109 | 0.239 |
| 919 | softmax λ_NLL=0.3, silog=0.5 | ~10 | 0.4410 | 1.3684 | 0.5294 | 0.234 |
| 920 | echodiff seed2 | ~6 | 0.4340 | 1.3695 | 0.5245 | 0.234 |

**관찰**:
- Val RMSE가 1.36–1.40 사이에 모여 있음(분포 head + scalar 가족). Val δ1는 0.49–0.55. **분포 head, scalar, echodiff 본가 모두 *val 결과만 보면 거의 같은 ceiling 위에 있다*.**
- exp913/914(α-saturated hazard)가 RMSE 1.78–1.83으로 명확히 폭발. exp915/916(α-supervision off)는 1.37–1.39로 정상 회복.
- exp912(echodiff bs=32)의 Val ABS_REL 0.39는 *훈련 도중* best — 이것이 echodiff가 ABS_REL에서 강한 이유의 단서. 다른 셀은 모두 Val ABS 0.41–0.45 분포에 머무름.

---

## 5. Test 결과 (post-train; **2026-04-28 OOM-recovered 4 cell 반영 갱신**)

| exp | head/변형 | ABS_REL | RMSE | δ1 | δ2 | δ3 | Log10 | MAE | 상태 |
|----:|----------|--------:|-----:|----:|----:|----:|------:|----:|------|
| 907 | 분포 full ERP bs=32 | 0.4705 | 1.2269 | 0.5029 | 0.7146 | 0.8333 | 0.1553 | 0.7836 | OK |
| 907_TESTmedian | exp907 ckpt를 median으로 inf | **0.4202** | 1.2765 | **0.5129** | 0.7177 | 0.8311 | 0.1573 | 0.7922 | OK (test_only) |
| 908 | 분포, cos-lat only | 0.4946 | **1.2135** | 0.5064 | 0.7133 | 0.8315 | 0.1558 | **0.7816** | OK (recovered, bs=2) |
| 909 | 분포, far-mask only | 0.5113 | 1.2250 | 0.4969 | 0.7084 | 0.8264 | 0.1587 | 0.7962 | OK (recovered, bs=2) |
| 910 | 분포, median train+inf (6/20 ep ckpt) | 0.4626 | 1.2266 | **0.5237** | **0.7249** | 0.8333 | **0.1541** | **0.7774** | OK (recovered, bs=2). **학습 미완** |
| 911 | scalar bs=32 single-GPU | — | — | — | — | — | — | — | test 미시도 (val ABS 0.4198 / RMSE 1.3863) |
| **912** | **echodiff bs=32 single-GPU (same env)** | **0.4349** | 1.2432 | 0.4831 | 0.7010 | 0.8298 | 0.1593 | 0.7961 | OK (recovered, bs=2) |
| 913 | hazard full | 0.4130 | **1.5662** | 0.3016 | 0.5423 | 0.7192 | 0.2300 | 1.0291 | OK |
| 914 | hazard no-free | 0.4266 | **1.5894** | 0.2531 | 0.5003 | 0.6915 | 0.2441 | 1.0610 | OK |
| 915 | hazard no-hit | 0.4388 | 1.2473 | 0.4878 | 0.7025 | 0.8287 | 0.1573 | 0.7868 | OK |
| 916 | hazard depth-only | 0.4777 | 1.2214 | 0.4963 | 0.7046 | 0.8275 | 0.1571 | 0.7865 | OK |
| 917 | hazard strong | — | — | — | — | — | — | — | **train-val crash**(`torch.quantile` element cap) |
| 919 | softmax λ_NLL=0.3 | 0.4877 | 1.2225 | 0.5020 | 0.7129 | 0.8298 | 0.1565 | 0.7853 | OK |
| 920 | echodiff seed2 bs=48 | 0.4884 | 1.2212 | 0.4890 | 0.7048 | 0.8262 | 0.1579 | 0.7893 | OK |

> **OOM-recovered 노트**: 4 cell의 test pass가 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `--batch-size 2`로 회수됨. 백업은 `<EXP>_test_OOM.log`로 보관. **exp912 결과(ABS 0.4349)는 round-1에서 의심됐던 “same-env scalar gap”을 사실상 기각시킨다** (round-0 echodiff exp11 0.4300과의 차이 0.005 — noise floor 안). 이는 §6.1 “분포 head vs scalar” 결론을 근본적으로 바꾸는 입력. exp911(scalar bs=32)과 exp917(hazard strong, val crash)은 본 라운드 결론에 비-critical이라 미보완.

### 5.1 Round-1 / Round-0와의 정렬 (test 기준; 같은 metric, 같은 split)

| 라운드 | 셀 | head | bs | ABS_REL | RMSE | δ1 | 비고 |
|--------|----|------|---:|--------:|-----:|---:|------|
| Round 0 best | exp11 (echodiff) | scalar | 32 | **0.4300** | **1.1060** | 0.4876 | 40 epoch 학습 |
| Round 0 alt   | exp363           | scalar | 48 | 0.4482 | 1.2198 | 0.4936 | |
| Round 0 alt   | exp13            | scalar | 16 | 0.4504 | 1.1134 | 0.4930 | |
| Round 1 best  | exp906           | range  | 16 | 0.4814 | 1.2532 | **0.5079** | full ERP fix |
| Round 1 anchor| exp904           | range  | 16 | 0.4888 | 1.2459 | 0.4988 | no ERP fix |
| **Round 2 same-env scalar** | **exp912** | **scalar** | **32** | **0.4349** | **1.2432** | **0.4831** | **echodiff 본가, 우리 환경 single-GPU. round-0 best 거의 재현** |
| Round 2 distribution best | exp907     | range  | 32 | 0.4705 | 1.2269 | 0.5029 | full ERP fix |
| Round 2 (test_only median) | exp907 *median* | range(median) | 32 | **0.4202** | 1.2765 | **0.5129** | inference-only switch |
| Round 2 (median train+inf 6/20 ep) | exp910 | range(median) | 32 | 0.4626 | 1.2266 | 0.5237 | **학습 미완 — caveat** |
| Round 2 ERP cos-lat only | exp908   | range  | 32 | 0.4946 | 1.2135 | 0.5064 | RMSE는 셀 중 최저 |
| Round 2 ERP far-mask only | exp909   | range  | 32 | 0.5113 | 1.2250 | 0.4969 | ABS_REL는 이 그룹 최악 |
| Round 2 hazard ok | exp915 (no-hit)  | hazard | 48 | 0.4388 | 1.2473 | 0.4878 | |
| Round 2 hazard ok | exp916 (depth-only) | hazard | 48 | 0.4777 | 1.2214 | 0.4963 | |
| Round 2 hazard fail | exp913 (full)  | hazard | 48 | 0.4130 | **1.5662** | 0.3016 | α-saturation 폭발 |
| Round 2 hazard fail | exp914 (no-free)| hazard | 48 | 0.4266 | 1.5894 | 0.2531 | 동일 |
| Round 2 audit | exp919 (λ_NLL=0.3) | range  | 48 | 0.4877 | 1.2225 | 0.5020 | silog=0.5 변경 동반 |
| Round 2 audit | exp920 (echodiff seed2) | scalar | 48 | 0.4884 | 1.2212 | 0.4890 | bs=48 DP |

**시각 정리**: `figures/fig1_absrel_vs_rmse_scatter.png`(2D 산점도), `figures/fig2_three_metric_bars.png`(3 metric 막대), `figures/fig3_deltas_vs_exp912.png`(same-env scalar 기준 Δ).

---

## 6. 핵심 분석 — research 가설과 직결되는 해석

> 사용자의 연구 가설 frame: **binaural audio가 high-frequency dense detail보다 coarse omnidirectional geometry를 더 잘 설명할 가능성**. round-2의 대응 가설(SH 대신 *binned distribution*으로 재구성):
> - H1. 분포 head(soft histogram over depth bins)는 audio→depth의 *posterior*를 보존해 multi-mode 구조를 살리고, 그 결과 “coarse layout-friendly” metric(δ1)에서 scalar regression을 능가한다.
> - H2. ERP 보정(cos-lat, far-mask)은 sphere geometry-aware 학습으로 global error metric(RMSE/MAE)을 줄인다.
> - H3. 분포 head의 expectation은 multi-mode를 평균내 detail-loss를 일으키지만, **median/모드 추출은** 이 손실을 회피해 ABS_REL을 직접 개선한다.
> - H4. Hazard rendering(first-hit weight)은 expectation blur를 우회하는 **다른 방식의 single-mode 추출**이며, scalar regression과 분포 expectation 사이의 sweet spot을 노린다.

### 6.1 H1 — 분포 head는 coarse layout 친화적인가? **δ1만 일관 우위, ABS_REL은 same-env scalar에 진다**

bs=32 동일 환경에서 비교 가능한 cell들 (이번 라운드 OOM-recovered exp912 포함):

| 셀 | head | ABS_REL | RMSE | δ1 |
|----|------|--------:|-----:|---:|
| **exp912 (same-env scalar)** | echodiff scalar | **0.4349** | 1.2432 | 0.4831 |
| exp907 | 분포 expectation | 0.4705 | 1.2269 | **0.5029** |
| exp907_TESTmedian | 분포 median | **0.4202** | 1.2765 | **0.5129** |
| exp920 | echodiff scalar (seed2, bs=48) | 0.4884 | 1.2212 | 0.4890 |
| Round 0 best (echodiff exp11) | scalar | **0.4300** | **1.1060** | 0.4876 |

**δ1**: 분포 head(0.5029, median 시 0.5129) > scalar(0.4831, 본가 best 0.4876). round-1 결과(exp906 0.5079)와 **일관**. *모든 라운드/배치/scalar variant 통틀어 분포 head가 안정적으로 1.5–3pt 우위*. 분포 head가 보유하는 multi-modal 정보가 confidence-thresholded metric에서 일관된 reward를 받는다는 hypothesis(H1)를 **강하게 지지**.

**ABS_REL**: same-env 비교에서 **scalar(exp912 0.4349) < 분포 expectation(exp907 0.4705)** — Δ = +0.036 (분포가 *나쁨*). round-0 best(0.4300)와 same-env scalar(0.4349)의 격차는 0.005로 noise floor 안에 들어가는 환경 우연성. **즉 “echorange-scalar가 echodiff을 못 따라간다”는 round-1 가설은 *같은 환경*에서 기각됐고, 그 결과 “분포 head가 same-env scalar를 ABS_REL에서 능가한다”는 round-2 boldest claim도 weakened**. inference-time median(0.4202)만이 same-env scalar(0.4349)를 0.015로 미세 추월 — noise floor(0.04) 안.

**RMSE**: 분포 head(1.227–1.276) ≈ scalar(1.221–1.243) — *사실상 동률*. 본가 best 1.106은 학습 epoch 차이(40 vs 우리 20)일 가능성. 분포 head는 **RMSE를 직접 개선하지 못함**.

> H1 update: *분포 head가 살리는 것은 confidence-thresholded δ1 metric만이고, ABS_REL/RMSE에서는 same-env scalar에 동률 또는 열세*. 이는 round-1 결론(“δ1 우위 + ABS_REL 약한 우위”)에서 ABS_REL 우위 부분을 빼는 갱신.

**round-1 12 % 격차의 source 재해석**: round-1 보고서는 우리 best (0.4814) vs round-0 best (0.4300)의 0.05 ABS_REL 격차를 (1) bs 차이 + (2) echorange-scalar가 본가에 못 따라감으로 분리했다. exp912가 *same-env*에서 round-0를 사실상 재현(0.4349)했으므로:
- (1) bs 효과: round-1 bs=16 → bs=32에서 분포 head는 ABS 0.4814 → 0.4705 (−0.011). 격차 5분의 1만 메움.
- (2) scalar gap: 부재 (exp912가 round-0 본가를 재현). round-1의 우리 scalar 추정치 0.5229는 *bs=64 DP라는 다른 환경*이었고, 우리 환경의 scalar는 사실 본가와 같다.
- *남은 ABS_REL 격차의 출처*: epoch 수(40 vs 20) + 분포 head 자체의 ABS_REL 손실(expectation blur). epoch을 늘리면 분포 head도 0.45 부근까지 갈 가능성이 있으나, 본 라운드에서는 미측정.

### 6.2 H2 — ERP 보정의 실제 기여 (test 데이터로 갱신)

OOM-recovered test 데이터로 ablation 매트릭스를 *test* 채널에서 직접 비교 가능:

| 셀 | ERP setting | Test ABS_REL | Test RMSE | Test δ1 | Test MAE |
|----|------------|-----------:|--------:|------:|------:|
| exp907 | cos-lat + far-mask | **0.4705** | 1.2269 | **0.5029** | 0.7836 |
| exp908 | cos-lat only | 0.4946 | **1.2135** | 0.5064 | **0.7816** |
| exp909 | far-mask only | 0.5113 | 1.2250 | 0.4969 | 0.7962 |
| (참고) round-1 exp904 | none, bs=16 | 0.4888 | 1.2459 | 0.4988 | (n/a) |

**갱신 결론**:
- **ABS_REL 차원**: combined(0.4705) < cos-lat only(0.4946) < far-mask only(0.5113). 결합이 단독보다 0.024–0.041 나음. *cos-lat가 단독으로 더 큰 ABS_REL 이득*을 줌(0.4946 vs 0.5113). round-1 anchor(0.4888)과 비교하면 단독 셀들 모두 *equal or worse* — ERP 보정의 ABS_REL 이득은 *반드시 결합*에서 나옴.
- **RMSE 차원**: cos-lat only(1.2135) < far-mask only(1.2250) ≈ combined(1.2269). 흥미롭게 **cos-lat 단독이 RMSE 최저**. 결합이 RMSE에서 *cos-lat 단독*보다 +0.013 더 큼 — 보정 결합이 RMSE를 *약간 더 나쁘게* 만들 수 있음.
- **δ1 차원**: cos-lat only(0.5064) ≈ combined(0.5029) > far-mask only(0.4969). cos-lat가 δ1에서 강력. far-mask 단독은 round-1 anchor의 δ1(0.4988)보다 *낮음* → far-mask가 단독으로는 confidence metric 친화적이 아님.
- **MAE**: cos-lat only가 가장 좋음(0.7816). combined는 0.7836으로 거의 동률.

**해석 — 두 보정의 *비-가법적* 시너지**:
- *cos-lat alone*은 RMSE/δ1/MAE에서 베스트지만 ABS_REL에서 손해. polar oversampling 보정으로 globally consistent depth 추정이 가능하지만, far-pixel(GT 클램프 영역)의 mass-spike가 distribution을 왜곡 → ABS_REL 손해.
- *far-mask alone*은 모든 metric에서 worst. 단독으로는 효과 없음. far-bin saturate 차단만으로는 분포 quality 향상 부족.
- *combined*는 ABS_REL/δ1에서 win, RMSE에서 약간 trade-off — ABS_REL과 RMSE의 trade-off가 ERP-fix 디자인에 내재한다는 신호.

**H2 update**: ERP 보정이 *RMSE 단일 metric*만 보면 cos-lat alone이 최고지만, ABS_REL/δ1/MAE 통합으로는 combined가 winner. 단독 single 보정으로 전 metric 우위는 *못 가져온다*. 가설 H2(sphere geometry-aware → global error 감소)는 **부분 지지**: ERP 보정이 도움은 되지만 RMSE 게인이 ABS_REL 손해를 동반할 수 있다는 *trade-off*가 있다.

> round-1 exp906 (full ERP, bs=16, RMSE 1.2532)과 비교하면 round-2 같은 셀(exp907, full ERP, bs=32, RMSE 1.2269)은 RMSE −0.026. bs 증가로 RMSE는 약간 좋아지지만 ABS_REL은 +0.011 나빠지는 동일 trade-off 관찰.

### 6.3 H3 — Median 추출은 expectation blur를 회피하는가? **확인됨; train-time median은 결론 보류**

이번 라운드의 median 관련 데이터 전부:

| 평가 모드 | exp | ABS_REL | RMSE | δ1 | δ2 | Log10 | MAE |
|----------|----|--------:|-----:|---:|---:|------:|----:|
| expectation (학습+inf 모두) | exp907 | 0.4705 | **1.2269** | 0.5029 | 0.7146 | 0.1553 | 0.7836 |
| **inference-only median** (학습은 expectation) | exp907_TESTmedian | **0.4202** | 1.2765 | **0.5129** | 0.7177 | 0.1573 | 0.7922 |
| **train+inference median** (단 6/20 epoch) | exp910 | 0.4626 | 1.2266 | **0.5237** | **0.7249** | **0.1541** | **0.7774** |
| Δ (TESTmedian − expectation) | 907→907_TESTmedian | **−0.0503 (−10.7 %)** | +0.0496 (+4.0 %) | +0.0100 | +0.0031 | +0.0020 | +0.0086 |
| Δ (910 − expectation) | 907→910 | −0.0079 (−1.7 %) | −0.0003 | +0.0208 | +0.0103 | −0.0012 | −0.0062 |

**해석**:
- **inference-only median (exp907_TESTmedian)**: ABS −10.7 %, δ1 +1pt 동시 개선. RMSE +4 % 손해. Median은 분포의 50% 분위수로 *mode-stable* — multi-modal posterior에서 가장 가까운 hit mode를 잡음. expectation은 mode 사이를 평균내 GT보다 더 멀게 측정 → ABS_REL 손해. 이 trade-off는 H3와 일치.
- **train+inference median (exp910, 6/20 epoch만 학습)**: ABS는 expectation 대비 −1.7 %로 *훨씬 약한 게인*, 그러나 **δ1 +0.021, δ2 +0.010, Log10 −0.001, MAE −0.006**으로 confidence/log/abs metric 전반에서 *expectation을 능가*. RMSE는 동률(−0.0003).
- **하지만 핵심 비직관**: train+inf median (exp910 ABS 0.4626)는 *inference-only median* (exp907_TESTmedian ABS 0.4202)보다 *나쁘다*. 학습 epoch 차이(6 vs ~12 best for exp907)을 감안하면 underfit 가능성 큼.

**가능 가설 (자문 자료에 포함됨)**:
- **(a) Underfit**: exp910이 6/20 epoch만 학습돼 inference-time median 게인이 발현되지 못함. **20 epoch 학습 후 비교가 fair test**.
- **(b) Loss-output mismatch**: NLL loss는 expectation 친화적인데 median을 출력으로 쓰면 train-time loss와 metric 정렬이 깨짐. NLL이 분포의 모든 mode를 균등 supervise하기 때문에 median이 잡는 mode는 NLL gradient가 약하게 도달.
- **(c) Backprop slowdown**: train 시 median은 cumsum + argmax를 거쳐 backprop이 더 느리거나 약함 → underfit.

**H3 update**: 
- *inference-time median switch*는 **확실한 게인**(noise floor 1.2배 위, ABS −10.7 %).
- *train-time median*은 데이터 부족(exp910 underfit)으로 결론 보류.
- 다음 라운드 우선순위 1(§9): exp910을 20 epoch 풀로 재학습. (a) vs (b)/(c) 구분.

### 6.4 H4 — Hazard rendering은 expectation blur를 회피해 게인을 주는가? **부분적 yes, 그러나 디자인-약점이 압도**

Hazard 셀 4종 대비 분포 head:

| 셀 | head | aux 형태 | ABS_REL | RMSE | δ1 |
|----|------|---------|--------:|-----:|---:|
| exp913 | hazard full | L_hit + L_free + BerHu/SILog (warmup jump) | 0.4130 | **1.5662** | **0.3016** |
| exp914 | hazard no-free | L_hit + BerHu/SILog | 0.4266 | **1.5894** | **0.2531** |
| exp915 | hazard no-hit | L_free + BerHu/SILog | 0.4388 | 1.2473 | 0.4878 |
| exp916 | hazard depth-only | BerHu/SILog only on rendered depth | 0.4777 | 1.2214 | 0.4963 |
| (참고) exp907 | softmax expectation | NLL+BerHu+SILog | 0.4705 | 1.2269 | 0.5029 |
| (참고) exp907_TESTmedian | softmax median | (= expectation 학습) | 0.4202 | 1.2765 | 0.5129 |

**관찰 1 — α-direct supervision은 깨졌다**:
- exp913, exp914 모두 RMSE 1.56–1.59, δ1 0.25–0.30 — **분포 head 대비 RMSE +0.34, δ1 −0.20** (대규모 폭발).
- 하지만 **ABS_REL은 0.41–0.43으로 분포 head보다 *좋다***. 이는 hazard renderer가 가까운 surface(작은 gt)에서는 단일-mode commit을 잘 하지만(ABS_REL 친화), 멀거나 얼룩진 영역에서는 비현실적 깊이를 commit해 RMSE/δ1를 망치는 패턴.
- 실패 메커니즘(round-3 보고서 + §3.2 epoch 3→4 jump): warmup 끝에서 λ_hit이 0.3→0.5로 점프 → α_hit이 0.95+로 saturate → ∂α/∂logit ≈ 0.05 → renderer가 잘못된 first commit에서 stuck. 이 mechanism이 **train D 점프**, **val RMSE 폭발**, **test δ1 collapse** 세 가지 신호로 일관되게 검출됨.

**관찰 2 — α-supervision off 셀은 분포 head 수준만 회복한다**:
- exp915 (L_free만), exp916 (depth-only): RMSE 1.21–1.25 — round-1 exp906 (RMSE 1.25)와 거의 동일. δ1도 0.49 부근.
- 즉 hazard renderer가 *parametric reformulation*으로서 의미 있는 capacity를 추가하지 *않는다* — 분포 head로 BerHu/SILog가 commit하던 것을 hazard renderer로도 비슷하게 할 수 있을 뿐.
- ABS_REL은 exp915 0.4388, exp916 0.4777 — exp907의 0.4705 대비 exp915가 더 좋고, exp916은 더 나쁨. 두 셀의 차이는 *L_free가 free-space prior로서 도움이 되는지*인데, 결과적으로 exp915 > exp916. 이는 free 손실이 hazard renderer를 잡아주는 데 의미 있다는 약한 증거.

**관찰 3 — Hazard direction은 *다음 라운드의 design rescue*가 필요하다**:
- α-direct BCE는 saturation 트랩으로 폭발(round-3 진단 검증).
- α-off는 hazard 자체의 ceiling이 분포 head와 동일.
- *Rendered quantity(w_j) NLL*, survival/ordinal, soft-target α-BCE 등의 round-4 후보가 round-1 보고서 §6에서 제안됐고 `n2_bulk_0428_r2.sh`에 들어가 있음.

**H4 평가**: hazard rendering 자체는 expectation blur를 *기술적으로 회피*하지만, **(1) 원형 디자인은 학습이 깨지고 (2) 그것을 끄면 분포 head와 같아진다**. 즉 H4는 "이론적으로 yes, 실측으로는 *아직 보여지지 않은 yes*". 다음 라운드 prerequisite.

### 6.5 신뢰성 — Noise floor와 게인의 통계적 의미 (test 데이터로 갱신)

`exp912 (echodiff bs=32 single-GPU)` ↔ `exp920 (echodiff bs=48 DP, seed2)`. 둘 다 echodiffusion 본가, 같은 lr/optim, 변동만 bs + DP + seed:

| 셀 | bs | Val ABS_REL | Val RMSE | Test ABS_REL | Test RMSE | Test δ1 |
|----|---:|----:|----:|-----:|-----:|------:|
| exp912 | 32 (single-GPU) | 0.3914 | 1.4028 | **0.4349** | **1.2432** | 0.4831 |
| exp920 | 48 (DP) | 0.4340 | 1.3695 | 0.4884 | 1.2212 | 0.4890 |

**Test 격차**: ΔABS_REL = +0.0535 (exp920 worse), ΔRMSE = −0.0220 (exp920 better), Δδ1 = +0.0059. 즉 같은 모델/환경/lr에서 bs(32→48) + DP(off→on) + seed(non-det) 변동으로 **test에서 ABS_REL 0.05, RMSE 0.02, δ1 0.006 자연 변동**.

> 주의: train.py에 `torch.manual_seed`가 wired되지 않아 두 run 모두 stochastic. 진짜 *seed-only* 변동은 따로 측정 안 됨 — exp912 ↔ exp920은 *환경 변동*까지 포함한 *상한* noise floor.

이 floor 위에서 round-2의 주요 격차:
- exp907 vs round-1 exp906: ΔABS_REL = −0.011 → **noise floor 1/5** (안)
- **exp907 (분포) vs exp912 (same-env scalar): ΔABS_REL = +0.036 (분포 worse)** → noise floor 안 (방향 일관)
- **exp907_TESTmedian vs exp912: ΔABS_REL = −0.015 (분포 median 우위)** → noise floor 안
- exp907_TESTmedian vs exp907: ΔABS_REL = −0.050 → **noise floor 동률** (1.2배 — 약하게 의미)
- exp913 vs exp915 (hazard 폭발 vs 정상): ΔRMSE = +0.32 → **noise floor 15배 — 명확히 유의**

**Round-2 boldest claim의 등급(갱신)**:
- “bs=32가 round-1의 12 % 격차를 좁힌다” → 격차 자체가 0.05인데 noise floor도 0.05 → **혼란**. round-1 ABS 0.4814 → round-2 ABS 0.4705는 0.011 좁힘만 — 격차의 22 %만. **약한 증거**.
- “분포 head가 same-env scalar를 ABS_REL에서 능가” (round-2 보고서가 한때 주장) → **기각**. exp912가 0.4349로 분포 expectation(0.4705)을 0.036 능가 — noise floor 안이지만 *방향이 분포에게 불리*.
- “분포 head가 same-env scalar를 δ1에서 능가” → +0.020, **noise floor(0.006) 3배 — 강한 증거**.
- “Median switch가 ABS_REL을 개선” → noise floor 1.2배. **중간 강도**.
- “Hazard 원형 디자인은 깨졌다” → noise floor 15배. **강한 증거**.
- “Hazard depth-only ≈ softmax” → noise floor 안. **null 결과(설계가 중립)**.

> 즉 round-2는 **δ1과 hazard 폭발만 명확히 증명한 라운드**. ABS_REL/RMSE 영역은 noise 안에서 trade-off 패턴만 관찰 — 다음 라운드의 epoch 증가나 cell pair 재학습 없이는 더 강한 결론 어려움.

> 사용자의 분석 요구 *“Are improvements consistent across metrics or only isolated”*에 대한 답: **exp907 vs exp906의 “bs 효과”는 isolated(ABS_REL/δ1만, RMSE는 안 나아짐) — 노이즈 안일 가능성이 큼**. **exp907_TESTmedian의 “median 효과”는 inverse trade-off (ABS_REL/δ1 ↑, RMSE ↓)** — 단일 axis 게인이 아닌 metric profile 변경.

---

## 7. Hypothesis-level 결론

### Q1. 분포 head가 binaural audio→ERP depth에 “coarse omnidirectional geometry”를 살리는가?
- **δ1만 강한 yes**, **ABS_REL은 same-env scalar에 진다**, **RMSE는 동률**.
- 새로 들어온 데이터(exp912 ABS 0.4349)는 분포 head expectation(0.4705)이 same-env scalar에 0.036 ABS_REL 뒤짐을 보여줌. 이는 round-1에서 *환경 격차*로 가렸던 사실. **분포 head의 ABS_REL 우위는 *원인 불명의 round-1 환경 격차에 의존했던* 인공물**이었다.
- 메커니즘 재해석: binaural audio는 표면 위치 *분포*는 조밀하게 안다 (δ1 우위 = "GT의 25 % 이내" 픽셀 비율이 일관 +1.5–3pt). 하지만 *대표값*(expectation)을 뽑을 때 multi-mode 평균이 ABS_REL을 깎음. Median switch만 same-env scalar를 미세 추월.
- 결국 H1은 **"분포가 confidence-thresholded metric에 우호적"**이라는 좁은 의미만 살아 있다. global error metric에서의 우위는 검증 안 됨.

### Q2. ERP-aware loss(cos-lat, far-mask)는 sphere layout 학습에 도움인가?
- **OOM-recovered test 데이터로 갱신**: cos-lat alone (exp908) RMSE/MAE/δ1 모두 best. far-mask alone(exp909) 모든 metric worst. combined(exp907)는 ABS_REL/δ1에서 win, RMSE에서 cos-lat alone 대비 약간 손해.
- cos-lat 단독이 RMSE 1.2135로 셀 중 최저 — *polar oversampling 보정* 단독으로 globally consistent depth 추정이 가능. 이는 ERP-spherical layout에 대한 audio 학습이 *위도 가중*만 정정하면 잘 되고 *far-mask*는 *그 위에 ABS_REL을 추가로 깎아주는* 보조 역할.
- **두 보정의 시너지는 합산이 아니라 *trade-off arbitration***: cos-lat alone이 RMSE 최저지만 ABS_REL은 0.4946. far-mask 추가 시 RMSE +0.013 손해 + ABS_REL −0.024 이득. 사용자(연구자)가 어느 metric을 우선시하느냐에 따라 single vs combined를 골라야 함.
- H2는 *조건부 yes*: cos-lat는 명확히 도움(단독, 결합 모두), far-mask는 ABS_REL trade에서만 도움.

### Q3. Hazard rendering이 분포 head의 expectation blur를 우회할 수 있는가?
- **이론적 yes, 라운드-2 실측 inconclusive (디자인 결함으로 폭발)**.
- α-direct BCE supervision은 saturation 트랩에 빠짐. 이는 round-1 보고서에서 미리 진단됐고, round-2가 그 진단을 실측으로 확인.
- **다음 라운드(round-4, scripts/n2_bulk_0428_r2.sh) prerequisite**: rendered quantity supervision(w_j NLL, survival, soft-α). 이 라운드의 결정은 이 부분이 풀린 후로 미뤄야 한다.

### Q4. Bin 수, 분포 표현력, 등급화 효과는?
- Round 2는 bin=32 고정 — bin 수 sweep을 다시 하지 않음. Round 1 §7 Q1 결론(32-bin sweet spot)은 그대로.
- 다만 bin=32 + log spacing의 **마지막 bin이 ~1.4 m 폭**을 가지며, GT가 10 m 클램프에 모임. 이 비대칭이 hazard renderer에서 last-bin α를 saturate시키는 보조 메커니즘일 가능성(round-1 보고서 §6 Q1 alt 가설 c).

### Q5. 다음에 무엇을 바꿔야 하는가? (갱신)
- (1) **Round-4 cell exp931 (event_nll)** — hazard rendered first-hit weight w_j에 NLL을 거는 main candidate (`n2_bulk_0428_r2.sh:289`). 만약 RMSE ≤ 1.30이면 hazard rescue.
- (2) exp910 재학습(median train+inference, bs=32 full 20ep) — H3의 train-time effect 측정. **OOM-recovered exp910 (6/20 epoch) 결과(δ1 0.5237 best in family)는 train-time median이 약속적이라는 신호** → priority 상승.
- (3) **분포 head 학습 epoch 증가 (40 epoch)** — round-0 echodiff best가 40 epoch로 학습된 점을 고려. 분포 head를 같은 epoch budget으로 학습하면 ABS_REL이 0.45 부근까지 회복할 가능성. 분포 head가 underfit이라면 round-0과의 ABS_REL 격차의 *남은 part*가 메워질 수 있음.
- (4) **silog=0.5 + 분포 head 단독 cell** — round-2 audit gap 정리.
- (5) ERP ablation 재테스트는 *완료* (이번 round-2 마무리에서). 다음 라운드 priority 아님.

---

## 8. Failure / Risk 분석

### 8.1 Hazard 원형의 specifics
- **Smoking gun**: D loss epoch 3→4 점프(§3.2). 정확히 λ_hit warmup 종료 시점.
- **Mechanism**: BCE(α, target=1)가 sigmoid를 1로 끌어올림. ∂α/∂logit = α(1−α) → α=0.95에서 0.0475, α=0.99에서 0.0099. depth loss(BerHu/SILog)가 “이 픽셀의 commit이 잘못됐다”고 신호를 줘도 hit-bin α를 조정할 grad가 0에 가까움 → renderer가 잘못된 first commit에서 영구 stuck.
- **Fix**: 라운드 4의 smooth ramp(`progress = min(1, epoch/ramp_ep)`)는 점프를 없앰 + rendered-quantity supervision은 α를 우회.

### 8.2 Round-2 audit gap이 분포 head 비교를 오염시키는 정도
- exp907 (silog=1.0) ↔ exp919 (silog=0.5)는 silog 가중치가 같이 바뀜. Round-1 ↔ Round-2 비교도 마찬가지 — round-1 silog=1.0 (echorange.yaml의 train.w_silog=0.5와 어긋난 mismatch가 round-2 audit으로 확인됨).
- **이는 round-1의 분포 head 결과를 그대로 round-2 best와 비교할 때 0.5x silog 미적용 문제를 동반함**. Round-2 audit fix(exp919, exp920, exp913–916)는 silog=0.5로 정렬됐으나 exp907 자체는 silog=1.0으로 학습.
- **얼마나 영향?** exp919는 silog=0.5 + λ_NLL=0.3로 두 변수가 동시 변동 → 단독 silog 영향 분리 불가. 다음 라운드에 silog=0.5 단독 분포-head cell이 필요.

### 8.3 OOM/truncated test의 영향 (해결됨)
- exp908/909/910/912 test가 1차 시도에서 OOM. 2026-04-28 후속 재실행에서 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + `--batch-size 2`로 회수 완료.
- **결론을 *근본적으로* 바꾸는 입력**: exp912 (echodiff bs=32 single-GPU, ABS 0.4349)이 들어옴으로써 §6.1 H1의 "분포 head ABS_REL 우위" 주장 weakened. round-1 시기 의심됐던 "echorange-scalar gap"이 실은 *부재했던* 사실 확인.
- ERP ablation 데이터 (§6.2)는 이제 test 채널에서 직접 비교 가능 → val-only 결론보다 강한 evidence.
- 백업 OOM 로그는 `<EXP>_test_OOM.log`로 보관(증적용).

### 8.4 환경/노이즈 floor
- exp912 ↔ exp920 비교는 bs(32 ↔ 48)와 seed(non-deterministic init/shuffle) 모두 변동. **순수 seed 노이즈를 측정한 것이 아님**.
- train.py에 `torch.manual_seed(...)`이 없음 → 같은 PYTHONHASHSEED=1로 두 번 돌려도 결과가 다를 수 있음. **다음 라운드 보강 필요**(audit critique already noted).

### 8.5 단일 데이터셋, 단일 split
- Matterport3D scene-disjoint split 1개에서만 측정. δ1, ABS_REL, RMSE의 0.01 단위 차이는 다른 split에서 충분히 뒤집힐 수 있다. **외부 일반화는 본 라운드 결과로 주장 불가**.

---

## 9. 다음 라운드 권고 (우선순위)

### Priority 1 — Round-4 hazard rescue (이미 launch)
- `scripts/n2_bulk_0428_r2.sh` 실행 중. cell exp931 (`event_nll + free`)이 main. RMSE ≤ 1.30이면 hazard 살리기 성공, > 1.42이면 hazard 폐기 + ordinal/quantile head로 pivot.
- 비용: 6 cell × ~150 분 = 15 시간(4 GPU 페어 병렬 시 ~4 시간 wall-clock). Round-1/2 진단을 단일 스윕으로 verify or kill.

### Priority 2 — Median 학습 (재실행, exp910 retrain)
- exp910은 6 epoch에서 끊겼음. 20 epoch full로 다시 학습 + test_only도 같이 — 결과:
  - **만약 train+inference median이 expectation+TESTmedian과 같은 수준이면**: median은 inference-time switch로 충분 → 분포 head 디자인은 그대로 두고 median을 default 출력 모드로 바꾸는 게 production guidance.
  - **만약 train+inference median이 더 좋다면**: train-time loss alignment(NLL이 expectation 친화적이지만 median을 출력으로 쓸 때의 mismatch)가 한 번 더 개선 가능.
- 비용: 1 cell × ~150 분.

### Priority 3 — ERP ablation 재테스트 (n9 또는 n2)
- exp908/909/910/912의 test_only를 재실행 (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 환경 변수 설정 + bs=4). 분리 ablation 결론(§6.2)을 test 데이터로 보강.
- 비용: 4 cell × ~5 분 = 20 분.

### Priority 4 — silog=0.5 + 분포 head 단독 cell
- exp907의 silog만 1.0→0.5로 바꾼 1-cell. round-2 audit critique를 분포 head에 직접 적용. 만약 결과가 exp907과 다르지 않으면 silog 가중치는 metric에 큰 영향이 없는 hyperparam.
- 비용: 1 cell × 100 분.

### Priority 5 — bin 수 fine-grain (round-1 §8.C에서 미뤄둠)
- bin=24/40/48 추가. 32-bin sweet spot 가설이 더 큰 그리드에서 깨지는지 확인. 단 round-4가 hazard 결론을 줘야 할 차례라 priority 낮음.

### 단기 결론
> **다음 라운드에서 *반드시* 측정해야 할 건 (1) hazard event_nll 결과(round-4 진행), (2) median train+inference 결과(exp910 retrain). 둘 다 라운드-2의 핵심 가설(H3, H4)을 직접 검증한다**.

---

## 10. 사용자 연구 가설(SH low-order guidance)에 대한 본 라운드의 함의

> 사용자의 큰 그림 가설은 “binaural audio는 dense detail보다 *low-order spherical structure*를 더 잘 안다”는 것. Round 2의 실험은 SH coefficients를 직접 supervise하지 *않지만*, 그 가설의 **proxy verification**을 제공한다.

- **분포 head의 δ1 우위 (모든 round/bs/scalar variant 통틀어 +1.5–3pt)**: 사용자 가설을 *행동적으로* 지지하는 가장 강한 신호. 분포 출력의 multi-mode가 audio가 표현 가능한 “coarse posterior”에 가깝다는 해석. **새 데이터(exp912)로 갱신**: same-env scalar(δ1 0.4831) 대비 분포 head(0.5029)가 +0.020 — noise floor 3배.
- **ABS_REL은 same-env scalar에 진다**: 분포 head expectation(0.4705) > scalar(0.4349). Median switch만 미세 추월(0.4202 vs 0.4349). 이는 “low-order layout만 audio가 안다”는 가설과 *부분 충돌* — coarse layout(δ1)은 살리지만 fine-grained 깊이 정확도(ABS_REL)는 못 살린다는 의미.
- **ERP cos-lat 보정의 RMSE/MAE/δ1 단독 우위**: 새 test 데이터로 cos-lat alone(exp908)이 RMSE 1.2135로 *셀 중 최저*. polar oversampling 보정이 spherical layout 표현(SH L=0,1 — overall scale, 단방향 dipole)과 정렬됨을 시사.
- **Far-mask는 ABS_REL 이득에서만 contributing**: 단독으로는 모든 metric worst, combined에 들어가야 ABS_REL 도움 → far-domain은 audio가 약하다는 가설(L=2+ 또는 high frequency mode가 audio에서 소실)과 호환.
- **Hazard renderer의 RMSE 폭발**: hard commit이 “distribution을 single mode로 collapse”시키는 design — SH framework에서 L=∞에 해당. 사용자의 “high-order는 audio로 학습 안 됨” 가설과 정량적으로 부합 (RMSE 1.57 폭발).

→ **연구 방향 제안 (갱신)**: round-2 데이터는 SH-direct supervision로의 확장에 일관된 방향성을 제공:
  - (a) **L=0,1,2 coefficient supervision branch** — ERP depth → real SH L≤2 변환을 GT로 두고 분포 head와 병렬로 SH coefficient regression head를 추가. 본 라운드의 cos-lat-alone RMSE 우위가 *L=0,1 차수 정보가 audio에 가장 가깝다*는 약한 증거.
  - (b) **분포 head + median을 SH coefficient의 angular profile에 직접 적용** — 분포 head는 *각 픽셀별* depth distribution을 만들지만, 이를 ERP-wise로 합산하면 coarse SH 계수 추정에 자연스레 연결.
  - (c) **Hazard 또는 hard-commit head는 SH framework에 부적합** — 본 라운드 결과로 정량화됨.
  - (d) Round-2 ABS_REL gap의 source가 "audio는 fine-grained accuracy를 못 안다"인지 "underfit이라 학습 epoch 부족"인지가 갈림길 — *epoch 40 cell*이 그 구분에 직접 답을 줌.

이 제안은 round-2 데이터에 *기반한 추측*이며, 직접 SH-coefficient 실험이 다음 단계 우선순위 후보.

---

## 11. 부록

### A. 학습 / 평가 명령 (재현용 요약)

```bash
cd /root/storage/implementation/shared_audio/baseline

# 전체 round-2 sweep (n9 server, bs=32)
bash scripts/n9_bulk_0427.sh

# round-2 audit + hazard sweep (n2 server, DP, bs=48)
bash scripts/n2_bulk_0428.sh

# round-4 (hazard rescue, 별도 라운드)
bash scripts/n2_bulk_0428_r2.sh
```

특정 cell만 학습:

```bash
# 예: exp907
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /opt/conda/bin/python3 train.py \
  --config echorange --experiment-name exp907_echorange_32log_d10_coslat_farmask_lr1e4_bs32 \
  --epochs 20 --lr 1e-4 --batch-size 32 --num-workers 4 \
  --depth-dir erp_depth_radial \
  --dataset-dir /root/local1/changwoo/matterport3d_0303_renew \
  --depth-head-type range --range-num-bins 32 --range-bin-spacing log \
  --range-min-depth 0.1 --range-max-depth 10.0 \
  --range-soft-label-sigma 0.14 --range-output-mode expectation \
  --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 1.0 \
  --erp-cos-lat-weight --erp-far-mask
```

### B. 라운드-2에서 발생한 코드/디자인 변경 요약

| 영역 | 변경 | 출처 |
|------|------|------|
| `models/bin_based/range_head.py` | Hazard 클래스(`HazardRangeDepthHead`), `hazard_supervision_loss`, `rendered_event_nll`, `survival_loss`, `soft_hit_bce_loss`, `hazard_free_loss` 추가 | round-3 hazard 도입 |
| `models/bin_based/echorange.py` | `depth_head_type='hazard'` 분기, hazard head 빌드 코드 | 동일 |
| `train.py` | `_train_step_echorange` hazard branch, smooth ramp warmup, ablation flags(`disable_hit_loss`, `disable_free_loss`, `hazard_depth_only`), `_hazard_diagnostics` 도입 (round-4 추가) | round-3, round-4 |
| `config/echorange.yaml` | `depth_head_type` default `'scalar'`, hazard hyperparam (warmup, λ), `lambda_silog: 0.5` 정렬 | round-2 audit fix |
| `test.py` | hazard CLI 인자 처리, range output mode override (test_only median용) | round-2 |
| `utils/test_utils.py` | hazard pred 정규화 path (range head와 공유) | round-3 |
| `scripts/n9_bulk_0427.sh`, `n9_bulk_0427_re.sh`, `n2_bulk_0428.sh` | 라운드 sweep launcher | 본 라운드 |

### C. 파일 / 산물 위치

- 학습 로그: `logs/n9_0427_train/exp9{07..20}_*.log`
- 테스트 로그: `logs/n9_0427_test/exp9{07..20}_*_test.log`
  - test_only median 결과: `logs/n9_0427_test/exp907_echorange_32log_d10_coslat_farmask_TESTmedian_test.log`
- Best checkpoints: `checkpoints/echorange_soundspaces_BS{32,48}_Lr0.0001_AdamW_exp9*/best_model.pth`
  - 분포 head SOTA: `exp907_echorange_32log_d10_coslat_farmask_lr1e4_bs32/best_model.pth`
  - Hazard family: `exp913–916` (bs=48)
- Code snapshot (round-1+round-2 통합): `logs/round1_bin-based/code_snapshot/`
- 라운드-1-frozen 코드 (참조용): `logs/round1_bin-based/code_snapshot/round1_frozen/`
- 보고서: `logs/round1_bin-based/round2/E20260428-exp907-920-bs32_ERPablation_hazard_audit_sweep.md` (이 파일)
- IO/code description: `logs/round1_bin-based/round2/E20260428-exp907-920-IO_and_code_description.md`
- 보존된 체크포인트 (Tier A, round2/ 안):
  - `E20260428-exp907-bs32_ERPfix_distribution_best.pth`
  - `E20260428-exp915-Hazard_nohit_best.pth`
  - `E20260428-exp916-Hazard_depthonly_best.pth`

### D. 환경 메모

- **n9 server**: 단일 GPU 모드 only (GPU 2 driver fault). bs=32 단일 GPU에서 24 GiB가 거의 90 % 사용.
- **n2 server**: 8-GPU node, DataParallel 기반 4-페어 병렬. bs=48 DP에서 per-GPU 24 샘플로 ~80 % 메모리 사용.
- **dataset path 변동**:
  - n9: `/root/local1/changwoo/matterport3d_0303_renew` (underscore)
  - n2: `/root/local1/changwoo/matterport3d_0303renew` (no underscore)
- **OOM 회피**: test pass에 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 권장 (라운드-2의 4 cell test가 이 설정 없이 OOM).

### E. 의미 있는 체크포인트(다음 라운드/외부 평가용 후보)

“의미 있는” 정의 = 본 라운드의 핵심 결론을 직접 받쳐주거나, 다음 라운드에서 비교 baseline으로 쓰일 수 있는 것.

| Tier | 체크포인트 | 의미 |
|------|-----------|------|
| **A (필수)** | `exp907` | round-2 분포 head SOTA, test_only median 비교용 |
| **A** | `exp915` (hazard no-hit) | hazard 가족에서 정상 학습한 best (RMSE 1.247) |
| **A** | `exp916` (hazard depth-only) | hazard renderer + BerHu/SILog only — capacity 비교 baseline |
| B | `exp912` (echodiff bs=32) | Round-2의 same-environment scalar baseline |
| B | `exp920` (echodiff bs=48 seed2) | Noise floor 측정용 second-seed |
| B | `exp911` (echorange-scalar bs=32) | echorange ↔ echodiffusion path equivalence 검증 |
| C | `exp913` / `exp914` (hazard 폭발) | 실패 디버깅용 |
| C | `exp919` (λ_NLL=0.3) | NLL weight ablation |

권장 실제 보존: **A 3개(exp907, exp915, exp916)을 round2/ 디렉터리에 archived 사본으로**, 나머지는 `checkpoints/` 원위치 유지(언제든 재테스트 가능).

---

## 12. 한 줄 평 (OOM-recovered 데이터 반영 후 갱신)

> Round 2는 **same-env scalar gap이 부재하고(echodiff 본가가 우리 환경에서 재현됨, exp912 ABS 0.4349), 분포 head는 δ1만 일관 우위(+0.020)이며 ABS_REL은 같은 환경 scalar에 *진다*** 는 것을 정량화했다. Median switch는 ABS_REL을 noise floor 1.2배 위로 개선하지만 RMSE는 손해. Hazard 원형 디자인은 saturation으로 폭발 — 깰 수 없게 부서진 상태에서 round-4 rescue로 넘김. **다음 결정은 (1) exp910 train-time median 재학습, (2) round-4 event_nll 결과, (3) 분포 head 40-epoch 학습 — 이 세 점에 달려 있다.**
