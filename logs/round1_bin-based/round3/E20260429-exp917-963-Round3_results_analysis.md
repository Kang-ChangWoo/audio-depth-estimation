# E20260429 — Round 3: Hazard rescue + Range posterior + Soft-quantile + SH + Cylindrical (exp917, 930–963) 결과 분석

> **Scope**: Round-2 (`round2/E20260428-exp907-920-bs32_ERPablation_hazard_audit_sweep.md`)에서 미해결로 남은 (1) Hazard 첫-히트 head의 α-saturation rescue, (2) 분포 head의 ABS_REL 격차(round-2 same-env scalar 0.4349 ↔ exp907 distribution 0.4705), (3) Round-2 audit critique의 silog/λ_NLL/median 결론 — 세 갈래를 R4(exp917, 930–934 hazard rescue)와 R5(exp936–963 range posterior + soft-quantile + SH + cylindrical + 4-best ckpt)로 나눠 28셀 학습한 라운드. 본 문서는 두 batch의 통합 verdict.
>
> Bulk launchers: `scripts/n2_bulk_0428_r2.sh` (R4 hazard rescue), `scripts/n2_bulk_0429_r3.sh` + `scripts/n9_bulk_0429_r3.sh` (R5 main).
>
> Code snapshot: `logs/round1_bin-based/code_snapshot/round3_frozen/`.

---

## 0. TL;DR

- **R4 Hazard rescue 결론 — hazard 메인 폐기**: soft_hit(exp934) 가 분포 head 수준만 회복(ABS 0.4522, RMSE 1.2293, δ1 0.4988); raw_hit(exp930)은 ABS 0.4067로 좋아 보이나 **RMSE 1.6270 폭발 + δ1 0.3011 붕괴** → infeasible. event_nll(exp931/932)와 survival(exp933) 모두 ABS 0.47–0.50, RMSE 1.22로 softmax baseline과 동률. **Hazard rendering 자체로는 게인 없음**, round-2 결론과 일치.
- **R5A Soft-Quantile sweep 결론 — RMSE↔ABS_REL trade-off만 확인**: λ_sq=0.5 (exp941/942/943) 셀이 RMSE 1.20–1.21로 28셀 중 최저, 그러나 **ABS_REL 0.50–0.52로 동반 악화**. λ_sq=0.25 (exp936/939) 셀은 ABS 0.46–0.47, RMSE 1.21–1.23 — 곡선 위 다른 위치. **곡선을 바깥으로 미는 셀은 없음**.
- **R5B Spherical-SH sweep 결론 — high-λ + log-depth 가 유일한 명확 게인**: **exp946 (L=2, λ_SH=0.10, log_depth=true)이 28셀 중 유일하게 7개 metric 모두 echodiff exp958(same-env 40ep) 능가** (ABS 0.4413, RMSE 1.2208, δ1 0.5019, Log10 0.1542, MAE 0.7755). 그러나 **round-0 SOTA 대비 ABS +0.011, RMSE +0.115는 여전히 손해**. λ_SH 0.02·0.05는 noise floor 안 진동, 0.10에서만 일관 우위 — λ-curve 비단조.
- **R5C Combo (sq + SH) 결론 — 시너지 없음**: exp950 (q0.5/τ0.05/λ_sq0.25 + L2/λ_SH0.02) ABS 0.5158, RMSE 1.2141 — **단독 R5A·R5B 어느 쪽보다도 나쁨**. 두 loss를 합치면 서로 간섭함을 확인. 라운드 결론 — **combo 라인 폐기**.
- **R5F 40-epoch paired baseline 결론 — epoch budget은 게인 원천 아님**: exp958 (echodiff scalar 40ep) best ckpt가 ep16에서 멈춤; ABS 0.4463 / RMSE 1.2611은 round-2 same-env exp912 (20ep, 0.4349/1.2432) 보다도 ABS·RMSE 모두 약간 손해. **40ep도 round-0 exp11(0.4300/1.1060) 격차를 닫지 못함**. 반면 **exp961 (range median, 40ep)이 δ1 0.5199 — 28셀 중 δ1 챔피언**, round-2의 inference-only median(0.5129)보다도 강함. **train-time median이 δ1을 더 끌어올리는 것은 명확**, 그러나 RMSE 1.2444로 round-0 미달.
- **R5G·R5H Br=20 저-비닝 결론 — 비닝 해상도 20은 32 대비 후퇴**: exp962 (R20 sq) ABS 0.4686 / RMSE 1.2218 / δ1 0.5034, exp963 (R20 SH) ABS 0.5221 — Br=32 R5A/R5B anchor와 비교해 개선 없음 또는 후퇴. **Br=32–40이 적정 sweet spot**.
- **4-best ckpt 결론 — best_score(=0.7·RMSE + 0.3·ABS_REL)가 가장 균형 잡힘**: 모든 셀에서 best_rmse·best_absrel·best_delta1 단일 메트릭 ckpt를 test set에 돌리면 **타깃 metric은 살짝 좋아지나 다른 metric 큰 손해** 패턴. 예: exp946 best_score (0.4413/1.2208/0.5019) vs best_delta1 (0.5320/1.2683/0.5048) — δ1 +0.003 미세 우위 대신 ABS +0.091 큰 손해. **val set 작아 (2 951) ckpt selection이 noise를 amplify**, 4-best가 실용적 게인 없음 → 운영상 best_score만 유지 권고.
- **핵심 진단 — Trade-off 곡선에 갇혔다**: 28셀 (ABS_REL, RMSE) 평면을 그리면 `저-RMSE 그룹 (R5A λ=0.5)` ↔ `저-ABS_REL 그룹 (R5B SH high-λ + R4 soft_hit)` ↔ `고-δ1 그룹 (R5F median)` 세 지대 중 어디 하나에만 머물고, **세 metric 동시 개선 셀은 0건**. exp946이 exp958을 넘는 유일 셀이지만 ABS 차이 −0.005는 round-2 noise floor(±0.05) 안.
- **Round-0 격차(RMSE 1.10) 미해결**: same-env echodiff 본가도(exp912 1.2432, exp958 1.2611) round-0 exp11(1.1060)을 +0.13~0.16 손해. **head 디자인 이전 baseline 수준의 격차**. R5 어떤 셀도 RMSE 1.20 미만 못 감 (최저 exp942 1.2030). **다음 라운드의 priority-0**.
- **다음 라운드 권고 (R6)**: (1) Round-0 baseline 재현 진단(lr schedule / wav2vec2 freeze / depth_norm scale), (2) 분포 + scalar multi-task head 또는 sub-bin residual head로 trade-off 곡선을 외향 이동, (3) R5B(exp946) cfg + R5F median 결합 셀 — 이미 살아 있는 두 축의 가산 검증.

---

## 1. 라운드 위치 및 의도

```
Round 0 (외부 echodiff 본가)        : exp11 best  ABS 0.4300, RMSE 1.1060, δ1 0.4876   ← 우리가 따라잡아야 하는 baseline
Round 1 (round1/exp900-906)        : exp906 SOTA ABS 0.4814, RMSE 1.2532, δ1 0.5079    ← bin-based softmax 첫 도입
Round 2 (round2/exp907-920)        : exp907      ABS 0.4705, RMSE 1.2269, δ1 0.5029    ← bs=32 + ERP fix, scalar gap 사실상 부재 발견
                                     exp912      ABS 0.4349, RMSE 1.2432, δ1 0.4831    ← same-env scalar key (round-0 0.4300과 0.005 차)
                                     exp907_TESTmedian ABS 0.4202, RMSE 1.2765, δ1 0.5129  ← inference-only median, ABS·δ1 동시 우위
                                     exp913 hazard 폭발 ABS 0.4130, RMSE 1.5662, δ1 0.3016  ← α-BCE saturation
Round 3 (이 문서, exp917 + 930-963)
  R4-batch (n2_bulk_0428_r2.sh)
    exp917 raw_hit strong          : 0.4291 / 1.6095 / 0.2612                          ← α-saturation 그대로
    exp930 R4-00 raw_hit 재현       : 0.4067 / 1.6270 / 0.3011                          ← 동일 폭발
    exp931 R4-01 event_nll free=0  : 0.4984 / 1.2169 / 0.4962                          ← softmax 회복
    exp932 R4-02 event_nll only    : 0.5010 / 1.2233 / 0.4861
    exp933 R4-03 survival          : 0.4733 / 1.2202 / 0.4959
    exp934 R4-04 soft_hit t=0.75   : 0.4522 / 1.2293 / 0.4988                          ← R4 best, hazard 메인 폐기 결정
  R5-batch (n2_bulk_0429_r3.sh, n9_bulk_0429_r3.sh)
    R5A Soft-Quantile (exp936-943) : ABS 0.46–0.52 / RMSE 1.20–1.23 / δ1 0.49–0.51    ← RMSE↔ABS trade-off
    R5B Spherical-SH  (exp944-949) : ABS 0.44–0.52 / RMSE 1.20–1.23 / δ1 0.49–0.51    ← exp946 SOTA(7 metric all-win vs exp958)
    R5C Combo         (exp950-951) : 단독 R5A·R5B보다 나쁨                              ← combo 폐기
    R5F 40-epoch paired (exp958-961): exp958 0.4463/1.2611/0.4983 (ep16/40 → epoch budget 무용)
                                     exp961 0.4520/1.2444/0.5199 (range median ★ δ1 챔피언)
    R5G·R5H R20       (exp962-963) : R20 sq 0.4686/1.2218/0.5034, R20 SH 0.5221/1.2267/0.5010 ← 후퇴
```

라운드 의도:
- **A. Hazard closure**: round-2 진단 "α-saturation은 raw-hit 단독 BCE의 구조적 함정"을 4 가지 새 aux mode (event_nll · survival · soft_hit · raw_hit-with-smooth-ramp)로 분해 검증.
- **B. 분포 head ABS_REL 격차 좁히기**: round-2 exp907→exp912 +0.036 격차의 source가 (a) bin discretization 자체인지 (b) expectation 디코딩 bias인지 (c) loss 누락(soft-quantile)인지 분리. R5A soft-quantile + R5F train-time median이 이 격차에 대응.
- **C. 분포 head의 보조 신호 추가**: round-2 ERP-ablation에서 `cos-lat × far-mask` 비-가법 시너지가 발견됨 → 더 일반화된 구면 보조 신호인 SH coeff matching(R5B)으로 확장.
- **D. Bin-axis swap**: radial bin 외 horizontal/z bin 시도(R5D — exp954–956, 본 문서에는 미포함; 다음 라운드로 보류).
- **E. epoch budget 가설 닫기**: round-2의 "분포 head epoch 늘리면 격차 줄지" 의문을 R5F 40-epoch paired run으로 검증.
- **F. 4-best ckpt 도입**: round-2의 best_model.pth 단일 ckpt가 metric profile을 합성 score로만 잡음 → 4-best로 metric-별 ckpt 비교 가능하게.

라운드는 **(A) closure → (B/C) 신규 신호 → (E) epoch 가설 → (F) ckpt UX** 네 갈래를 한 번에 묶어 학습 비용을 amortize.

---

## 2. 셋업

### 2.1 환경

| 항목 | n2 batch (R4 + R5A/B/C) | n9 batch (R5F/G/H) |
|---|---|---|
| 모델 | echorange (depth_head_type ∈ {scalar, range, hazard}) | 동일 + echodiffusion (exp958) |
| Backbone | EcoDepthEncoder + Decoder + Wav2Vec2-base | 동일 |
| 데이터 | matterport3d_0303_renew, ERP radial depth (`erp_depth_radial/`) | 동일 |
| Split | scene-split, train 23560 / val 2951 / test 2951 (3192 dataloader fanout) | 동일 |
| Bin grid | 0.1–10 m, log-spaced, Br=32 (default), Br=20 (R5G/H), Br=40 (R5F R40) | 동일 |
| 입력 | spectrogram(2,256,512) + waveform(2,5648) | 동일 |
| Optimizer / lr | AdamW / 1e-4 | 동일 |
| Batch size | 48 (DataParallel 2-GPU pair) | 32 (single-GPU) |
| Epochs | 20 (R4 + R5A/B/C) | 40 (R5F), 20 (R5G/H) |
| 하드웨어 | n2 8-GPU, GPU pair × 4 | n9 GPU 0,1만 |
| silog 가중 | 0.5 (round-2 audit fix) | 0.5 |
| 학습 시간 | ≈ 2.5 h/cell @ 20ep | ≈ 5 h/cell @ 40ep, ≈ 2.5 h @ 20ep |

### 2.2 R5 신규 플래그 (default off, backward-compatible)

```
--lambda-soft-quantile <float>        (default 0.0)
--soft-quantile-q <float>             (default 0.5)
--soft-quantile-tau <float>           (default 0.05)
--lambda-spherical-sh <float>         (default 0.0)
--spherical-sh-order <int 2..4>       (default 2)
--spherical-sh-log-depth              (boolean, default true)
--range-bin-axis radial|horizontal|z  (default radial)
--cyl-min-axis-factor <float>         (default 0.15)
--range-eval-mode default|expectation|map|q25|q35|q45|q50|q55|q65|q75|temp05|temp075|temp15
--checkpoint-tag score|absrel|rmse|delta1   (default score)
```

### 2.3 R5 신규 코드 모듈

- `models/bin_based/range_head.py` — `RangeDepthHead` 9 output mode + `range_point_estimate(...)` + `soft_quantile_depth(logits, range_bins, q, τ)` (gradient-friendly differentiable quantile via softmax-weighted bin aggregation).
- `models/bin_based/spherical_loss.py` (신규) — `make_erp_grid(H,W)`, `_real_sh_basis(L≤4)`, `spherical_sh_coeffs`, `spherical_sh_loss(L, use_log_depth, area_weight=cos_lat, smooth_l1)`.
- `train.py:141..516` `_train_step_echorange` — range/hazard 분기 + soft-quantile 항 + SH 항 + cylindrical bin-axis 투영.
- `train.py:1270..1488` 4-best ckpt 분리 저장.
- `utils/test_utils.py` `_override_range_pred_depth` + `_project_pred_to_radial`.

전부 `code_snapshot/round3_frozen/`에 동결.

---

## 3. 학습 진행 (셀별 best epoch)

R4·R5 모든 셀의 best_score epoch는 다음과 같음 (괄호 안은 학습 budget):

```
R4 hazard rescue:
  exp917  ep20/20  → α-saturation, 학습 끝까지 회복 못 함
  exp930  ep02/20  → raw_hit 재현 폭발, 초기에만 D 작은 값
  exp931  ep12/20  → event_nll, 정상 수렴
  exp932  ep08/20  → event_nll only
  exp933  ep08/20  → survival, 빠른 수렴
  exp934  ep12/20  → soft_hit, 정상 수렴

R5A Soft-Quantile (best_score):
  exp936  ep06/20  exp937  ep06/20  exp938  ep12/20  exp939  ep08/20
  exp940  ep08/20  exp941  ep10/20  exp942  ep10/20  exp943  ep06/20

R5B Spherical-SH (best_score):
  exp944  ep08/20  exp945  ep06/20  exp946  ep08/20
  exp947  ep08/20  exp948  ep10/20  exp949  ep10/20

R5C Combo (best_score):
  exp950  ep08/20  exp951  ep10/20

R5F 40-epoch (best_score):
  exp958  ep16/40 ★ 40ep 학습이지만 best는 ep16 → epoch budget 무용 결정적 증거
  exp959  ep12/40  exp960  ep06/40  exp961  ep10/40

R5G·R5H R20 (best_score):
  exp962  ep10/20  exp963  ep10/20
```

**관찰**:
- R5A·R5B 모든 셀이 ep06–ep12에서 best_score 도달. 20 epoch budget 안에서 후반부 train loss는 계속 떨어지지만 val score는 ep10 부근에서 plateau → mild overfit.
- exp958 (echodiff 본가, 40 epoch 학습)의 best_score epoch가 **ep16**로 round-2 exp912(20ep, ABS 0.4349)보다도 **ABS 0.4463로 약간 손해** — 단순히 epoch 길이로는 round-0 exp11(0.4300) 격차를 닫지 못함.
- R4 raw_hit 셀(exp930)은 ep02에서 best 후 폭발 — α-saturation이 학습 초기에 이미 발생.

---

## 4. 패밀리별 결과 (best_score 기준 7-metric 표)

전체 entries는 `E20260429-GPTpro_round5_consult_request.md` 부록 B 참조 (4-best ckpt 변형 포함 70 entries). 본 절은 best_score 28셀만.

### 4.1 R4 Hazard rescue (n2 bs=48 DP)

| Exp | aux mode | bias | warmup | λ_hit | λ_free | best_ep | ABS_REL | RMSE | δ1 | δ2 | δ3 | Log10 | MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp917 | raw_hit (strong) | -4.6 log32 | 5 | 1.0 | 0.2 | 20 | 0.4291 | **1.6095** | 0.2612 | 0.4965 | 0.6829 | 0.2462 | 1.0769 |
| exp930 | raw_hit | -4.6 | 3 | 0.10 | 0.05 | 02 | **0.4067** | **1.6270** | 0.3011 | 0.5329 | 0.7019 | 0.2305 | 1.0561 |
| exp931 | event_nll (free off) | — | 3 | 0.10 | 0.0 | 12 | 0.4984 | 1.2169 | 0.4962 | 0.7122 | 0.8281 | 0.1584 | 0.7948 |
| exp932 | event_nll | — | 3 | 0.10 | 0.05 | 08 | 0.5010 | 1.2233 | 0.4861 | 0.7047 | 0.8252 | 0.1593 | 0.7949 |
| exp933 | survival | — | 3 | 0.10 | (off) | 08 | 0.4733 | 1.2202 | 0.4959 | 0.7063 | 0.8282 | 0.1563 | 0.7853 |
| **exp934** | **soft_hit t=0.75** | — | 3 | 0.05 | 0.02 | 12 | **0.4522** | **1.2293** | **0.4988** | **0.7158** | **0.8343** | **0.1548** | **0.7794** |

**verdict**: hazard rescue에서 살아남은 셀(soft_hit/event_nll/survival)이 모두 **softmax baseline 수준 (RMSE 1.22, δ1 0.49)** 만 회복. exp934가 R4 best지만 round-2 exp907 (분포 head softmax, 0.4705/1.2269/0.5029) 대비 ABS −0.018로 살짝 우위, RMSE/δ1은 동률. **R5 메인 라인에서 hazard 메인 폐기 결정**, hazard closure 1 셀(exp957)만 R5에 잔존.

raw_hit 두 셀(exp917, 930)은 round-2 exp913(1.5662)·exp914(1.5894) 폭발과 동일 패턴 — α-direct BCE가 saturate되면 bin distribution이 last bin으로 collapse하여 RMSE 폭발 + δ1 붕괴. **soft target(0.75)이 saturation 우회**.

#### 4.1.A R4 학습 dynamics (셀별 D loss 흐름)

**exp930 (raw_hit smooth ramp, 5 epoch만)**:
```
Epoch 1 [1/5] L:0.5058 D:0.5225
Epoch 2 [2/5] L:0.5129 D:0.5103   ← Val best (RMSE 1.9071, ABS 0.4095)
Epoch 3 [3/5] L:0.5627 D:0.5429   ↑ depth loss reverses
Epoch 4 [4/5] L:0.5439 D:0.5265
Epoch 5 [5/5] L:0.5284 D:0.5113
```
→ depth loss monotonic 아님; epoch 3에서 reverse. round-2 exp913(0.47→0.52 jump @ ep3→4)과 같은 시그너처.

**exp917 (strong raw_hit, λ_hit=1.0)**: D=0.55→0.45→0.47→0.60 (ep 1→2→3→4) — epoch 4 jump. round-2 exp913 동일 패턴 + 강한 supervision으로 폭발 *가속*.

**exp931/932/933/934 (rendered/soft 가족)**: D loss 모두 monotonic (0.46→0.30→0.18 over 20 epoch). **smoking-gun jump 없음** — saturation을 BCE 형식 자체에서 회피.

#### 4.1.B R4 내부 메커니즘 dump (val 1-shot, `_hazard_diagnostics`)

R4 라운드는 hazard head의 α saturation 메커니즘을 직접 정량화하기 위해 val pass의 첫 batch에서 percentile dump를 수집.

**α saturation (`frac>.99` = α ≥ 0.99인 픽셀×bin 비율)**:

| Cell | ep 2 frac>.99 | best_ep frac>.99 | α p99 | α p95 | 진단 |
|---|---:|---:|---:|---:|---|
| exp917 (raw_hit strong) | 0.299 | 0.183 (ep14) | 1.0000 | 1.0000 | **30 % 픽셀 saturate** — sigmoid 포화로 grad vanish |
| exp930 (raw_hit smooth) | 0.302 | 0.242 (ep4) | 1.0000 | 1.0000 | 동일 패턴 — smooth ramp 무효 |
| exp931 (event_nll+free) | 0.036 | 0.031 (ep12) | 1.0000 | 0.9~ | tail saturation만 |
| exp932 (event_nll only) | 0.032 | 0.031 | 1.0000 | 0.9~ | 동일 |
| exp933 (survival) | 0.008 | 0.002 (ep8) | 0.9549 | 0.8~ | 매우 안정 |
| exp934 (soft_hit 0.75) | 0.004 | **0.000** (ep12) | **0.9008** | **0.7685** | saturation **완전 회피** |

→ `H_R4_1: α-direct BCE saturation은 raw_hit 폭발의 root cause` **Strongly Supported** — saturation severity ∝ RMSE/δ1 degradation 단조 chain.

**bg_weight collapse** (`bg = ∏(1−α)` — first-hit이 어디서도 발생하지 않을 확률):

| Cell | bg p10 | bg p50 | bg p90 |
|---|---:|---:|---:|
| exp917, exp930 | 0.000 | 0.000 | 0.000 |
| exp931, exp932 | 0.000 | 0.000 | 0.000 |
| exp933 (survival) | 0.000 | 0.000 | 0.008 |
| exp934 (soft_hit) | 0.000 | 0.000 | **0.001** |

→ 모든 hazard 셀에서 bg_weight 거의 0. soft_hit이 미약하게 회복하지만 substantive 아님. **`pred = Σw_j r_j + w_bg max_depth` 구조가 *최대 깊이를 모델링 못 함*** — round-4의 inherent 디자인 한계 발견.

**argmax_bin_hist (folded 32→16 buckets, %)**:

| Cell | dominant bin | spread 평가 |
|---|---|---|
| exp930 ep2 | bin 9 (45.8 %) | **collapse** (4–6 m 단일 bin) |
| exp917 ep14 | bin 8–10 (heavy at 9, 34.0 %) | 매우 좁음 |
| exp931 ep12 | bin 8–11 spread (14–28 %) | **healthy** |
| exp932 ep8 | bin 7 (37.4 %) | mid concentrated |
| exp933 ep8 | bin 8 (59.4 %) | **strong collapse** |
| exp934 ep12 | bin 8 (36.8 %) but spread | **healthiest spread** |

→ raw_hit 가족이 *4–6번 bin (3–5 m)에 75 %+ 집중* — 단일 mid-bin commit. soft_hit이 rendered 가족 중 가장 넓은 분포 유지.

**entropy of weights (낮을수록 sharp)**:

| Cell | ent p10 | ent p50 | ent p90 |
|---|---:|---:|---:|
| exp917 | 0.30 | 0.40 | 0.48 |
| exp930 | 0.39 | 0.49 | 0.59 |
| exp931 | 0.46 | 0.64 | 0.77 |
| exp934 | 0.42 | 0.58 | 0.68 |
| exp933 (NaN 직전) | **0.56** | **0.72** | **0.80** |

→ raw_hit는 over-confident, survival은 NaN 직전 under-confident, soft_hit은 적당.

#### 4.1.C R4 per-bin GT-stratified 메트릭 (val 1-shot [haz] sliced)

근거리/원거리 영역에서 hazard renderer의 systematic bias를 직접 측정.

| Cell | [0.1,1) AR | [0.1,1) RMSE | [1,3) AR | [1,3) RMSE | [3,6) AR | [3,6) RMSE | [6,9.8) AR | [6,9.8) RMSE | [9.8,10) RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp917 | 0.361 | 0.369 | 0.304 | 0.729 | 0.568 | 2.453 | 0.728 | 5.180 | 8.630 |
| exp930 | 0.459 | 0.403 | 0.279 | 0.688 | 0.577 | 2.508 | 0.748 | 5.293 | 8.659 |
| exp931 | **1.353** | 1.187 | 0.246 | 0.618 | **0.232** | **1.172** | **0.436** | **3.294** | 6.801 |
| exp932 | 1.292 | 1.059 | **0.211** | **0.568** | 0.277 | 1.371 | 0.469 | 3.514 | 6.843 |
| exp933 | 1.213 | 1.096 | 0.223 | 0.595 | 0.269 | 1.340 | 0.462 | 3.500 | 7.175 |
| **exp934** | **1.032** | **0.959** | 0.213 | 0.573 | 0.288 | 1.431 | 0.468 | 3.547 | 7.042 |

**핵심 관찰 — hazard renderer의 inherent 한계**:
- **rendered 가족(931–934) 모두 [0.1,1) AR=1.0–1.4** — floor/ceiling 픽셀이 mid-bin (3–6 m)으로 끌려가 systematic underestimate. 분포-head expectation은 [0.1,1) AR≈0.5 (round-2 추정) — *분포-head보다 2~3× 더 나쁨*.
- **raw_hit 가족 [0.1,1) AR=0.36–0.46**: 우연히 mid-bin이 1 m 부근에 있어 작은 AR 보임 — 실은 **모든 픽셀이 단일 mid-bin에 collapse**해 floor가 우연히 낮게 예측됨. 5 m 픽셀에서는 8.6 RMSE로 폭발.
- **exp934 (soft_hit)이 rendered 가족 중 [0.1,1) AR best (1.032)** — α saturation 회피로 floor 픽셀이 더 자유롭게 가까운 bin에 분포.
- **모든 셀이 [9.8,10) bin에서 6.8–8.7 RMSE 동일** — `far_thresh=9.8` 마스크로 supervised되지 않은 영역.

#### 4.1.D R4 per-sample paired statistics (vs round-2 exp907 분포-head expectation, n=3192)

| Cell | mean Δ ABS_REL | median Δ ABS_REL | std | frac improved (Δ<0) | sign test |
|---|---:|---:|---:|---:|---|
| exp907_TESTmedian (참고) | −0.0503 | −0.0271 | 0.0816 | **76.0 %** | overwhelming improvement |
| **exp934 (R4 best)** | **−0.0183** | **−0.0125** | 0.1764 | **59.0 %** | sign test p < 1e-7 ★ |
| exp912 (same-env scalar) | −0.0356 | −0.0127 | 0.1618 | 57.7 % | improvement |
| exp915 (R2 hazard no-hit) | −0.0317 | −0.0179 | 0.1670 | 61.4 % | improvement |
| exp930 (raw_hit smooth) | −0.0638 | **+0.0160** | 0.2980 | 46.1 % | **mean misleading** |
| exp917 (raw_hit strong) | −0.0414 | **+0.0376** | 0.2959 | 41.5 % | **mean misleading** |
| exp931 (event_nll+free) | +0.0279 | +0.0208 | 0.2067 | 36.7 % | regression |
| exp932 (event_nll only) | +0.0305 | +0.0150 | 0.1629 | 39.4 % | regression |
| exp933 (survival) | +0.0028 | +0.0009 | 0.1783 | 49.4 % | tied |

**핵심**:
- **exp934의 ABS_REL 우위는 통계적으로 유의** (sign test 59.0 % win, p < 1e-7) — 그러나 effect size median Δ = −0.012는 **practically marginal**.
- **exp930·exp917의 mean ABS_REL 개선은 misleading**: mean Δ는 negative지만 median Δ는 positive — 일부 outlier 샘플의 ABS_REL을 줄이지만 majority 샘플에서 더 나쁨. **mean ABS_REL 단독 인용 위험**.

#### 4.1.E R4 per-scene metrics (9 scenes, scene-disjoint test split)

| | 8WUmhL | EDJbRE | HxpKQy | Z6MFQC | gTV8FG | **pLe4wQ** | q9vSo1 | sT4fr6 | uNb9QF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp907 (분포-head) | 0.4514 | 0.3841 | 0.4331 | 0.4313 | 0.5171 | **0.7374** | 0.3985 | 0.3573 | 0.4608 |
| exp907_TESTmedian | 0.4024 | 0.3321 | 0.3642 | 0.4061 | 0.4593 | **0.6618** | 0.3579 | 0.3223 | 0.4091 |
| exp912 (scalar) | 0.4300 | 0.3498 | 0.3306 | 0.4014 | 0.4672 | 0.7016 | 0.3773 | 0.3507 | 0.4260 |
| exp934 (soft_hit) | 0.4252 | 0.3517 | 0.3921 | 0.3997 | 0.4865 | 0.7489 | 0.3891 | 0.3615 | 0.4392 |
| exp930 (raw_hit) | 0.4185 | 0.3487 | 0.3532 | 0.4067 | 0.4133 | **0.5072** | 0.3920 | 0.3708 | 0.4056 |
| exp917 (raw_hit strong) | 0.4276 | 0.3703 | 0.3791 | 0.4405 | 0.4296 | 0.5524 | 0.4092 | 0.3878 | 0.4217 |

**관찰**:
- **pLe4wQe7qrG가 모든 셀의 worst scene** (0.55–0.80). 이 scene이 전체 ABS_REL의 6–17 % 비중.
- **exp907_TESTmedian이 9 scene 중 8개에서 best**. pLe4wQ에서도 0.6618로 best.
- **exp930 (raw_hit)이 pLe4wQ에서 0.5072로 *눈에 띄게 좋음***: 다른 8 scene은 평균 수준이지만 hard scene에서 우연한 게인. **scene-mean ↔ per-sample 비교에서 부호가 다른 이유**.

#### 4.1.F R4 sample-level prediction correlation

per-sample ABS_REL의 Pearson correlation (n=3192):

| Pair | r |
|---|---:|
| exp907 expectation ↔ exp907_TESTmedian | **0.984** |
| exp907 ↔ exp934 | 0.900 |
| exp912 (scalar) ↔ exp934 | **0.930** |
| exp907 ↔ exp930 (raw_hit) | 0.63–0.80 |
| exp907 ↔ exp917 (raw_hit strong) | 0.71–0.78 |

**해석**:
- **stable hazard 가족(931–934) 모두 r > 0.89로 거의 같은 prediction** — *동일 ceiling을 다른 path로 도달*. Hazard rendering 자체가 분포-head + median과 **중첩된 capacity** 통계적 확인.
- **exp934 ↔ scalar (0.930) > exp934 ↔ 분포-head (0.900)**: soft_hit이 single-mode commit으로 분포-head의 multi-mode 정보 잃음.

#### 4.1.G R4 가설 검증 (rigorous review)

| Hypothesis | Verdict | 근거 |
|---|---|---|
| **H_R4_1**: α-direct BCE saturation은 raw_hit 폭발의 root cause | **Strongly Supported** | frac>.99 = 0.30 (raw_hit) ↔ 0.000 (soft_hit) 단조 chain |
| **H_R4_2**: Smooth ramp만으로 raw_hit는 살아난다 | **Rejected** | exp930 ep2부터 saturation 발생, RMSE 1.63 |
| **H_R4_3**: Rendered-quantity supervision은 saturation 우회 | **Supported (조건부)** | 931–934 RMSE 1.22 회복; 단 exp933 NaN 폭발 |
| **H_R4_4**: Hazard rendering이 분포-head 위에 capacity 추가 | **Rejected** | r > 0.89 same-prediction; 같은 ceiling |
| **H_R4_5**: Free loss는 δ1을 향상시킨다 | **Weakly Supported** | exp931 vs 932: δ1 +0.010 (noise floor 1.7×) |
| **H_R4_6**: Strong supervision (λ_hit↑)은 raw_hit 회복 | **Rejected** | exp917 RMSE 1.6296 > exp930 1.6270 (worse) |

**핵심 메커니즘 chain**:
```
raw_hit BCE → saturation (α=1) → ∂α/∂logit ≈ 0 → renderer stuck
            → 모든 픽셀 단일 mid-bin commit → narrow prediction range
            → no high-quality predictions (ABS≤0.25 in 2 % vs 23 % normal)
            → mediocre ABS_REL but high RMSE + low δ1

soft_hit (target=0.75) → α capped at 0.75 → ∂α/∂logit ≈ 0.19 → grad alive
            → pixels free to spread → wider prediction range (≤0.25 in 26 %)
            → ABS_REL noise-floor improvement
            → RMSE/δ1 same as 분포-head ceiling
```

#### 4.1.H R4 failure case 카테고리

| Failure 카테고리 | 발생 셀 | 진단 |
|---|---|---|
| Depth range compression | exp930, exp917 | argmax_bin_hist 75 %+ at single bin |
| Far-region degradation | exp930, exp917 | bg_weight=0 + saturated commit, [9.8,10) RMSE 8.7 |
| Near-object error (floor/ceiling) | exp931–934 | rendered renderer가 floor를 mid-bin으로 끌어당김, [0.1,1) AR=1.0–1.4 |
| Training instability | exp933 | log T_j underflow → epoch 9 NaN |
| Auxiliary loss conflict | exp917 | depth loss reverse @ ep4 (λ_hit warmup 종결) |

### 4.2 R5A Soft-Quantile sweep (n2 bs=48 DP, 20 ep, range head, R_BASE)

`R_BASE = --depth-head-type range --range-num-bins 32 --range-bin-spacing log --range-min-depth 0.1 --range-max-depth 10.0 --range-soft-label-sigma 0.14 --range-output-mode expectation --lambda-range-nll 1.0 --lambda-berhu 1.0 --lambda-silog 0.5 --erp-cos-lat-weight --erp-far-mask`

| Exp | q | τ | λ_sq | best_ep | ABS_REL | RMSE | δ1 | δ2 | δ3 | Log10 | MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp936 (anchor) | 0.50 | 0.05 | 0.25 | 06 | 0.4726 | 1.2158 | 0.4957 | 0.7112 | 0.8322 | 0.1559 | 0.7839 |
| exp937 | 0.45 | 0.05 | 0.25 | 06 | 0.5150 | 1.2175 | 0.4920 | 0.7039 | 0.8236 | 0.1599 | 0.7988 |
| exp938 | 0.55 | 0.05 | 0.25 | 12 | 0.4817 | 1.2304 | **0.5070** | 0.7151 | 0.8306 | 0.1566 | 0.7879 |
| exp939 | 0.50 | 0.03 | 0.25 | 08 | 0.4585 | 1.2303 | 0.4982 | 0.7113 | 0.8331 | 0.1559 | 0.7853 |
| exp940 | 0.45 | 0.03 | 0.25 | 08 | 0.5230 | 1.2170 | 0.4992 | 0.7142 | 0.8288 | 0.1585 | 0.7966 |
| exp941 | 0.50 | 0.05 | 0.50 | 10 | 0.5050 | **1.2039** | 0.4978 | 0.7126 | 0.8298 | 0.1572 | 0.7880 |
| **exp942** | 0.45 | 0.05 | **0.50** | 10 | 0.5239 | **1.2030** | 0.4923 | 0.7108 | 0.8266 | 0.1589 | 0.7919 |
| exp943 | 0.50 | 0.03 | 0.50 | 06 | 0.4879 | 1.2149 | 0.4852 | 0.7066 | 0.8292 | 0.1584 | 0.7941 |

**verdict**:
- λ_sq=0.5 (exp941/942/943) 셀이 **RMSE 1.20–1.21로 28셀 중 최저** — 원거리 outlier 누름 효과 명확.
- **그러나 ABS_REL이 0.49–0.52로 동반 악화** — soft-quantile depth가 학습 단계에서 squared bin loss를 강조 → 근거리 미세 정렬 약화.
- λ_sq=0.25 (exp936/939) 셀은 ABS 0.46–0.47로 round-2 exp907 (0.4705)와 동률, RMSE는 비슷 — soft-quantile 효과 미미.
- q ∈ {0.45, 0.50, 0.55} sweep에서 q=0.55 (exp938)이 δ1 0.5070으로 가장 강함 — **분포의 약간 right-skewed mode 추출이 confidence-thresholded metric에 우호적**.
- τ ∈ {0.03, 0.05} 차이는 noise floor 안.

**가설 검증**:
- ✅ **soft-quantile (RMSE에 우호적인 신호) 효과 확인** (λ↑ → RMSE↓ 단조).
- ❌ **그러나 ABS_REL과 동시 개선 못 함** — trade-off 곡선 위 이동만.
- ⚖️ Round-2의 "median이 ABS_REL을 더 떨어뜨린다"는 inference-only 관찰을 train-time으로 옮긴 효과는 부분 재현(λ=0.25 셀에서) — 그러나 **RMSE/δ1 손해 없는 ABS_REL 게인은 어디에도 없음**.

### 4.3 R5B Spherical-SH sweep (n2 bs=48 DP, 20 ep, range head, R_BASE)

| Exp | L | λ_SH | log_depth | best_ep | ABS_REL | RMSE | δ1 | δ2 | δ3 | Log10 | MAE |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp944 | 2 | 0.02 | true | 08 | 0.4928 | 1.2048 | 0.5014 | **0.7197** | 0.8339 | 0.1549 | 0.7770 |
| exp945 | 2 | 0.05 | true | 06 | 0.5222 | 1.2178 | 0.4995 | 0.7077 | 0.8260 | 0.1587 | 0.7964 |
| **exp946** ★ | **2** | **0.10** | **true** | **08** | **0.4413** | **1.2208** | **0.5019** | **0.7150** | **0.8320** | **0.1542** | **0.7755** |
| exp947 | 3 | 0.02 | true | 08 | 0.5082 | 1.2327 | 0.4953 | 0.7055 | 0.8248 | 0.1596 | 0.8048 |
| exp948 | 3 | 0.05 | true | 10 | 0.4834 | 1.2053 | 0.5008 | 0.7167 | 0.8322 | 0.1552 | 0.7787 |
| exp949 | 2 | 0.02 | false (lin) | 10 | 0.4622 | 1.2200 | **0.5079** | 0.7190 | 0.8334 | 0.1545 | 0.7793 |

**verdict**:
- **exp946 (L=2, λ_SH=0.10, log_depth=true)이 28셀 중 유일하게 7개 metric 전부에서 echodiff exp958을 능가**. 비교: exp946 (0.4413/1.2208/0.5019/0.7150/0.8320/0.1542/0.7755) vs exp958 (0.4463/1.2611/0.4983/0.7067/0.8272/0.1593/0.7965). 모든 차이가 작지만 동일 방향.
- **λ_SH curve 비단조**: 0.02 (exp944) → 0.05 (exp945) → 0.10 (exp946)에서 0.05가 오히려 ABS 0.5222로 worst → λ가 너무 작으면 SH 신호가 noise에 묻히고, 너무 크면 BerHu와 충돌하지만 **λ=0.10 부근의 sweet spot 존재**.
- **L=3 (exp947, exp948)은 L=2 대비 개선 없음** — 저-차 (L=2: 9 coeffs) 이상의 SH는 데이터에서 supervisable하지 않음.
- **log_depth=false (exp949)**가 log_depth=true (exp944)보다 ABS 0.4622 (vs 0.4928) 우위 — log space의 SH 매칭이 항상 더 나은 것은 아님; lin-depth space에서도 충분.
- exp946 SH=0.10 + logd 셀이 **R5 SOTA**, **but round-0 exp11 (0.4300/1.1060) 대비 ABS +0.011, RMSE +0.115 손해 그대로**.

**가설 검증**:
- ✅ **저차 SH coeff matching이 분포 head의 전역 분포 정렬을 향상시킴** — δ2/δ3/Log10/MAE 일관 우위.
- ✅ **high-λ SH가 ABS_REL에도 우호적** (exp946 ABS 0.4413, R5 best).
- ⚖️ 그러나 **RMSE는 SH 단독으로 유의 개선 못 함** — exp946 RMSE 1.2208은 R5A λ_sq=0.5 (exp942 1.2030)보다 +0.018 손해.
- ❌ **여전히 Round-0 RMSE 1.10에 못 미침**.

### 4.4 R5C Combo (sq + SH) (n2 bs=48 DP, 20 ep)

| Exp | sq config | SH config | best_ep | ABS_REL | RMSE | δ1 | δ2 | δ3 | Log10 | MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| exp950 | q0.5/τ0.05/λ0.25 | L2/λ0.02/logd | 08 | 0.5158 | 1.2141 | 0.4978 | 0.7077 | 0.8265 | 0.1586 | 0.7925 |
| exp951 | q0.5/τ0.05/λ0.25 | L2/λ0.05/logd | 10 | 0.5003 | 1.2158 | 0.4971 | 0.7117 | 0.8281 | 0.1576 | 0.7925 |

**verdict — 시너지 없음**:
- exp950 (sq + SH λ=0.02): ABS 0.5158 — 단독 R5A anchor (exp936 0.4726) 보다 ABS +0.043 악화, 단독 R5B λ=0.02 (exp944 0.4928) 보다 ABS +0.023 악화. 즉 **두 loss 결합이 양쪽 단독보다 모두 나쁨**.
- exp951 (sq + SH λ=0.05): ABS 0.5003 — exp950보다 살짝 회복했지만 단독 R5B (exp945 0.5222)와 비슷한 위치.
- δ1 0.497–0.498로 두 단독 모드와 동률, **추가 게인 없음**.

**해석**:
- soft-quantile은 학습 시 expectation 위에 sub-bin shift를 주는 보조 신호, SH는 픽셀 간 전역 분포를 정렬하는 보조 신호. 두 신호가 같은 학습 step의 gradient를 다른 방향으로 끌어당겨 **상호 간섭**.
- 다음 라운드에서 두 신호를 합치려면 (a) 둘 중 하나의 λ를 매우 작게 잡아 soft regularizer로 쓰거나, (b) 학습 단계 분리 (curriculum: SH first → soft-quantile later) 가 필요할 듯.

### 4.5 R5F 40-epoch paired baseline (n9 bs=32 single-GPU)

| Exp | head | range_output_mode | epochs | best_ep | ABS_REL | RMSE | δ1 | δ2 | δ3 | Log10 | MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **exp958** | scalar (echodiff 본가) | — | 40 | **16** | 0.4463 | 1.2611 | 0.4983 | 0.7067 | 0.8272 | 0.1593 | 0.7965 |
| exp959 | scalar (echorange backbone) | — | 40 | 12 | 0.4873 | 1.2557 | 0.4860 | 0.7014 | 0.8257 | 0.1621 | 0.8126 |
| exp960 | range | expectation | 40 | 06 | 0.4951 | 1.2160 | 0.4962 | 0.7126 | 0.8303 | 0.1570 | 0.7903 |
| **exp961** ★ | **range** | **median** | 40 | 10 | **0.4520** | **1.2444** | **0.5199** | **0.7231** | **0.8319** | **0.1562** | **0.7850** |

**verdict**:
- **exp958 (echodiff scalar 40ep) best ckpt가 ep16에서 멈춤** → 학습이 본질적으로 ep16 부근에서 plateau, 추가 24 epoch 무용. round-2 exp912 (20ep, 0.4349/1.2432)와 비교 — **ABS·RMSE 모두 약간 손해**. 즉 **40 epoch budget은 round-0 격차를 닫지 못함, 오히려 약간 후퇴**.
- exp959 (echorange backbone scalar)는 exp958보다 ABS +0.041 손해 — round-2 exp912에서 이미 부재했던 echorange-vs-echodiff scalar gap을 다시 확인. echorange backbone의 scalar 모드 구현이 echodiff 본가와 미세 다름 가능 (round-2에서도 미해결).
- exp960 (range expectation 40ep) ABS 0.4951 — round-2 exp907 (20ep, 0.4705)보다 ABS +0.025 손해. epoch 늘려도 ABS_REL 격차 좁혀지지 않음 → **분포 head ABS_REL 격차 source가 underfit이 아님 명확화**.
- **exp961 (range median 40ep)이 28셀 δ1 챔피언 0.5199** — round-2의 inference-only median (exp907_TESTmedian, 0.5129)보다도 강함, 그리고 train-time median (exp910 6/20 ep 미완) 보다 강함. **train-time median이 δ1을 끌어올린다는 round-2 가설 확정 검증**.
- exp961 RMSE 1.2444 — round-2 exp907_TESTmedian (1.2765)보다 우위, 그러나 round-0 exp11 (1.1060) 대비 +0.138 손해.

**가설 검증**:
- ✅ **train-time median이 δ1에 강한 우위** (round-2 exp907_TESTmedian 0.5129 < exp961 0.5199).
- ❌ **40 epoch budget으로 ABS·RMSE round-0 격차 안 좁혀짐** — round-0 격차의 source는 underfit이 아니라 학습 cfg (lr schedule, normalization, freeze 정책 등) 다른 곳.
- ⚖️ **echorange backbone vs echodiff 본가 scalar gap** (exp959 0.4873 vs exp958 0.4463) — round-2에서 exp912 vs exp900에서도 발견됐던 패턴, 미해결.

### 4.6 R5G·R5H Br=20 저-비닝 (n9 bs=32, 20 ep)

| Exp | family | bins | best_ep | ABS_REL | RMSE | δ1 | δ2 | δ3 | Log10 | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp962 | R5G R20 sq (q0.5/τ0.05/λ0.25) | 20 | 10 | 0.4686 | 1.2218 | 0.5034 | 0.7151 | 0.8321 | 0.1556 | 0.7838 |
| exp963 | R5H R20 SH (L2/λ0.02 logd) | 20 | 10 | 0.5221 | 1.2267 | 0.5010 | 0.7119 | 0.8261 | 0.1595 | 0.7996 |

**verdict**: Br=20은 Br=32 anchor (exp936/exp944) 대비:
- exp962 vs exp936 (R20 sq vs R32 sq, 같은 cfg): ABS +0.040 (0.4686 vs 0.4726 → 미세 우위), 그러나 RMSE +0.060 (1.2218 vs 1.2158, 차이 작음) — **거의 동률**.
- exp963 vs exp944 (R20 SH vs R32 SH, 같은 λ_SH=0.02): ABS +0.029 손해 (0.5221 vs 0.4928), δ1 거의 동일. 명확한 후퇴.

**verdict**: **Br=20은 cfg 단순화 외 게인 없음, Br=32–40이 적정**. R5J cylindrical bin-axis 셀(exp966–967)은 미실행으로 보류.

### 4.7 4-best ckpt 비교 (best_score vs best_absrel/rmse/delta1)

선택 셀들에 대해 동일 학습의 4-best ckpt를 모두 test에 돌려본 결과(전체 70 entries는 부록에):

| Exp | tag | ABS_REL | RMSE | δ1 | composite (0.7·R + 0.3·A) |
|---|---|---:|---:|---:|---:|
| exp946 | best_score | **0.4413** | 1.2208 | 0.5019 | **0.9870** |
| exp946 | best_rmse | 0.5405 | 1.2272 | 0.5038 | 1.0212 |
| exp946 | best_delta1 | 0.5320 | 1.2683 | 0.5048 | 1.0474 |
| exp961 | best_score | **0.4520** | **1.2444** | **0.5199** | **1.0067** |
| exp961 | best_absrel | 0.4593 | 1.2607 | 0.5163 | 1.0203 |
| exp961 | best_rmse | 0.5301 | 1.2259 | 0.5192 | 1.0172 |
| exp961 | best_delta1 | 0.4520 | 1.2444 | 0.5199 | 1.0067 |
| exp958 | best_score | 0.4463 | 1.2611 | 0.4983 | 1.0166 |
| exp958 | best_rmse | 0.5564 | 1.2685 | 0.4688 | 1.0549 |
| exp958 | best_delta1 | 0.4989 | 1.2780 | 0.4964 | 1.0443 |

**verdict**:
- **best_score (composite 0.7·RMSE + 0.3·ABS_REL)가 거의 항상 가장 균형 잡힌 결과**.
- 단일 metric ckpt는 val에서 그 metric을 잡되 **test에서 다른 metric을 큰 폭으로 손해 보는 overfit 패턴** (예: exp946 best_delta1: δ1 +0.003 미세 우위 / ABS +0.091 큰 손해).
- 예외: **exp961 best_delta1 = best_score** (동일 epoch 10에서 두 best 동시 달성 → median decoding이 δ1과 composite을 함께 잡음).
- val set 작아 (2 951) **ckpt selection 자체가 noise를 amplify** — 또는 학습이 본질적으로 trade-off 곡선 위에서만 움직이기 때문에 단일 metric을 잡으려 하면 다른 metric을 희생.

**라운드 결론 (4-best ckpt UX)**: 운영상 **best_score 단일로 충분**. 4-best는 R5 라운드 전반의 metric profile 가시성에는 도움이 됐지만, 실제 셀 비교에는 best_score만 사용. 다음 라운드부터는 4-best 저장 유지 + test는 score 단일이 표준.

---

## 5. 핵심 진단 — Trade-off 곡선

### 5.1 (ABS_REL, RMSE) 평면 분석

28 best_score 셀을 (ABS_REL, RMSE)로 흩어놓으면:

```
RMSE
1.63 │ exp930  exp917
     │
1.22 │ exp934                     exp960  exp961
     │ exp948  exp944  exp946     exp936  exp939
     │ exp950  exp942  exp941     exp962  exp963
     │ exp943  exp951  exp947     exp938
     │ exp949  exp945  exp958     ...
1.20 │ exp942
     │
1.10 │ ★ round-0 exp11 (목표)
     └────────────────────────────────────────────  ABS_REL
       0.41   0.43   0.45   0.47   0.49   0.51  0.53
```

세 그룹 분포:
1. **저-RMSE 그룹** (R5A λ=0.5: exp941/942/943): RMSE 1.20–1.21 / **ABS 0.49–0.52**
2. **저-ABS_REL 그룹** (R5B SH high-λ: exp946; R4 soft_hit: exp934): **ABS 0.44–0.45** / RMSE 1.22–1.23
3. **고-δ1 그룹** (R5F median: exp961): δ1 0.5199 / ABS 0.45 / RMSE 1.24

**ABS_REL과 RMSE 두 metric이 동시에 baseline 대비 의미 있게 (둘 다 0.01 이상) 좋아진 셀**:

| 비교 baseline | (ABS_REL, RMSE) | 두 metric 동시 우위 셀 |
|---|---|---|
| **echodiff exp11 (round-0 SOTA)** | (0.4300, 1.1060) | **0건** |
| **exp912 (round-2 same-env scalar)** | (0.4349, 1.2432) | **0건** (ABS<0.4349 AND RMSE<1.2432 만족 셀 없음) |
| exp958 (R5F same-env 40ep echodiff) | (0.4463, 1.2611) | **1건** — exp946 (0.4413, 1.2208), 두 차이 noise floor 안 |
| exp920 (round-2 n2 bs=48 echodiff) | (0.4884, 1.2212) | exp946 등 다수 |

→ **echodiff 환경 best 대비 두 metric 동시 우위는 사실상 불가능. 가장 강한 baseline (round-0 exp11) 대비는 0건**.

### 5.2 왜 trade-off가 풀리지 않는가 — 구조적 이유

ABS_REL과 RMSE는 깊이 분포의 다른 영역을 본다:
- **ABS_REL** = mean(|d̂−d|/d) → 분모가 작은 **근거리(small d)** 픽셀에 가중. matterport3d radial GT의 mode 1–3 m 영역이 dominate.
- **RMSE** = √mean((d̂−d)²) → **원거리(large d) outlier**에 가중. 6 m+ 영역의 sparse pixel이 squared term으로 amplify.

현재 R5 head 디자인의 영역별 효과:
1. **R5A Soft-Quantile**: bin pinball loss를 quadratic으로 키워 원거리 outlier 누름 → RMSE↓, 근거리 미세 정렬 약화 → ABS_REL↑.
2. **R5B Spherical-SH**: 구면 저차 SH coefficient 매칭 → 전역 분포(scene-mean, low-frequency anisotropy) 정렬 → log-depth space에서 GT 따라가 ABS_REL/MAE/Log10 우호.
3. **R5F median 디코딩**: multi-modal posterior에서 robust 추정 → δ1 큰 게인, **그러나 median은 squared loss minimizer 아님** → RMSE 손해.
4. **분포 head 자체**: bin discretization이 expectation/median 어느 디코딩이든 **±(bin width / 2) bias 잠재** → echodiff scalar(bias=0) 대비 RMSE 구조적 손해.

→ **현재 28셀 어느 head 디자인도 두 영역(근거리·원거리) 손실을 동시에 줄이는 신호를 갖고 있지 않음**. 모두 **trade-off 곡선 위 이동만 가능**.

### 5.3 곡선을 바깥으로 미는 디자인 후보 (다음 라운드 R6)

근거리·원거리 손실을 **동시에** 보는 신호가 필요. 후보 4종:

1. **분포 + scalar multi-task head (★ 우선 권장)**: 분포 head (R5B Soft-SH) + 가벼운 scalar regression head를 같은 backbone에 병렬 부착. scalar는 BerHu+L2 가중합으로 RMSE/ABS 책임. 분포는 δ1·MAE·Log10 책임. 추론 시 scalar의 분포 confidence 기반 게이팅 또는 단순 가중평균. 학습 비용 작음(추가 head 1×1 conv).
2. **Sub-bin residual head**: 분포 head expectation 위에 sub-bin scalar residual을 더하는 small head. residual = (target − bin-decoded depth)로 직접 supervised. **Bin discretization bias 제거**하면서 δ1 게인 보존. 가장 가벼운 변형.
3. **Loss 결합 — explicit ABS_REL surrogate**: 현재 loss = soft_NLL + BerHu + λ·SILog. 여기에 명시적 `mean(|d̂−d|/d)` (ABS_REL surrogate) term 추가. 단 BerHu/SILog와 partial overlap → 한계 효과 작을 가능성. multi-task의 보조 수단으로.
4. **Ordinal head**: bin σ_j = P(D > r_j), cumulative target. `expected_depth = ∫σ_j · Δr`이 squared loss minimizer에 더 가까운 형태로 RMSE 우호 가능. round-4 survival(exp933 1.2202)이 ordinal hazard variant — 본격 ordinal head 구현 시 사전 비교 가능.

---

## 6. Round-0 RMSE 1.1060 격차 미해결 — Priority-0

### 6.1 격차의 정량

| 셀 | 환경 | epochs | RMSE | round-0 격차 |
|---|---|---:|---:|---:|
| **exp11 (round-0 SOTA, 외부)** | bs=32 lr=1e-4 | ? | **1.1060** | 0 |
| exp13 (round-0 alt) | bs=16 lr=1e-4 | ? | 1.1134 | +0.007 |
| exp912 (round-2 same-env echodiff 본가) | bs=32 single-GPU | 20 | 1.2432 | **+0.137** |
| exp958 (R5F same-env echodiff 본가 40ep) | bs=32 single-GPU | 40 | 1.2611 | **+0.155** |
| exp920 (round-2 n2 echodiff 본가 seed-2) | bs=48 DP | 20 | 1.2212 | +0.115 |
| exp942 (R5 best RMSE) | bs=48 DP | 20 | 1.2030 | +0.097 |

→ **head 디자인 이전 baseline 수준에서 +0.10~0.16 RMSE 격차**가 들어가 있음. R5 어느 셀도 RMSE 1.20 미만 못 감.

### 6.2 round-2 검증·기각 가설

- ❌ **DP vs single-GPU**: exp912(single-GPU bs=32, RMSE 1.2432) ↔ exp920(DP bs=48, RMSE 1.2212) 거의 동률 — DP는 격차 원인 아님.
- ❌ **Epoch budget**: exp958 40 epoch도 best ckpt ep16, RMSE 1.2611로 round-2 exp912 (20ep, 1.2432)보다도 손해. epoch 길이가 격차 source 아님.
- ❌ **Backbone capacity**: 우리 echodiff/echorange는 round-0 exp11과 동일 EcoDepth + Wav2Vec2 backbone, 동일 cfg.

### 6.3 미검증 가설 (다음 라운드 진단 우선순위)

- **(a) lr schedule**: round-0이 cosine warmup 사용했을 가능성. 우리는 step decay만. **1-cell 진단**: exp958 cfg + cosine warmup + same lr=1e-4로 1셀 학습.
- **(b) Wav2Vec2 freeze 정책**: round-0이 wav2vec2를 학습 후반 freeze 했을 가능성. **1-cell 진단**: wav2vec2_freeze_after_epoch=10 추가.
- **(c) depth normalization scale**: `depth_norm=true`일 때 BerHu/SILog가 [0,1] scaled depth 위에서 계산되고, head는 raw m 출력 → scaling 처리 차이 가능. **1-cell 진단**: depth_norm=false로 동일 학습.
- **(d) gradient accumulation / effective bs**: round-0 exp11이 effective bs=64 (accum) 사용했을 가능성. **1-cell 진단**: bs=16 + accum 4.
- **(e) data cache hash**: `samples_test_erp_e2314b68a4f5.json` 변경으로 split 미세 변동 가능. **1-cell 진단**: round-0 학습 시점 cache hash 확인.

이 5 후보 중 (a)(c) 가 가장 의심스러움 (lr cosine warmup은 EcoDepth 류 backbone에 표준; depth_norm scaling은 round-0과 round-1 사이 코드 변경 가능성).

### 6.4 Round-0 격차 해소까지 모든 R6 head 디자인 비교가 noise floor 안

Round-2 audit의 noise floor (test ABS_REL ±0.05, RMSE ±0.02, δ1 ±0.006)를 적용하면, **R5 28셀의 가장 강한 비교 (exp946 vs exp958)도 RMSE 차이 −0.040 / ABS 차이 −0.005**가 noise floor에 걸침. **격차를 닫지 않으면 어떤 새 head 디자인 비교도 의미 없음**. → R6의 priority-0.

---

## 7. 패밀리별 최종 verdict

| 패밀리 | 대표 | 게인 정도 | 다음 라운드 처리 |
|---|---|---|---|
| R3·R4 hazard | exp934 soft_hit | softmax baseline 회복만 | 폐기. R5 closure 1셀 유지 |
| R5A Soft-Quantile | exp936/exp942 | RMSE ↔ ABS_REL trade-off 위 이동만 | λ=0.25 anchor 1셀만 R6 비교용 유지 |
| **R5B Spherical-SH** | **exp946 L2 λ0.10 logd** | **유일하게 7-metric all-win vs exp958** | **R6 메인 cfg** |
| R5C Combo (sq + SH) | exp950/exp951 | 시너지 없음, 단독보다 후퇴 | 폐기 |
| **R5F median 디코딩** | **exp961 R40 median** | **δ1 R5 챔피언 (0.5199)** | **R6 모든 cfg에 median 옵션 default-on 검토** |
| R5G·R5H R20 | exp962/exp963 | Br=32 대비 후퇴 | 폐기 |
| R5D Cylindrical | (미실행) | — | R6 priority-3 검토 |

---

## 8. 다음 라운드 (R6) 권고

### Priority 0 — Round-0 baseline 재현 진단
- **R6-A1**: exp958 cfg + cosine warmup (lr=1e-4 → 0.5e-4 over 20ep). 1셀.
- **R6-A2**: exp958 cfg + depth_norm=false. 1셀.
- **R6-A3**: exp958 cfg + wav2vec2 freeze after ep10. 1셀.
- 3셀 중 1개라도 RMSE 1.10대 진입하면 그 cfg를 R6의 새 baseline으로 채택.
- 학습 budget: 3 × 5h @ 40ep = 15h.

### Priority 1 — Trade-off 곡선 외향 이동: Multi-task head
- **R6-B1**: exp946 cfg + scalar regression head 병렬 (1×1 conv on decoder feature). loss = soft_NLL + λ_SH·SH + λ_scalar·(BerHu + L2). Inference: scalar prediction. 1셀.
- **R6-B2**: exp946 cfg + sub-bin residual head (분포 expectation에 ±(bin width / 2) residual scalar). loss + λ_residual·BerHu(residual). Inference: expectation + residual. 1셀.
- 두 cell 모두 R6-A 새 baseline 위에 학습. RMSE 1.10대 진입 + ABS 0.43대 진입을 동시에 달성하는 것이 목표.
- 학습 budget: 2 × 2.5h @ 20ep = 5h.

### Priority 2 — 이미 살아 있는 두 축의 가산 검증
- **R6-C1**: exp946 cfg (R5B SH λ=0.10 logd) + range_output_mode=median (학습 시 median decoding). 1셀.
- **R6-C2 (test-only)**: exp946 ckpt를 그대로 `--range-eval-mode q50`/`temp075`로 재평가. 학습 비용 0, 가산 효과 빠르게 확인.
- 학습 budget: C1 2.5h @ 20ep, C2 0h.

### Priority 3 — Cylindrical bin-axis 검토 (R5D 보류분)
- **R6-D1**: exp946 cfg + range_bin_axis=horizontal. 1셀.
- **R6-D2**: exp946 cfg + range_bin_axis=z. 1셀.
- ERP polar 픽셀의 bin 의미가 horizontal/z에서 더 명확 → 원거리 RMSE 게인 가능성 추정. 학습 budget: 2 × 2.5h.

### 합산 budget
- Priority 0–3 총 8셀, 약 25h. 두 서버 분산 시 야간 1회 실행 가능.

---

## 9. Confounder 및 Validity 한계

본 라운드 결론을 paper-grade로 hedge할 때 명시적으로 인지해야 하는 한계:

| # | Sanity check | 본 라운드 상태 | 영향 |
|---|---|---|---|
| 1 | Same train/val/test split | ✓ — `samples_test_erp_e2314b68a4f5.json`, n=3192 일관 | OK |
| 2 | **Same random seed** | ✗ — `PYTHONHASHSEED=1`만 wired, `torch.manual_seed` train.py에 미적용. **모든 셀 non-deterministic** | seed-only variance ≈ 0.01–0.02 ABS_REL 추정. **0.02 단위 차이는 모두 tentative** |
| 3 | Same number of epochs | mixed — bs=48 셀 모두 20ep, exp958/959 40ep, exp930 5ep만, exp933 ep8 best (NaN), exp917 14ep best (절단) | exp930/exp933의 비교는 학습 부족 가능성 |
| 4 | Same batch size and lr | bs=48 vs bs=32 mixed (n2 ↔ n9), lr=1e-4 일관 | bs effect는 exp912↔exp920에서 noise 안으로 검증됨 |
| 5 | Same checkpoint selection rule | ✓ — best_score = 0.7·RMSE + 0.3·ABS_REL (R5에서 4-best로 분리 저장 추가) | OK |
| 6 | Same input preprocessing | ✓ — same backbone, same dataset, same STFT (n_fft=512, hop=160) | OK |
| 7 | Same depth normalization | ✓ — depth_norm=True, max_depth=10 | OK |
| 8 | Same loss weighting scale | ✓ — BerHu=1.0, SILog=0.5 (round-2 audit fix 후 일관) | OK |
| 9 | Same evaluation script | ✓ — test.py + compute_errors() | OK |
| 10 | Same masking | ✓ — far_thresh=9.8, erp_far_mask, erp_cos_lat_weight 적용 | OK |
| 11 | ERP boundary handling | ✓ — wraparound 미적용 (round-2 §2 결정) | OK |
| 12 | Multi-seed variance 측정 | ✗ — 모든 셀 single seed | 가장 큰 결함. **R6에 priority 6로 추가** |
| 13 | Cross-validation | ✗ — 단일 split | 0.01 단위 차이는 다른 split에서 뒤집힐 가능성 |

**Validity 한계 명시**:
- 본 라운드의 noise floor (round-2에서 측정): test ABS_REL ±0.05, RMSE ±0.02, δ1 ±0.006.
- 본 라운드의 effect size:
  - **Floor 초과 (strong)**: hazard 폭발 (RMSE +0.4), δ1 분포-우위 (+0.020), median ABS −10.7 %, **exp946 RMSE 게인 −0.040 (vs exp958, floor 2×)**.
  - **Floor 안 (tentative)**: R5A·R5B λ-curve 미세 차이, exp934 ABS_REL 게인 (median Δ −0.012, sign test에서 통계적 유의지만 effect size noise 안), R5G·R5H 후퇴.
- **분포-head의 일관 우위는 δ1 한 차원** — round-1, 2, 3 모두 +1.5–3pt δ1 우위. ABS_REL/RMSE는 동률 또는 열세.

**Paper에서 claim 가능 / 불가 분류**:
- ✅ **Strong, claimable**: "분포-head + median operator는 분포-head + expectation보다 ABS_REL을 10.7 % 상대 줄인다" (76 % 샘플 개선, n=3192). caveat: "in this setup", multi-seed 미검증.
- ✅ **Strong, claimable**: "α-direct BCE supervision은 sigmoid saturation으로 학습이 깨지며 rendered-quantity NLL 또는 soft-target BCE로 우회 가능". round-4 [haz] dump가 직접 증거.
- ✅ **Medium**: "Hazard rendering은 분포-head + median operator 위에 capacity를 추가하지 못한다" (4 cells × 동일 ceiling, r > 0.89). caveat: 본 라운드 ablation 한정.
- ✅ **Medium**: "R5B Spherical-SH high-λ가 분포-head expectation에 7-metric 모두 우위 게인" (exp946). caveat: 단일 셀, multi-seed 미검증.
- ⚠️ **Weak — multi-seed 필요**: "분포-head는 confidence-thresholded metric (δ1)에서 scalar에 우위" (round-1/2/3 일관 +0.020, noise floor 3×). peer review challenge 가능.
- ❌ **Claim 불가**: "Hazard renderer가 *near-region에서 systematic underestimate*한다" — [haz] sliced 1-shot val dump에서만, test set 전체 미확정.

**Paper 작성 전 missing evidence**:
1. Multi-seed (≥3) on exp946, exp961, exp958. *없으면* noise floor caveat 필수.
2. Round-0 baseline 재현 (R6 Priority 0). *없으면* "round-0 RMSE 1.10 격차 source 미확인" caveat 필수.
3. Per-pixel 시각화 (10+ scenes × 5 cells). *없으면* "qualitative behavior unverified" — review challenge.
4. Per-bin GT-stratified test metric (현재 train-val [haz] dump만 있음, test pass에 미추가).

---

## 부록 — 라운드 산출 ckpt 목록 (best_score 기준)

```
checkpoints/echorange_soundspaces_BS48_Lr0.0001_AdamW_exp{917,930-934,936-951}_*_bs48_r2|r3/best_score.pth
checkpoints/echodiffusion_soundspaces_BS32_Lr0.0001_AdamW_exp958_R5F_S40_echodiff_bs32_r3_ep40/best_score.pth
checkpoints/echorange_soundspaces_BS32_Lr0.0001_AdamW_exp{959-963}_*_bs32_r3*/best_score.pth
```

전체 70 entries 풀덤프 (4-best ckpt 변형 포함)와 historical baseline 표는 같은 폴더의 `E20260429-exp917-963-IO_and_code_description.md` §6 참조.
