# round3_frozen — code snapshot driving R4 (hazard rescue) and R5 (range posterior + soft-quantile + SH + cylindrical)

이 디렉토리는 `logs/n9_0427_test/` 안의 28개 실험(exp917, exp930–963)을 학습·평가한 **그 시점의 코드 스냅샷**입니다. 라운드 3에서 라운드 5까지의 누적 결과입니다.

## 파일 매핑 (절대 경로 → 스냅샷 파일)

| 스냅샷 파일 | 원본 | 비고 |
|---|---|---|
| `range_head.py` | `models/bin_based/range_head.py` | RangeDepthHead, HazardRangeDepthHead, soft_range_nll_loss, R4 hazard losses(rendered_event_nll/survival_loss/soft_hit_bce_loss/hazard_free_loss), `range_point_estimate`, `soft_quantile_depth`. |
| `echorange.py` | `models/bin_based/echorange.py` | depth_head_type ∈ {scalar, range, hazard}; range_output_quantile/temperature/bin_axis wiring. |
| `spherical_loss.py` | `models/bin_based/spherical_loss.py` | ERP cos(lat) area weighting + low-order SH coeff matching (L≤4). |
| `__init__.py` | `models/bin_based/__init__.py` | re-export. |
| `echorange.yaml` | `config/echorange.yaml` | echorange family default cfg. |
| `echodiffusion.yaml` | `config/echodiffusion.yaml` | echodiffusion baseline default cfg. |
| `_train_step_echorange.py` | `train.py:141..516` (verbatim 발췌) | scalar/range/hazard 3 모드, soft-quantile + SH + cylindrical. |
| `_train_loop_4best.py` | `train.py:1270..1488` (excerpt) | per-metric 4-best 체크포인트. |
| `_test_utils_evaluate_echorange_branch.py` | `utils/test_utils.py:1..101 + 268..292` | range_eval_mode 후처리 + cyl→radial 투영. |
| `n2_bulk_0429_r3.sh` | `scripts/n2_bulk_0429_r3.sh` | n2 서버용 R5 main bulk launcher (exp936–957). |
| `n9_bulk_0429_r3.sh` | `scripts/n9_bulk_0429_r3.sh` | n9 서버용 R5 main bulk launcher (exp958–967). |
| `n2_bulk_0428_r2.sh` | `scripts/n2_bulk_0428_r2.sh` | R4 hazard rescue launcher (exp917, exp930–934). |

## 라운드 정의

- **Round 0** — 외부 echo diffusion 본가 학습 (exp11 best: ABS_REL 0.4300 / RMSE 1.1060 / δ1 0.4876).
- **Round 1** (`logs/round1_bin-based/round1/`) — bin-based softmax head 도입 (exp900–906). Best exp906: 0.4814 / 1.2532 / 0.5079.
- **Round 2** (`logs/round1_bin-based/round2/`) — bs=32 ERP-ablation, hazard 첫 시도 (exp907–920). Best exp907 expectation: 0.4705 / 1.2269 / 0.5029. exp907_TESTmedian: 0.4202 / 1.2765 / 0.5129. Hazard(exp913) 폭발.
- **Round 3** — R3 Round 4 hazard rescue (exp917, 930–934). soft_hit / event_nll / survival 3 종 aux + smooth ramp + far-thresh fix. soft_hit이 분포 head 수준만 회복, hazard 자체 게인 없음 결론.
- **Round 5** (현재 코드) — RangeDepthHead 메인화 + posterior 9가지 representative + soft-quantile train loss + SH aux + cylindrical bin-axis + 4-best ckpt. exp936–963 셀.

## 28개 실험 패밀리 매핑

| 셀 ID | 패밀리 | 메인 변수 |
|---|---|---|
| exp917 | R3 echo-range hazard 잔해 (anchor) | hazard log-delta strong |
| exp930 | R4 hazard rescue | aux=raw_hit (재현) |
| exp931 | R4 hazard rescue | aux=event_nll free=0 |
| exp932 | R4 hazard rescue | aux=event_nll only |
| exp933 | R4 hazard rescue | aux=survival |
| exp934 | R4 hazard rescue | aux=soft_hit, target=0.75 |
| exp936–943 | R5A Soft-Quantile sweep | (q ∈ {0.45, 0.50, 0.55}) × (τ ∈ {0.03, 0.05}) × (λ_sq ∈ {0.25, 0.50}) |
| exp944–949 | R5B Spherical-SH sweep | L ∈ {2, 3} × λ_SH ∈ {0.02, 0.05, 0.10} × log_depth ∈ {true, false} |
| exp950–951 | R5C Combo (Soft-Q + SH) | (q=0.5/τ=0.05/λ=0.25) + (L=2/λ_SH ∈ {0.02, 0.05}) |
| exp958 | R5F 40-epoch echo diffusion | scalar baseline ep40 (재현) |
| exp959 | R5F 40-epoch echorange-scalar | scalar (echorange 본가) ep40 |
| exp960 | R5F 40-epoch range expectation | anchor ep40 |
| exp961 | R5F 40-epoch range median | anchor ep40 (round-2 exp907 median 재현) |
| exp962 | R5G R20 soft-quantile (n9, low-bin) | anchor (q=0.5/τ=0.05/λ=0.25), bins=20 |
| exp963 | R5H R20 SH (n9, low-bin) | L=2/λ=0.02, bins=20 |

> n9 환경은 단일 GPU bs=32, n2 환경은 DataParallel 2-GPU bs=48. 모든 ckpt는 best_score(0.7·RMSE + 0.3·ABS_REL)에서 추출.

## 주의

- Round 5의 soft-quantile / SH / cylindrical 플래그는 **모두 default off**. 기존 R2 셀(exp907 등)을 재실행하면 동일 결과.
- 4-best 체크포인트는 Round 5에서 도입 — Round 4 셀 평가 시점에는 best_model.pth(=best_score) 단일 파일.
- Hazard 셀(exp913–934)은 Round 5 코드의 smooth ramp(`progress = min(1, epoch / ramp_epochs)`) 적용 — Round 3의 (epoch≤ramp_epoch) → (epoch>ramp_epoch) 단계 점프는 제거됨.
