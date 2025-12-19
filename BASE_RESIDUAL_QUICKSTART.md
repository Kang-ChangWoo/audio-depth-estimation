# Base + Residual 빠른 시작 가이드

## 🚀 바로 실행하기

### 1. 기본 학습 (가장 간단)

```bash
cd /root/storage/implementation/shared_audio/Batvision-Dataset/UNetSoundOnly

python train_base_residual.py \
  --dataset batvisionv2 \
  --experiment_name my_first_exp
```

### 2. W&B 로깅 포함 (권장)

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --use_wandb \
  --experiment_name exp1
```

### 3. Adaptive Loss 사용 (커리큘럼 러닝)

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --use_wandb \
  --use_adaptive_loss \
  --warmup_epochs 20 \
  --experiment_name adaptive_exp
```

---

## 📊 결과 확인

### 시각화 파일

학습 중 자동으로 생성됩니다:

```
results/{experiment_name}/epoch_0010_decomposition.png
results/{experiment_name}/epoch_0020_decomposition.png
...
```

각 이미지는 4개 컬럼:
1. **Base Depth** - 방의 구조 (벽, 바닥, 천장)
2. **Residual** - 세부 보정 (가구, 물체)
3. **Final Depth** - Base + Residual
4. **Ground Truth** - 정답

### W&B 대시보드

```
train/loss_base      - 구조 학습 진행도
train/loss_sparse    - Residual 크기
train/loss_recon     - 최종 정확도
val/decomposition    - 시각화
val/rmse             - 성능 메트릭
```

---

## 🎯 핵심 아이디어

### 왜 Base + Residual?

오디오 → Depth는 **under-constrained** 문제:
- 한 번에 전체 depth 학습 → 어려움 ❌
- 단계별 학습 → 쉬움 ✅

1. **Base**: 방의 전체 구조 학습 (쉬운 문제)
2. **Residual**: 세부 디테일 보정 (작은 문제)
3. **Final = Base + Residual**

### Loss Function (3가지 성분)

```python
L_total = λ1 * L_reconstruction    # 최종 결과가 GT와 같아야 함
        + λ2 * L_structural        # Base가 구조를 학습하도록 유도
        + λ3 * L_sparsity          # Residual은 작게 유지
```

**핵심**: Layout G.T. 없이도, Depth의 저주파 성분을 구조로 사용!

---

## ⚙️ 주요 파라미터

### Loss 가중치

```bash
# 기본값 (균형잡힌 설정)
--lambda_recon 1.0    # 재구성 loss
--lambda_base 0.5     # 구조 loss
--lambda_sparse 0.1   # 희소성 penalty
```

**튜닝 가이드:**
- Base가 너무 noisy → `--lambda_base` 증가 (0.8, 1.0)
- Residual이 너무 억제됨 → `--lambda_sparse` 감소 (0.05, 0.01)
- 큰 방 → `--lowpass_kernel` 증가 (12, 16)

### 모델 크기

```bash
--base_channels 64    # 기본 (약 30M params)
--base_channels 32    # 작음 (약 8M params)
--base_channels 128   # 큼 (약 120M params)
```

---

## 📈 예상 결과

### 기존 UNet 대비

✅ **20-30% 빠른 수렴**
- Base가 처음 10-20 epoch에 구조 학습
- Residual이 이후 세부사항 학습

✅ **더 나은 구조**
- 벽면이 더 직선적
- 바닥/천장이 더 평평

✅ **해석 가능성**
- Base depth를 보면 모델이 배운 방 구조 확인 가능
- 에러가 구조 문제인지 디테일 문제인지 구분 가능

---

## 🔍 문제 해결

### Base가 Final과 거의 같음
→ Residual이 너무 억제됨
```bash
--lambda_sparse 0.05  # 감소
```

### Residual이 너무 큼
→ Base가 구조를 제대로 못 배움
```bash
--lambda_base 1.0  # 증가
```

### Base에 날카로운 엣지
→ Low-pass 필터가 약함
```bash
--lowpass_kernel 12  # 증가
```

### 학습이 불안정
→ Adaptive loss 사용
```bash
--use_adaptive_loss --warmup_epochs 20
```

---

## 📁 파일 구조

```
UNetSoundOnly/
├── models/
│   └── base_residual_model.py          # 모델 아키텍처
├── utils_base_residual_loss.py         # Loss 함수들
├── train_base_residual.py              # 학습 스크립트 ⭐
├── BASE_RESIDUAL_GUIDE.md              # 상세 가이드
├── BASE_RESIDUAL_QUICKSTART.md         # 이 파일
└── run_base_residual_examples.sh       # 예제 명령어
```

---

## 💡 실전 팁

### 1. 첫 실험

```bash
# 빠른 테스트
python train_base_residual.py \
  --batch_size 32 \
  --epochs 10 \
  --experiment_name quick_test
```

결과 확인:
- `results/base_residual_*/epoch_0010_decomposition.png`
- Base가 구조를 보여주는지 확인

### 2. 본격 학습

```bash
# 제대로 된 실험
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 256 \
  --learning_rate 0.002 \
  --optimizer AdamW \
  --use_wandb \
  --use_adaptive_loss \
  --experiment_name production_run
```

### 3. 비교 실험

기존 모델과 비교:
```bash
# 기존 UNet
python train.py --dataset batvisionv2 --experiment_name baseline_unet

# Base+Residual
python train_base_residual.py --dataset batvisionv2 --experiment_name base_res
```

W&B에서 두 실험 비교!

---

## 🎓 더 알아보기

### 상세 문서

- **BASE_RESIDUAL_GUIDE.md**: 전체 설명, 수식, 이론
- **run_base_residual_examples.sh**: 모든 예제 명령어

### 이론 배경

Taylor Series 유추:
```
f(x) ≈ f(a) + f'(a)(x-a)
      └─ base  └─ residual
```

Depth도 비슷하게:
```
D_final ≈ D_structure + D_details
          └─ base       └─ residual
```

---

## ✅ 체크리스트

학습 전:
- [ ] 데이터셋 경로 확인
- [ ] GPU 사용 가능 확인
- [ ] W&B 설정 (선택)
- [ ] 실험 이름 정함

학습 중:
- [ ] Loss 값 감소 확인
- [ ] 시각화 주기적으로 확인
- [ ] Base가 구조 학습하는지 확인

학습 후:
- [ ] 최종 decomposition 확인
- [ ] RMSE 기존 모델과 비교
- [ ] Base depth 분석

---

## 🚀 지금 바로 시작!

```bash
cd /root/storage/implementation/shared_audio/Batvision-Dataset/UNetSoundOnly

# 1분 테스트
python train_base_residual.py --batch_size 8 --epochs 2 --experiment_name test

# 실제 학습
python train_base_residual.py --dataset batvisionv2 --use_wandb --experiment_name exp1
```

**성공을 기원합니다! 🎉**

