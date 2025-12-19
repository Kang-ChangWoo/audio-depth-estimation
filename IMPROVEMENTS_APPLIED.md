# Base + Residual Model - Performance Improvements Applied

## 🎯 적용된 수정사항 요약

성능 향상을 위해 다음 5가지 크리티컬한 문제를 수정했습니다:

---

## ✅ 1. Activation Functions 추가 (가장 중요!)

### 문제
- 출력에 activation이 없어서 depth가 음수가 될 수 있었음
- Base와 Residual이 제약 없이 학습

### 해결
```python
# Base depth: 항상 양수 (구조는 양수여야 함)
base_depth = torch.sigmoid(base_depth_raw) * max_depth

# Residual: +/- 허용하되 제한 (보정값, max_depth의 ±20%)
residual = torch.tanh(residual_raw) * (max_depth * 0.2)

# Final: 유효 범위로 clamp
final_depth = torch.clamp(base_depth + residual, 0, max_depth)
```

**효과**: Base가 실제로 양수 구조를 학습, Residual이 작은 보정만 담당

---

## ✅ 2. Gradient Detachment (Curriculum Learning)

### 문제
- Base와 Residual이 독립적으로 학습되지 않음
- 두 디코더가 같은 gradient를 받음

### 해결
```python
# Early epochs (1-20): 함께 학습
if epoch <= warmup_epochs:
    final_depth = base_depth + residual

# Later epochs (21+): Base 고정, Residual만 학습
else:
    final_depth = base_depth.detach() + residual
```

**효과**: 
- Phase 1: Base가 구조 학습
- Phase 2: Residual이 디테일 refine

---

## ✅ 3. Loss 가중치 개선

### Before
```python
lambda_recon = 1.0
lambda_base = 0.5    # 너무 약함
lambda_sparse = 0.1  # 너무 약함
lowpass_kernel = 8   # 너무 작음
```

### After
```python
lambda_recon = 1.0
lambda_base = 0.8    # 60% 증가
lambda_sparse = 0.2  # 100% 증가
lowpass_kernel = 16  # 100% 증가
```

**Adaptive Loss (warmup)**:
```python
# Epoch 1-20
lambda_base_init = 1.5   # Base 학습 강화
lambda_recon_init = 0.5  # Recon 약화
lambda_sparse = 0.3      # Sparsity 강화

# Epoch 21+
lambda_base_final = 0.5
lambda_recon_final = 1.0
lambda_sparse = 0.3
```

**효과**: Base가 제대로 구조를 학습하고, Residual이 과도하게 커지지 않음

---

## ✅ 4. Low-pass Kernel 크기 증가

### 문제
- 8x8 kernel은 256x256 이미지에 너무 작음 (3%)
- 구조 추출이 불충분

### 해결
- 8 → 16 (두 배 증가)
- 256x256 이미지의 6.25% 영역

**효과**: Base가 더 부드러운 구조를 학습

---

## ✅ 5. Max Depth 제약 추가

### 문제
- 모델이 depth 범위를 몰랐음

### 해결
```python
# 모델 생성 시 max_depth 전달
model = create_base_residual_model(
    ...
    max_depth=cfg.dataset.max_depth,  # 30.0
    ...
)
```

**효과**: Sigmoid/Tanh가 올바른 범위로 스케일링

---

## 📊 예상 효과

### 시각화에서 확인할 것

#### Before (문제):
- Base ≈ Final (Residual이 거의 0)
- 또는 Residual ≈ Final (Base가 무의미)
- Base에 high-frequency 노이즈

#### After (개선):
- ✅ Base = smooth structure (벽, 바닥, 천장)
- ✅ Residual = small corrections (물체 경계, ±0.2 * max_depth)
- ✅ Final = Base + Residual (명확한 분리)

### 성능 지표

- **수렴 속도**: 20-30% 빨라짐
- **RMSE**: 5-10% 개선 예상
- **구조 정확도**: 벽면이 더 직선적, 바닥이 더 평평
- **해석성**: Base depth만 봐도 방 구조 파악 가능

---

## 🚀 새로운 실행 명령어

### 기본 설정 (개선된 기본값)

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --experiment_name improved_v1
```

### Adaptive Loss 포함 (강력 추천!)

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --use_wandb \
  --use_adaptive_loss \
  --warmup_epochs 20 \
  --experiment_name improved_adaptive
```

### W&B로 기존 모델과 비교

```bash
# 새 실험 (개선된 버전)
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --use_wandb \
  --use_adaptive_loss \
  --experiment_name v2_improved

# W&B에서 bs64_adaptive vs v2_improved 비교
```

---

## 🔍 결과 확인 방법

### 1. 시각화 확인

```bash
# 새 결과 확인
ls -lh results/base_residual_*_improved*/epoch_*_decomposition.png

# 이미지에서 확인:
# - Base가 부드러운가?
# - Residual이 작은가? (blue/red가 약한가?)
# - Final이 정확한가?
```

### 2. Loss 모니터링

```python
# 학습 중 출력:
Epoch 1: Loss=X.XX (recon=X.XX, base=X.XX, sparse=X.XX)
```

**기대값**:
- `sparse` loss가 작아야 함 (< 0.5)
- `base` loss가 초기에 빠르게 감소
- `recon` loss가 안정적으로 감소

### 3. W&B 대시보드

Plot 생성:
1. **Loss Components**: base vs sparse vs recon over time
2. **Residual Magnitude**: `train/loss_sparse` (작아져야 함)
3. **Performance**: `val/rmse` (기존 모델과 비교)

---

## 📝 변경된 파일들

1. ✅ `models/base_residual_model.py`
   - `__init__`: max_depth 파라미터 추가
   - `forward`: Sigmoid/Tanh activation 추가
   - `create_base_residual_model`: max_depth 전달

2. ✅ `utils_base_residual_loss.py`
   - `BaseResidualLoss`: 기본 가중치 변경 (0.8, 0.2, 16)
   - `AdaptiveBaseResidualLoss`: 초기값 강화 (1.5, 0.3)

3. ✅ `train_base_residual.py`
   - Argument defaults: 개선된 가중치
   - Model creation: max_depth 전달
   - Training loop: Gradient detachment 추가
   - Validation loop: 동일한 detachment 적용

---

## 💡 추가 실험 아이디어

### 더 공격적인 설정 (큰 방, 복잡한 구조)

```bash
python train_base_residual.py \
  --lambda_base 1.0 \
  --lambda_sparse 0.3 \
  --lowpass_kernel 24 \
  --experiment_name aggressive
```

### 보수적인 설정 (작은 방, 간단한 구조)

```bash
python train_base_residual.py \
  --lambda_base 0.5 \
  --lambda_sparse 0.1 \
  --lowpass_kernel 12 \
  --experiment_name conservative
```

---

## 🎓 이론적 배경

### Taylor Series Analogy

```
f(x) ≈ f(a) + f'(a)(x-a) + ...
      └─ base  └─ residual

Depth(x,y) ≈ Structure(x,y) + Details(x,y)
             └─ sigmoid(base) └─ tanh(res)
```

### Activation 선택 이유

- **Sigmoid for Base**: 
  - 출력: [0, max_depth]
  - 구조는 항상 양수
  - 부드러운 gradient

- **Tanh for Residual**:
  - 출력: [-0.2*max_depth, +0.2*max_depth]
  - 보정은 +/- 허용
  - 중심이 0 (기본적으로 보정 안 함)

---

## ✅ 성공 지표

다음을 달성하면 성공:

1. ✅ Base depth가 부드럽고 양수
2. ✅ Residual이 작음 (평균 절댓값 < 1.0)
3. ✅ Final RMSE가 기존 대비 5% 이상 개선
4. ✅ 수렴이 20 epoch 이내에 안정화
5. ✅ Visualization에서 명확한 분리

---

**모든 수정사항이 적용되었습니다! 🎉**

지금 바로 학습을 시작하세요:

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --use_wandb \
  --use_adaptive_loss \
  --experiment_name improved_final
```

