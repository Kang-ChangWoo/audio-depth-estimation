# Base Generalization Update - Critical Fix

## 🎯 핵심 문제 발견

> **"Base가 전체 데이터셋을 일반화할 수 있어야 효과가 있다"**

이전 구현에서 Base와 Residual decoder가 **동일한 용량**(6.8M params)을 가져서:
- Base가 각 샘플에 overfitting (일반화 실패)
- Residual이 할 일이 없어짐 (과억제)
- 성능 향상 없음 (RMSE 정체)

---

## ✅ 적용된 수정사항

### 1. Base Decoder 용량 대폭 축소 ⭐⭐⭐

**Before**:
```python
base_channels = 64
Base Decoder: 64 → 128 → 256 → 512 channels
Parameters: ~6.8M (Residual과 동일)
```

**After**:
```python
base_ch = base_channels // 4  # 16 channels
Base Decoder: 16 → 32 → 64 → 128 channels
Parameters: ~0.4M (Residual의 1/17)
```

**효과**: 
- ✅ Base가 세부사항을 memorize 못함 → **일반화 강제**
- ✅ Base는 대략적인 구조만 학습
- ✅ Residual이 세부사항 담당

---

### 2. Loss 가중치 재조정

#### BaseResidualLoss (기본):
```python
# Before
lambda_base = 0.8
lambda_sparse = 0.2

# After
lambda_base = 1.2    # 50% 증가 - Base 학습 강화
lambda_sparse = 0.05  # 75% 감소 - Residual 자유롭게
```

#### AdaptiveBaseResidualLoss (Curriculum):
```python
# Before
lambda_recon_init = 0.5
lambda_base_init = 1.5
lambda_sparse = 0.3
warmup_epochs = 20

# After
lambda_recon_init = 0.3   # Base 중심 학습
lambda_base_init = 2.0    # Base 강화
lambda_sparse = 0.05      # Residual 억제 완화
warmup_epochs = 50        # Base가 충분히 일반화
```

**효과**:
- ✅ Base가 먼저 일반화된 구조 학습
- ✅ Residual이 적절히 활성화 (sparse_loss 0.2~0.4 예상)
- ✅ Epoch 50까지 Base 중심, 이후 Residual refinement

---

### 3. Residual 범위 확대

```python
# Before
residual = tanh(...) * (max_depth * 0.2)  # ±20%

# After
residual = tanh(...) * (max_depth * 0.3)  # ±30%
```

**이유**: Base 용량이 줄었으니 Residual이 더 많은 역할

---

## 📊 예상 효과

### Architecture 비교

| Component | Before | After | 비율 |
|-----------|--------|-------|------|
| Base Decoder | 6.8M params | 0.4M params | 1/17 |
| Residual Decoder | 6.8M params | 6.8M params | 1x |
| **Total** | 25.9M | 19.5M | **-25%** |

### 학습 패턴 변화

#### Before (문제):
```
Epoch 10: sparse=0.04  → Residual 거의 0
          Base ≈ Final → Base가 모든 것 학습 시도
          Val RMSE: 2.6 (정체)
```

#### After (예상):
```
Epoch 10: sparse=0.3   → Residual 활발
          Base ≠ Final → 명확한 역할 분담
          Val RMSE: 2.2 (개선!)
          
Epoch 50: Base 고정   → 일반화된 구조 완성
          Residual만 학습 → 세부사항 refine
          Val RMSE: 1.8 (목표!)
```

---

## 🎓 이론적 배경

### PCA/SVD Decomposition

```
Depth = Base_generalized + Residual_specific

Base (Low-rank):
- 전체 데이터셋의 주요 성분
- 모든 샘플에 공통
- 적은 파라미터로 표현 가능
- "평균적인 방의 구조"

Residual (High-rank):
- 각 샘플의 고유 특성
- 샘플별로 다름
- 많은 파라미터 필요
- "이 방만의 특수한 배치"
```

### Capacity vs Generalization

```
High Capacity Base (Before):
└─ Memorization → 각 샘플별로 다르게 학습
└─ Poor Generalization → 새 샘플에 적용 안됨

Low Capacity Base (After):
└─ Forced Generalization → 공통 패턴만 학습
└─ Better Transfer → 새 샘플에도 잘 적용
```

---

## 🚀 실행 명령어

### 기본 설정 (개선된 기본값)

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --experiment_name generalized_v1
```

### Adaptive Loss 포함 (강력 추천!)

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --use_wandb \
  --use_adaptive_loss \
  --warmup_epochs 50 \
  --experiment_name generalized_adaptive
```

### 커스텀 설정

```bash
python train_base_residual.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --use_wandb \
  --use_adaptive_loss \
  --lambda_base 2.0 \
  --lambda_sparse 0.05 \
  --warmup_epochs 50 \
  --experiment_name generalized_custom
```

---

## 📈 모니터링 포인트

### 1. Sparse Loss 확인

```
Epoch 1-10:  sparse ≈ 0.3~0.4  (Residual 활발) ✅
Epoch 50+:   sparse ≈ 0.2~0.3  (안정화) ✅

만약 sparse < 0.1 이면:
→ lambda_sparse를 더 낮춰야 함 (0.05 → 0.02)
```

### 2. Base vs Final 비교

시각화에서 확인:
- **Base**: 매우 부드러움, 대략적인 구조만
- **Residual**: 물체 경계에서 ±값, 평균 0.3~0.5 정도
- **Final**: Base + 명확한 디테일

### 3. 일반화 확인

```python
# 학습 중 로그:
Epoch 10: base_loss 감소 (1.5 → 1.0)  # 구조 학습
Epoch 50: base_loss 안정 (0.8 수준)   # 일반화 완료
Epoch 100: sparse_loss 적절 (0.2~0.3) # Residual 활발
```

---

## 🔍 Validation 체크리스트

### Phase 1 (Epoch 1-20): Base 학습

- [ ] Base loss 빠르게 감소
- [ ] Sparse loss 0.3~0.4 유지
- [ ] Base depth가 부드러움
- [ ] Val RMSE 서서히 개선

### Phase 2 (Epoch 20-50): 통합 학습

- [ ] Base loss 안정화
- [ ] Sparse loss 서서히 감소 (0.3 → 0.2)
- [ ] Residual이 디테일 학습 시작
- [ ] Val RMSE 지속 개선

### Phase 3 (Epoch 50+): Residual Refinement

- [ ] Base 고정 (detached)
- [ ] Residual만 학습
- [ ] Sparse loss 0.2~0.3 유지
- [ ] Val RMSE 최종 수렴

---

## 💡 성공 지표

### Minimum Requirements

- ✅ Sparse loss > 0.15 (Residual이 활발)
- ✅ Base depth가 부드럽고 일반화됨
- ✅ Val RMSE < 2.2 (기존 대비 15% 개선)
- ✅ Delta1 > 0.45 (정확도 향상)

### Ideal Results

- 🎯 Sparse loss ≈ 0.2~0.3 (균형)
- 🎯 Base depth가 모든 샘플에 비슷한 구조
- 🎯 Val RMSE < 1.8 (기존 대비 30% 개선)
- 🎯 Delta1 > 0.55 (높은 정확도)

---

## 🐛 문제 해결

### 문제 1: Sparse가 여전히 너무 작음 (< 0.1)

**해결**:
```bash
--lambda_sparse 0.02  # 더 낮춤
--lambda_base 1.5     # Base를 약간 약화
```

### 문제 2: Val RMSE가 초기에 높음

**정상**: Base가 충분히 일반화되기 전까지는 높을 수 있음
- Epoch 20까지 기다리기
- 그 후에도 2.8 이상이면 문제

### 문제 3: Base와 Final이 너무 다름

**해결**:
```bash
--lambda_base 2.5  # Base loss 더 강화
--warmup_epochs 60  # 더 긴 warmup
```

---

## 📝 변경된 파일

1. ✅ `models/base_residual_model.py`
   - Base decoder: 64 → 16 channels (1/4 용량)
   - Residual 범위: 20% → 30%

2. ✅ `utils_base_residual_loss.py`
   - BaseResidualLoss: lambda_base=1.2, lambda_sparse=0.05
   - AdaptiveLoss: init_base=2.0, sparse=0.05, warmup=50

3. ✅ `train_base_residual.py`
   - 기본값: lambda_base=1.2, lambda_sparse=0.05, warmup=50

---

## 🎯 핵심 인사이트

**문제**: Base와 Residual이 같은 용량 → Base가 일반화 실패

**해결**: Base 용량 1/17로 축소 → **강제 일반화**

**결과**: 
- Base = 전체 데이터셋의 "평균 구조"
- Residual = 각 샘플의 "고유 특성"
- 명확한 역할 분담 → 성능 향상!

---

**이제 진짜 작동할 것입니다!** 🚀

모니터링하면서 sparse loss가 0.2~0.3 수준을 유지하는지 확인하세요!

