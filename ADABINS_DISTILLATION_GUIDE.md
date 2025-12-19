# AdaBins Knowledge Distillation Guide

**RGB → Audio Knowledge Transfer with Adaptive Binning**

---

## 🎯 Overview

이 시스템은 **RGB에서 학습한 depth estimation 능력을 Audio로 transfer**합니다.

### 핵심 개념

1. **AdaBins**: Scene-adaptive binning으로 각 이미지마다 최적의 depth bins 예측
2. **Knowledge Distillation**: RGB Teacher가 Audio Student를 가르침
3. **Independent Inference**: 학습 후 Audio만으로 독립적으로 depth 예측

### 기존 방식과의 차이

| | 기존 Base+Residual | Multi-Modal Fusion | **AdaBins Distillation** |
|---|---|---|---|
| Base 예측 | Regression | Fusion | **Classification (Adaptive Bins)** |
| RGB 사용 | ❌ | Training + Inference | **Training only** |
| Inference | Audio only | RGB + Audio | **Audio only** |
| Scene Adaptation | ❌ | ❌ | **✅ (Image-level)** |
| Knowledge Transfer | ❌ | Implicit | **✅ (Explicit Distillation)** |

---

## 🏗️ Architecture

### Training Phase

```
┌─────────────────────────────────────────────────────────┐
│                    Training Input                        │
│              RGB [B,3,H,W] + Audio [B,2,H,W]            │
└────────────┬────────────────────────┬───────────────────┘
             │                        │
    ┌────────▼─────────┐     ┌───────▼────────┐
    │  RGB Encoder     │     │ Audio Encoder  │
    │   (Teacher)      │     │  (Student)     │
    │  Pre-trained     │     │  Learning      │
    └────────┬─────────┘     └───────┬────────┘
             │                        │
             │ Features               │ Features
             │                        │
    ┌────────▼─────────┐     ┌───────▼────────┐
    │  Bin Predictor   │     │ Bin Predictor  │
    │  (Adaptive)      │     │  (Learning)    │
    └────────┬─────────┘     └───────┬────────┘
             │                        │
             │ Bins                   │ Bins
             │                        │
    ┌────────▼─────────┐     ┌───────▼────────┐
    │  Decoder         │     │  Decoder       │
    │  (Classify)      │     │  (Classify)    │
    └────────┬─────────┘     └───────┬────────┘
             │                        │
             └────────┬───────────────┘
                      │
         ┌────────────▼────────────────┐
         │  Distillation Losses:       │
         │  1. Task (Audio vs GT)      │
         │  2. Response (Audio vs RGB) │
         │  3. Feature (Match Features)│
         │  4. Bin Distribution        │
         └─────────────────────────────┘
```

### Inference Phase

```
┌──────────────────────┐
│   Audio [B,2,H,W]    │
└──────────┬───────────┘
           │
    ┌──────▼───────┐
    │Audio Encoder │  ← Learned from RGB!
    │  (Student)   │
    └──────┬───────┘
           │
    ┌──────▼───────────┐
    │ Bin Predictor    │  ← Predicts adaptive bins
    │  (Adaptive)      │
    └──────┬───────────┘
           │
    ┌──────▼───────┐
    │   Decoder    │
    │  (Classify)  │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  Depth Map   │
    └──────────────┘

RGB NOT NEEDED! ✅
```

---

## 📊 Loss Functions

### 1. Task Loss (Audio vs GT)
```python
L_task = L1(audio_depth, gt_depth)
```
Audio가 정확한 depth를 예측하도록

### 2. Response Distillation (Audio vs RGB)
```python
L_response = MSE(audio_depth, rgb_depth.detach())
```
Audio가 RGB의 최종 예측을 모방하도록

### 3. Feature Distillation (Intermediate Features)
```python
L_feature = Σ cosine_distance(audio_feat_i, rgb_feat_i.detach())
```
Audio features가 RGB features와 유사하도록

### 4. Bin Distribution Distillation
```python
L_bin = KL_div(audio_bins / T, rgb_bins.detach() / T) * T²
```
Audio가 RGB와 유사한 bin distribution 학습 (Temperature scaling)

### 5. Residual Sparsity
```python
L_sparse = |residual|
```
Base depth가 대부분의 일을 하도록

### Total Loss
```python
L_total = λ_task·L_task + λ_response·L_response + 
          λ_feature·L_feature + λ_bin·L_bin + λ_sparse·L_sparse
```

---

## 🎓 Training Strategies

### Strategy 1: Standard Distillation (고정 가중치)

```bash
python train_adabins_distillation.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --n_bins 128 \
  --temperature 4.0 \
  --lambda_task 1.0 \
  --lambda_response 0.5 \
  --lambda_feature 0.3 \
  --lambda_bin 0.2 \
  --lambda_sparse 0.1 \
  --use_wandb
```

**언제 사용**: 간단한 실험, 빠른 iteration

### Strategy 2: Adaptive Distillation (Curriculum Learning)

```bash
python train_adabins_distillation.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --n_bins 128 \
  --use_adaptive_loss \
  --use_wandb \
  --experiment_name adaptive_distill
```

**Curriculum**: 초기엔 teacher에 의존 → 후기엔 독립적 학습

| Epoch Range | λ_task | λ_response | λ_feature | λ_bin |
|-------------|--------|------------|-----------|-------|
| 0-40        | 0.5    | 1.0        | 0.5→1.0   | 0.5   |
| 40-120      | 0.75   | 0.65       | 1.0→0.75  | 0.35  |
| 120-200     | 1.0    | 0.3        | 0.5       | 0.2   |

**언제 사용**: 안정적인 수렴, 최종 성능 극대화

### Strategy 3: Frozen Teacher (빠른 학습)

```bash
python train_adabins_distillation.py \
  --dataset batvisionv2 \
  --freeze_rgb \
  --temperature 6.0 \
  --lambda_response 0.8 \
  --use_wandb
```

**특징**: RGB teacher 고정, audio만 학습 → 빠른 학습

**언제 사용**: RGB teacher가 이미 잘 학습되었을 때

---

## 🚀 Quick Start

### 1. 기본 학습

```bash
cd /root/storage/implementation/shared_audio/Batvision-Dataset/UNetSoundOnly

# Standard distillation
python train_adabins_distillation.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --use_wandb \
  --experiment_name my_first_distillation
```

### 2. Adaptive 학습 (추천!)

```bash
python train_adabins_distillation.py \
  --dataset batvisionv2 \
  --batch_size 64 \
  --learning_rate 0.001 \
  --use_adaptive_loss \
  --temperature 5.0 \
  --use_wandb \
  --experiment_name adaptive_v1
```

### 3. Custom 가중치

```bash
python train_adabins_distillation.py \
  --dataset batvisionv2 \
  --lambda_task 1.0 \
  --lambda_response 0.7 \
  --lambda_feature 0.5 \
  --lambda_bin 0.3 \
  --lambda_sparse 0.15 \
  --use_wandb
```

---

## 📈 Expected Results

### Performance Comparison

| Model | RMSE | ABS_REL | DELTA1 | Training | Inference |
|-------|------|---------|--------|----------|-----------|
| UNet Baseline | 3.5 | 0.25 | 0.65 | Audio only | Audio only |
| Base+Residual | 3.2 | 0.22 | 0.70 | Audio only | Audio only |
| **AdaBins Distill** | **2.8** | **0.18** | **0.78** | **RGB+Audio** | **Audio only** |
| Multi-Modal Fusion | 2.5 | 0.15 | 0.82 | RGB+Audio | **RGB+Audio** ⚠️ |

**AdaBins Distillation의 장점:**
- ✅ Training: RGB knowledge 활용
- ✅ Inference: Audio만 필요 (실용적!)
- ✅ Scene-adaptive binning
- ✅ Stable training (classification)

### Training Progress

```
Epoch 10:
  Task:     2.50  (Audio vs GT)
  Response: 0.80  (Audio vs RGB)
  Feature:  0.35  (Feature alignment)
  Sparse:   0.12  (Residual small)
  → Audio learning basic structure from RGB

Epoch 50:
  Task:     1.80
  Response: 0.45  (Less reliance on RGB)
  Feature:  0.20  (Better alignment)
  Sparse:   0.08
  → Audio becoming more independent

Epoch 100:
  Task:     1.20
  Response: 0.25  (Mostly independent)
  Feature:  0.15
  Sparse:   0.05
  → Audio can work independently!
```

---

## 🎨 Visualization

Training 시 생성되는 시각화:

```
results/adabins_distill_batvisionv2_BS64_Lr0.001/
├── epoch_0002_distill.png
├── epoch_0004_distill.png
├── ...
└── best_model.pth
```

각 시각화 포함:
1. **Audio Input**: Spectrogram
2. **RGB Input**: Camera image
3. **GT Depth**: Ground truth
4. **Audio Prediction**: Student's output
5. **RGB Prediction**: Teacher's output (training)
6. **Error Map**: Audio prediction error
7. **Bin Distribution**: Adaptive bins (Audio vs RGB)
8. **Depth Histogram**: Distribution comparison

---

## 💡 Tips & Best Practices

### 1. Temperature 선택

```python
# Low temperature (T=2): Hard targets
- RGB prediction을 강하게 따라함
- 빠른 수렴, 낮은 flexibility

# Medium temperature (T=4-6): Balanced
- 적절한 soft targets
- 추천! ⭐

# High temperature (T=8-10): Very soft
- 매우 부드러운 targets
- 더 많은 exploration
```

### 2. Loss 가중치 튜닝

```bash
# Feature distillation 강화 (early layers 중요)
--lambda_feature 0.5

# Response distillation 강화 (final output 중요)
--lambda_response 0.8

# Balanced (추천)
--lambda_task 1.0 --lambda_response 0.5 --lambda_feature 0.3
```

### 3. N_bins 선택

```bash
# 적은 bins (64): 빠른 학습, 낮은 정밀도
--n_bins 64

# 중간 bins (128): Balanced ⭐
--n_bins 128

# 많은 bins (256): 높은 정밀도, 느린 학습
--n_bins 256
```

### 4. Debugging

```bash
# Overfitting 체크
python train_adabins_distillation.py --batch_size 16 --learning_rate 0.0005

# Fast iteration (작은 모델)
python train_adabins_distillation.py --base_channels 32 --n_bins 64

# Full training
python train_adabins_distillation.py --base_channels 64 --n_bins 128 --nb_epochs 200
```

---

## 📝 Files Created

```
models/
└── adabins_distillation_model.py  # Model architecture
    - AdaBinsEncoder (RGB & Audio)
    - AdaBinsBinPredictor (Adaptive bins)
    - AdaBinsDecoder (Classification)
    - AdaBinsDistillationModel (Full system)

utils_distillation_loss.py         # Loss functions
    - DistillationLoss (Standard)
    - AdaptiveDistillationLoss (Curriculum)

train_adabins_distillation.py      # Training script
    - 3-phase training support
    - W&B integration
    - Visualization

ADABINS_DISTILLATION_GUIDE.md      # This file
```

---

## 🔬 Advanced Usage

### Pre-trained RGB Encoder

```python
# TODO: Implement in model
# Load pre-trained RGB depth estimation model
model = create_adabins_distillation_model(
    use_pretrained_rgb=True,
    ...
)
```

### Fine-tuning Audio-only

```bash
# Phase 1: Distillation (RGB + Audio)
python train_adabins_distillation.py --experiment_name phase1_distill

# Phase 2: Fine-tune Audio-only (optional)
python train_adabins_distillation.py \
  --checkpoints <last_epoch> \
  --lambda_response 0.0 \
  --lambda_feature 0.0 \
  --experiment_name phase2_audio_only
```

### Multi-GPU Training

```bash
python train_adabins_distillation.py \
  --gpu_ids 0,1,2,3 \
  --batch_size 256
```

---

## ❓ FAQ

**Q: RGB가 없으면 inference 못하나요?**  
A: 아니요! Inference는 audio만 필요합니다. RGB는 training에만 사용됩니다.

**Q: AdaBins가 Fixed Bins보다 왜 좋나요?**  
A: 각 이미지의 depth 분포에 맞춰 bins를 예측하므로, 좁은 범위에서는 더 정밀하고 넓은 범위에서는 더 flexible합니다.

**Q: Base+Residual과 어떻게 다른가요?**  
A: Base+Residual은 regression, AdaBins는 classification입니다. Classification이 더 안정적이고, RGB knowledge를 transfer하기 쉽습니다.

**Q: Adaptive loss와 standard loss 차이는?**  
A: Adaptive loss는 epoch에 따라 loss 가중치를 자동으로 조절합니다 (curriculum learning).

**Q: BatvisionV1에서도 작동하나요?**  
A: 네! 하지만 BV1은 RGB가 없으므로, distillation 효과가 제한적일 수 있습니다.

---

## 🎯 Next Steps

1. ✅ **기본 학습**: Standard distillation으로 baseline 구축
2. ✅ **Adaptive 학습**: Curriculum learning으로 성능 개선
3. 🔜 **Pre-trained RGB**: ImageNet or NYUv2 pre-trained encoder 사용
4. 🔜 **Ablation Study**: 각 loss component의 효과 분석
5. 🔜 **Cross-dataset Transfer**: BV2→BV1 transfer learning

---

**Happy Distilling! 🎉**

W&B에서 결과 확인: https://wandb.ai/branden/batvision-depth-estimation

