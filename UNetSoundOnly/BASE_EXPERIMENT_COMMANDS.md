# Base 실험 실행 가이드 (train.py)

## 현재 디폴트 설정
Config 파일(`conf/mode/train.yaml`, `conf/dataset/batvisionv2.yaml`)에 이미 설정되어 있음:

- ✅ **Audio Format**: `mel_spectrogram`
- ✅ **Loss Function**: `Combined` (L1 + SI-log)
  - `l1_weight: 0.237`
  - `silog_weight: 0.637`
  - `silog_lambda: 0.869`
- ✅ **Optimizer**: `AdamW`
- ✅ **Learning Rate**: `0.002`
- ✅ **Batch Size**: `256`

---

## 1. 가장 간단한 실행 (디폴트 설정 사용)

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --experiment_name base_default
```

**또는 스크립트로 실행:**
```bash
./run_base_experiment.sh
```

이 명령어는 **자동으로** 다음을 사용합니다:
- Mel-spectrogram (디폴트)
- Combined loss (L1 + SI-log, 디폴트)
- W&B 로깅

---

## 2. 실험 옵션

### 2.1 SI-log Lambda 조정

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --silog_lambda 0.85 \
  --experiment_name base_silog085
```

### 2.2 Loss Weight 조정

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --l1_weight 0.3 \
  --silog_weight 0.7 \
  --experiment_name base_custom_weights
```

### 2.3 Pure SI-log Loss 사용

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --criterion SIlog \
  --experiment_name base_silog_only
```

### 2.4 Pure L1 Loss 사용 (비교용)

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --criterion L1 \
  --experiment_name base_l1_only
```

### 2.5 Spectrogram으로 변경 (Mel 대신)

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --audio_format spectrogram \
  --experiment_name base_spectrogram
```

### 2.6 Batch Size / Learning Rate 조정

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --batch_size 128 \
  --learning_rate 0.001 \
  --experiment_name base_bs128_lr001
```

---

## 3. 고급 실험

### 3.1 최적화된 설정 (권장)

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --experiment_name base_optimized
```

### 3.2 체크포인트에서 재개

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --experiment_name base_default \
  --checkpoints 50
```

### 3.3 Validation 빈도 조정

```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --validation_iter 5 \
  --experiment_name base_val5
```

### 3.4 모든 옵션 활용

```bash
python train.py \
  --dataset batvisionv2 \
  --audio_format mel_spectrogram \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --criterion Combined \
  --l1_weight 0.237 \
  --silog_weight 0.637 \
  --silog_lambda 0.869 \
  --batch_size 256 \
  --learning_rate 0.002 \
  --optimizer AdamW \
  --validation_iter 2 \
  --experiment_name base_full
```

---

## 4. 비교 실험

### 4.1 Loss 함수 비교

```bash
# L1 only
python train.py --dataset batvisionv2 --use_wandb --criterion L1 --experiment_name compare_l1

# SI-log only
python train.py --dataset batvisionv2 --use_wandb --criterion SIlog --experiment_name compare_silog

# Combined (default)
python train.py --dataset batvisionv2 --use_wandb --experiment_name compare_combined
```

### 4.2 Audio Format 비교

```bash
# Mel-spectrogram (default)
python train.py --dataset batvisionv2 --use_wandb --experiment_name compare_mel

# Spectrogram
python train.py --dataset batvisionv2 --use_wandb --audio_format spectrogram --experiment_name compare_spec
```

### 4.3 SI-log Lambda 비교

```bash
# Lambda 0.5
python train.py --dataset batvisionv2 --use_wandb --silog_lambda 0.5 --experiment_name lambda_05

# Lambda 0.85
python train.py --dataset batvisionv2 --use_wandb --silog_lambda 0.85 --experiment_name lambda_085

# Lambda 0.869 (default)
python train.py --dataset batvisionv2 --use_wandb --experiment_name lambda_default
```

---

## 5. 실험 결과 확인

### W&B 대시보드
```
https://wandb.ai/branden/batvision-depth-estimation
```

### 로컬 결과
- **체크포인트**: `./checkpoints/unet_baseline_batvisionv2_BS256_Lr0.002_AdamW_{experiment_name}/`
- **시각화**: `./results/unet_baseline_batvisionv2_BS256_Lr0.002_AdamW_{experiment_name}/`
- **로그**: `./logs/unet_baseline_batvisionv2_BS256_Lr0.002_AdamW_{experiment_name}/`

### 시각화 확인
```bash
# 최신 결과 확인
ls -lht results/unet_baseline_*/epoch_*_prediction.png | head -5

# 특정 epoch 확인
open results/unet_baseline_batvisionv2_BS256_Lr0.002_AdamW_base_default/epoch_0050_prediction.png
```

---

## 6. 현재 Config 설정 확인

### 확인 명령어
```bash
# Train config
cat conf/mode/train.yaml

# Dataset config
cat conf/dataset/batvisionv2.yaml
```

### 현재 디폴트 값
```yaml
# conf/mode/train.yaml
criterion: Combined
l1_weight: 0.237
silog_weight: 0.637
silog_lambda: 0.869
learning_rate: 0.002
optimizer: AdamW
batch_size: 256
epochs: 200

# conf/dataset/batvisionv2.yaml
audio_format: mel_spectrogram
max_depth: 30.0
images_size: 256
```

---

## 7. 권장 실험 시퀀스

### Step 1: 기본 실험 (디폴트 설정)
```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --experiment_name base_step1
```

### Step 2: Best Model 저장 추가
```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --experiment_name base_step2
```

### Step 3: SI-log Lambda 튜닝
```bash
# Lambda 0.85 시도
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --save_best_model \
  --silog_lambda 0.85 \
  --experiment_name base_step3_lambda085
```

### Step 4: Loss Weight 튜닝
```bash
# SI-log 비중 증가
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --save_best_model \
  --l1_weight 0.2 \
  --silog_weight 0.8 \
  --experiment_name base_step4_silog08
```

---

## 8. 문제 해결

### Out of Memory
```bash
# Batch size 줄이기
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --batch_size 128 \
  --experiment_name base_bs128
```

### Loss가 발산할 때
```bash
# Learning rate 줄이기
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --learning_rate 0.001 \
  --experiment_name base_lr001
```

### Validation이 너무 느릴 때
```bash
# Validation 빈도 줄이기
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --validation_iter 5 \
  --experiment_name base_val5
```

---

## 9. Config 파일 직접 수정 (선택사항)

디폴트 설정을 영구적으로 변경하고 싶다면:

```bash
# 1. Train config 수정
vim conf/mode/train.yaml

# 2. Dataset config 수정
vim conf/dataset/batvisionv2.yaml

# 3. 수정 후 실행 (argument 없이 디폴트 사용)
python train.py --dataset batvisionv2 --use_wandb --experiment_name base_modified_config
```

---

## 10. 요약

### 가장 간단한 실행 (디폴트 설정 사용)
```bash
python train.py --dataset batvisionv2 --use_wandb --experiment_name base_default
```

### 가장 추천하는 설정
```bash
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --experiment_name base_recommended
```

### 디버깅용 (빠른 테스트)
```bash
python train.py \
  --dataset batvisionv2 \
  --batch_size 32 \
  --experiment_name base_debug
```

### 스크립트로 실행
```bash
./run_base_experiment.sh
```

---

## 11. 다른 실험 파일과 비교

| 실험 파일 | 특징 |
|----------|------|
| `train.py` | **기본 baseline** (U-Net) |
| `train_base_residual.py` | Base + Residual 구조 |
| `train_binaural_attention.py` | Binaural attention mechanism |
| `train_rgb_depth.py` | RGB 기반 (비교용) |
| `train_adabins_distillation.py` | Distillation |

**Base 실험 = `train.py` 사용!**
