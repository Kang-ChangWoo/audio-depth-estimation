# Base 실험 실행 가이드 - Batvision V1

## ⚠️ Batvision V1 vs V2 차이점

| 항목 | V1 | V2 |
|------|----|----|
| **Audio Format** | `spectrogram` | `mel_spectrogram` |
| **Mel 지원** | ❌ 미지원 | ✅ 지원 |
| **Max Depth** | 12.0m | 30.0m |
| **Learning Rate** | 0.001 | 0.002 |
| **Batch Size** | 128 | 256 |
| **Depth Norm** | True | False |

---

## 현재 디폴트 설정 (V1)
Config 파일(`conf/dataset/batvisionv1.yaml`)에 설정되어 있음:

- ✅ **Audio Format**: `spectrogram` (mel_spectrogram은 V1에서 미지원)
- ✅ **Loss Function**: `Combined` (L1 + SI-log)
  - `l1_weight: 0.237`
  - `silog_weight: 0.637`
  - `silog_lambda: 0.869`
- ✅ **Max Depth**: `12.0m`
- ✅ **Depth Norm**: `True`

Paper 권장 설정:
- Learning Rate: `0.001`
- Batch Size: `128`

---

## 1. 가장 간단한 실행 (디폴트 설정 사용)

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name base_v1_default
```

**또는 스크립트로 실행:**
```bash
./run_base_v1_experiment.sh
```

**특징:**
- Spectrogram 사용 (V1 디폴트)
- Combined loss (L1 + SI-log)
- Paper 권장 hyperparameters
- W&B 로깅

---

## 2. 실험 옵션

### 2.1 SI-log Lambda 조정

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --silog_lambda 0.85 \
  --experiment_name base_v1_silog085
```

### 2.2 Loss Weight 조정

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --l1_weight 0.3 \
  --silog_weight 0.7 \
  --experiment_name base_v1_custom_weights
```

### 2.3 Pure SI-log Loss 사용

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --criterion SIlog \
  --experiment_name base_v1_silog_only
```

### 2.4 Pure L1 Loss 사용 (비교용)

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --criterion L1 \
  --experiment_name base_v1_l1_only
```

### 2.5 Waveform 사용 (Spectrogram 대신)

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --audio_format waveform \
  --experiment_name base_v1_waveform
```

**⚠️ 주의**: BatvisionV1은 `mel_spectrogram`을 지원하지 않습니다!
```bash
# ❌ 에러 발생
python train.py --dataset batvisionv1 --audio_format mel_spectrogram

# ✅ 올바른 사용
python train.py --dataset batvisionv1 --audio_format spectrogram
python train.py --dataset batvisionv1 --audio_format waveform
```

---

## 3. 고급 실험

### 3.1 최적화된 설정 (권장)

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name base_v1_optimized
```

### 3.2 체크포인트에서 재개

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name base_v1_default \
  --checkpoints 50
```

### 3.3 Validation 빈도 조정

```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --validation_iter 5 \
  --experiment_name base_v1_val5
```

### 3.4 Learning Rate 튜닝

```bash
# Paper default (0.001)
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name base_v1_lr001

# 더 낮은 learning rate
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.0005 \
  --batch_size 128 \
  --experiment_name base_v1_lr0005

# 더 높은 learning rate
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.002 \
  --batch_size 128 \
  --experiment_name base_v1_lr002
```

---

## 4. 비교 실험

### 4.1 Loss 함수 비교

```bash
# L1 only
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --criterion L1 \
  --experiment_name v1_compare_l1

# SI-log only
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --criterion SIlog \
  --experiment_name v1_compare_silog

# Combined (default)
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name v1_compare_combined
```

### 4.2 Audio Format 비교

```bash
# Spectrogram (default)
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name v1_compare_spec

# Waveform
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --audio_format waveform \
  --experiment_name v1_compare_wave
```

### 4.3 V1 vs V2 비교

```bash
# V1
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name compare_v1

# V2
python train.py \
  --dataset batvisionv2 \
  --use_wandb \
  --learning_rate 0.002 \
  --batch_size 256 \
  --experiment_name compare_v2
```

---

## 5. 실험 결과 확인

### W&B 대시보드
```
https://wandb.ai/branden/batvision-depth-estimation
```

### 로컬 결과
- **체크포인트**: `./checkpoints/unet_baseline_batvisionv1_BS128_Lr0.001_AdamW_{experiment_name}/`
- **시각화**: `./results/unet_baseline_batvisionv1_BS128_Lr0.001_AdamW_{experiment_name}/`
- **로그**: `./logs/unet_baseline_batvisionv1_BS128_Lr0.001_AdamW_{experiment_name}/`

### 시각화 확인
```bash
# 최신 결과 확인
ls -lht results/unet_baseline_batvisionv1_*/epoch_*_prediction.png | head -5

# 특정 epoch 확인
open results/unet_baseline_batvisionv1_BS128_Lr0.001_AdamW_base_v1_default/epoch_0050_prediction.png
```

---

## 6. 권장 실험 시퀀스

### Step 1: 기본 실험 (Paper 설정)
```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name v1_step1
```

### Step 2: Best Model 저장 추가
```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name v1_step2
```

### Step 3: SI-log Lambda 튜닝
```bash
# Lambda 0.85 시도
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --save_best_model \
  --learning_rate 0.001 \
  --batch_size 128 \
  --silog_lambda 0.85 \
  --experiment_name v1_step3_lambda085
```

### Step 4: Loss Weight 튜닝
```bash
# SI-log 비중 증가
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --save_best_model \
  --learning_rate 0.001 \
  --batch_size 128 \
  --l1_weight 0.2 \
  --silog_weight 0.8 \
  --experiment_name v1_step4_silog08
```

---

## 7. 문제 해결

### Out of Memory
```bash
# Batch size 줄이기 (V1은 이미 128로 작음)
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 64 \
  --experiment_name v1_bs64
```

### Loss가 발산할 때
```bash
# Learning rate 더 줄이기
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.0005 \
  --batch_size 128 \
  --experiment_name v1_lr0005
```

### Validation이 너무 느릴 때
```bash
# Validation 빈도 줄이기
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --validation_iter 5 \
  --experiment_name v1_val5
```

---

## 8. Config 파일 확인

### 확인 명령어
```bash
# V1 Dataset config
cat conf/dataset/batvisionv1.yaml

# Train config (V1/V2 공통)
cat conf/mode/train.yaml
```

### V1 디폴트 값
```yaml
# conf/dataset/batvisionv1.yaml
audio_format: spectrogram  # mel_spectrogram 미지원!
max_depth: 12.0  # V2는 30.0
depth_norm: True  # V2는 False
images_size: 256

# Paper 권장 (V1 전용)
learning_rate: 0.001  # V2는 0.002
batch_size: 128  # V2는 256
```

---

## 9. 중요한 차이점 요약

### ⚠️ V1에서 사용 불가능한 것들
```bash
# ❌ 에러 발생
python train.py --dataset batvisionv1 --audio_format mel_spectrogram
```

### ✅ V1에서 사용 가능한 Audio Format
- `spectrogram` (디폴트)
- `waveform`

### 📊 V1 특성
- **Max Depth**: 12.0m (V2보다 훨씬 작음)
- **Depth Norm**: True (정규화됨)
- **Paper 설정**: LR=0.001, BS=128

---

## 10. 요약

### 가장 간단한 실행 (Paper 설정)
```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name base_v1_default
```

### 가장 추천하는 설정
```bash
python train.py \
  --dataset batvisionv1 \
  --use_wandb \
  --save_best_model \
  --best_metric rmse \
  --learning_rate 0.001 \
  --batch_size 128 \
  --experiment_name base_v1_recommended
```

### 디버깅용 (빠른 테스트)
```bash
python train.py \
  --dataset batvisionv1 \
  --learning_rate 0.001 \
  --batch_size 32 \
  --experiment_name base_v1_debug
```

### 스크립트로 실행
```bash
./run_base_v1_experiment.sh
```

---

## 11. V1 vs V2 비교표

| 설정 | V1 | V2 |
|------|----|----|
| **실행 스크립트** | `./run_base_v1_experiment.sh` | `./run_base_experiment.sh` |
| **Audio Format** | `spectrogram` | `mel_spectrogram` |
| **Mel 지원** | ❌ | ✅ |
| **Max Depth** | 12.0m | 30.0m |
| **Learning Rate** | 0.001 | 0.002 |
| **Batch Size** | 128 | 256 |
| **Depth Norm** | True | False |
| **Loss** | Combined (L1+SI) | Combined (L1+SI) |

---

## 12. 빠른 참조

```bash
# V1 기본 실행
./run_base_v1_experiment.sh

# V1 + Best Model
python train.py --dataset batvisionv1 --use_wandb --save_best_model --learning_rate 0.001 --batch_size 128 --experiment_name v1_best

# V1 + Waveform
python train.py --dataset batvisionv1 --use_wandb --audio_format waveform --learning_rate 0.001 --batch_size 128 --experiment_name v1_wave

# V1 + Pure SI-log
python train.py --dataset batvisionv1 --use_wandb --criterion SIlog --learning_rate 0.001 --batch_size 128 --experiment_name v1_silog
```






