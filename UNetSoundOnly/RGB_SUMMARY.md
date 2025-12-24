# RGB Depth Model - Quick Summary

## 생성된 파일들

### 1. 모델 파일
**`models/rgb_depth_model.py`** (새로 생성)
- RGB 입력 (3채널)을 위한 Depth Estimation 모델
- U-Net 기반 encoder-decoder 구조
- Binaural attention 모델과 **feature 크기 호환**
- `return_features=True` 옵션으로 distillation을 위한 중간 feature 추출 가능

```python
# 사용 예제
from models.rgb_depth_model import create_rgb_depth_model
model = create_rgb_depth_model(base_channels=64)

# 일반 forward
depth = model(rgb_image)

# Distillation을 위한 feature 추출
depth, features = model(rgb_image, return_features=True)
# features = {'x1', 'x2', 'x3', 'x4', 'x5', 'd1', 'd2', 'd3', 'd4'}
```

### 2. 학습 스크립트
**`train_rgb_depth.py`** (새로 생성)
- `train_binaural_attention.py`와 유사한 구조
- BatvisionV1/V2 데이터셋 지원
- W&B 로깅 지원
- Checkpoint 저장/로딩 기능

```bash
# 기본 학습
python train_rgb_depth.py --dataset batvisionv2 --batch_size 64 --use_wandb

# Teacher 모델 학습 (distillation용)
python train_rgb_depth.py \
    --dataset batvisionv2 \
    --base_channels 64 \
    --nb_epochs 200 \
    --experiment_name rgb_teacher_for_kd
```

### 3. 실행 예제 스크립트
**`run_rgb_depth_examples.sh`** (새로 생성)
- 8가지 사전 설정된 학습 예제
- Interactive 메뉴 또는 직접 실행 가능

```bash
# Interactive 모드
bash run_rgb_depth_examples.sh

# 특정 예제 실행
bash run_rgb_depth_examples.sh 6  # Teacher model 학습
```

### 4. 호환성 검증 스크립트
**`verify_feature_compatibility.py`** (새로 생성)
- RGB 모델과 Binaural 모델의 feature 차원 호환성 검증
- Distillation 준비 상태 확인

```bash
python verify_feature_compatibility.py
```

### 5. 문서
**`RGB_DEPTH_README.md`** (새로 생성)
- 상세한 사용 설명서
- 아키텍처 비교
- Distillation 가이드
- 예제 코드

---

## 핵심 차이점: RGB vs. Binaural Audio

| 특성 | RGB Model | Binaural Audio Model |
|------|-----------|----------------------|
| **입력** | 3 channels (RGB) | 2 channels (Stereo Audio) |
| **Encoder** | Single encoder | Dual encoder (Left/Right) |
| **특수 모듈** | 없음 | Cross-attention between L/R |
| **파라미터 수** | ~20M (base=64) | ~40M (base=64) |
| **Feature 크기** | ✅ 호환 | ✅ 호환 |

---

## Feature 호환성 (Distillation을 위해 중요!)

두 모델 모두 동일한 feature 차원을 생성:

```
Level x1: [B, 64, 256, 256]     ← RGB encoder == Audio fusion output
Level x2: [B, 128, 128, 128]    ← RGB encoder == Audio fusion output
Level x3: [B, 256, 64, 64]      ← RGB encoder == Audio fusion output
Level x4: [B, 512, 32, 32]      ← RGB encoder == Audio fusion output
Level x5: [B, 512, 16, 16]      ← RGB encoder == Audio fusion output
```

이 호환성 덕분에:
- **Feature-level distillation** 가능
- RGB teacher → Audio student 지식 전달
- 중간 layer의 representation을 직접 매칭

---

## 다음 단계: Distillation 구현

### 1단계: RGB Teacher 학습

```bash
bash run_rgb_depth_examples.sh 6
```

### 2단계: Distillation 코드 작성

`train_distillation.py` 생성 (pseudo-code):

```python
# Teacher (RGB)
teacher = create_rgb_depth_model(base_channels=64)
teacher.load_state_dict(torch.load('checkpoints/rgb_teacher/best_model.pth'))
teacher.eval()

# Student (Audio)
student = create_binaural_attention_model(base_channels=64)

# Training loop
for audio, rgb, depth_gt in dataloader:
    # Teacher prediction
    with torch.no_grad():
        depth_teacher, feats_teacher = teacher(rgb, return_features=True)
    
    # Student prediction (audio만 사용)
    depth_student, feats_student = student(audio, return_features=True)
    
    # Losses
    loss_task = criterion(depth_student, depth_gt)  # Ground truth
    loss_kd = F.mse_loss(depth_student, depth_teacher)  # Depth distillation
    
    # Feature matching
    loss_feat = 0
    for level in ['x1', 'x2', 'x3', 'x4', 'x5']:
        loss_feat += F.mse_loss(feats_student[level], feats_teacher[level])
    
    # Total loss
    loss = loss_task + λ_kd * loss_kd + λ_feat * loss_feat
    loss.backward()
    optimizer.step()
```

### 3단계: Audio 모델 수정

`binaural_attention_model.py`에 feature 반환 기능 추가:

```python
def forward(self, x, return_features=False):
    # ... existing code ...
    
    if return_features:
        features = {
            'x1': left_feats['x1'],  # After fusion
            'x2': left_feats['x2'],
            'x3': left_feats['x3'],
            'x4': left_feats['x4'],
            'x5': left_feats['x5']
        }
        return depth, features
    return depth
```

---

## 실험 제안

### Baseline
1. **RGB only** (upper bound)
2. **Audio only** (baseline)

### Distillation Experiments
3. **Audio + KD (depth)**: Depth 예측만 distillation
4. **Audio + KD (depth + features)**: Depth + intermediate features distillation
5. **Audio + KD (adaptive)**: 학습 진행에 따라 distillation weight 조정

### 예상 결과
```
RGB only:              RMSE = X (best)
Audio only:            RMSE = Y
Audio + KD (depth):    RMSE = Y - δ1
Audio + KD (full):     RMSE = Y - δ2 (δ2 > δ1)
```

---

## 체크리스트

- [x] RGB 모델 구현 (`rgb_depth_model.py`)
- [x] RGB 학습 스크립트 (`train_rgb_depth.py`)
- [x] 실행 예제 스크립트 (`run_rgb_depth_examples.sh`)
- [x] Feature 호환성 검증 (`verify_feature_compatibility.py`)
- [x] 문서 작성 (`RGB_DEPTH_README.md`, `RGB_SUMMARY.md`)
- [ ] RGB teacher 모델 학습
- [ ] Audio 모델에 `return_features` 추가
- [ ] Distillation 학습 스크립트 작성
- [ ] Distillation 실험 및 평가

---

## 빠른 시작

```bash
# 1. Feature 호환성 확인
python verify_feature_compatibility.py

# 2. RGB teacher 모델 학습 시작
bash run_rgb_depth_examples.sh 6

# 3. 학습 모니터링 (W&B 사용 시)
# https://wandb.ai/your-project/batvision-rgb-depth

# 4. Best checkpoint 확인
ls -lh checkpoints/rgb_teacher_for_kd/best_model.pth
```

---

**모든 파일이 준비되었습니다! 🎉**

이제 RGB teacher 모델을 학습하고, distillation을 통해 audio 모델의 성능을 향상시킬 수 있습니다.






