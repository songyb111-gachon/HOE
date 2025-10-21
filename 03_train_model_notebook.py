# %% [markdown]
# # 🎓 U-Net 모델 학습
#
# 256×256 타일로 Forward EM Near-Field Intensity Prediction U-Net 모델을 학습합니다.
#
# **Output**: EM Near-Field Intensity (|Ex|² + |Ey|² + |Ez|²)
#
# ## 📋 목차
# 1. 환경 설정 및 임포트
# 2. 파라미터 설정
# 3. 데이터 로더 생성
# 4. 모델 생성
# 5. 학습
# 6. 결과 시각화

# %% [markdown]
# ## 1. 환경 설정 및 임포트

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# PyTorch 코드 경로 추가
sys.path.append('pytorch_codes')

from models import ForwardPhaseUNet, MultiScalePhaseUNet, PhaseAmplitudeUNet
from datasets import ForwardPhaseDataset, create_dataloaders
from utils import WeightedMSELoss, Trainer

# GPU 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ PyTorch 설정 완료!")
print(f"   Device: {device}")
print(f"   PyTorch 버전: {torch.__version__}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. 파라미터 설정

# %%
# ==================== 데이터 파라미터 ====================
DATA_PATH = 'data/forward_intensity_tiles/train'
BATCH_SIZE = 16                    # 타일 기반이므로 더 큰 배치 사용 가능
NUM_WORKERS = 4                     # 데이터 로딩 워커

# ==================== 모델 파라미터 ====================
MODEL_TYPE = 'basic'                # 'basic', 'multiscale', 'phase_amplitude'
LAYER_NUM = 5                       # U-Net 레이어 수
BASE_FEATURES = 64                  # 기본 feature 수
DROPOUT_RATE = 0.2                  # Dropout 비율
USE_BATCHNORM = True                # BatchNorm 사용 여부

# ==================== 학습 파라미터 ====================
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
LOSS_TYPE = 'mse'                   # 'mse', 'weighted_mse'

# ==================== 체크포인트 파라미터 ====================
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
EXPERIMENT_NAME = f'forward_phase_{MODEL_TYPE}_tiles'
SAVE_FREQ = 5                       # N epoch마다 저장

# 디렉토리 생성
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
Path(LOG_DIR).mkdir(exist_ok=True)

print("✅ 파라미터 설정 완료!")
print(f"\n📊 학습 설정:")
print(f"   데이터 경로: {DATA_PATH}")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   모델 타입: {MODEL_TYPE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   학습률: {LEARNING_RATE}")
print(f"   Device: {device}")

# %% [markdown]
# ## 3. 데이터 로더 생성

# %%
print("📂 데이터 로딩 중...")

# 데이터 로더 생성
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_path=DATA_PATH,
    dataset_type='forward_phase',
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_split=0.8,
    val_split=0.2,
    normalize=False
)

print("\n✅ 데이터 로더 생성 완료!")
print(f"   훈련 배치: {len(train_loader)}")
print(f"   검증 배치: {len(val_loader)}")

# 샘플 데이터 확인
sample = next(iter(train_loader))
print(f"\n📊 배치 크기:")
print(f"   입력: {sample['image'].shape}  # (batch, C, H, W)")
print(f"   출력: {sample['target'].shape}")

# %% [markdown]
# ## 4. 모델 생성

# %%
print("🔨 모델 생성 중...")

# 모델 생성
if MODEL_TYPE == 'basic':
    model = ForwardPhaseUNet(
        in_channels=1,
        out_channels=1,
        layer_num=LAYER_NUM,
        base_features=BASE_FEATURES,
        dropout_rate=DROPOUT_RATE,
        output_activation='linear',
        use_batchnorm=USE_BATCHNORM
    )
elif MODEL_TYPE == 'multiscale':
    model = MultiScalePhaseUNet(
        in_channels=1,
        out_channels=1,
        layer_num=LAYER_NUM,
        base_features=BASE_FEATURES,
        dropout_rate=DROPOUT_RATE,
        use_batchnorm=USE_BATCHNORM
    )
elif MODEL_TYPE == 'phase_amplitude':
    model = PhaseAmplitudeUNet(
        in_channels=1,
        layer_num=LAYER_NUM,
        base_features=BASE_FEATURES,
        dropout_rate=DROPOUT_RATE,
        use_batchnorm=USE_BATCHNORM
    )

model = model.to(device)

# 모델 정보 출력
print("\n✅ 모델 생성 완료!")
model.get_model_summary()

# 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 파라미터 수:")
print(f"   전체: {total_params:,}")
print(f"   학습 가능: {trainable_params:,}")

# %% [markdown]
# ## 5. Loss Function 및 Optimizer 설정

# %%
# Loss function
if LOSS_TYPE == 'mse':
    criterion = nn.MSELoss()
elif LOSS_TYPE == 'weighted_mse':
    criterion = WeightedMSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10, verbose=True
)

print("✅ Loss function 및 Optimizer 설정 완료!")
print(f"   Loss: {LOSS_TYPE}")
print(f"   Optimizer: Adam")
print(f"   Learning rate: {LEARNING_RATE}")

# %% [markdown]
# ## 6. 학습

# %%
print("\n" + "="*80)
print("🚀 학습 시작!")
print("="*80)

# Trainer 생성
trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    experiment_name=EXPERIMENT_NAME
)

# 학습 실행
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=NUM_EPOCHS,
    scheduler=scheduler,
    save_freq=SAVE_FREQ
)

print("\n" + "="*80)
print("🎉 학습 완료!")
print("="*80)

# %% [markdown]
# ## 7. 학습 곡선 시각화

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss 곡선
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning rate 곡선
axes[1].plot(history['learning_rate'], linewidth=2, color='green')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Learning Rate Schedule')
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.show()

print("\n✅ 학습 곡선 시각화 완료!")

# %% [markdown]
# ## 8. 검증 세트 예측 샘플

# %%
print("\n🔍 검증 세트에서 예측 샘플 생성 중...")

model.eval()
with torch.no_grad():
    # 검증 배치 하나 가져오기
    val_batch = next(iter(val_loader))
    inputs = val_batch['image'].to(device)
    targets = val_batch['target'].to(device)
    
    # 예측
    predictions = model(inputs)
    
    # CPU로 이동
    inputs = inputs.cpu().numpy()
    targets = targets.cpu().numpy()
    predictions = predictions.cpu().numpy()

# 처음 4개 샘플 시각화
num_to_show = min(4, len(inputs))

fig, axes = plt.subplots(num_to_show, 4, figsize=(16, 4*num_to_show))
if num_to_show == 1:
    axes = axes.reshape(1, -1)

for idx in range(num_to_show):
    # 입력
    axes[idx, 0].imshow(inputs[idx, 0], cmap='gray')
    axes[idx, 0].set_title(f'Sample {idx}: Input')
    axes[idx, 0].axis('off')
    
    # Ground Truth
    im1 = axes[idx, 1].imshow(targets[idx, 0], cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[idx, 1].set_title(f'Sample {idx}: Ground Truth')
    axes[idx, 1].axis('off')
    plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046)
    
    # Prediction
    im2 = axes[idx, 2].imshow(predictions[idx, 0], cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[idx, 2].set_title(f'Sample {idx}: Prediction')
    axes[idx, 2].axis('off')
    plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046)
    
    # Error map
    error = np.abs(targets[idx, 0] - predictions[idx, 0])
    im3 = axes[idx, 3].imshow(error, cmap='hot')
    axes[idx, 3].set_title(f'Sample {idx}: Error\nMAE={np.mean(error):.3f}')
    axes[idx, 3].axis('off')
    plt.colorbar(im3, ax=axes[idx, 3], fraction=0.046)

plt.tight_layout()
plt.show()

print(f"\n✅ {num_to_show}개 샘플 예측 시각화 완료!")

# %% [markdown]
# ## 9. 다음 단계
#
# 모델 학습이 완료되었습니다! 다음 노트북으로 이동하세요:
#
# **`04_sliding_window_prediction_notebook.py`**: 대형 이미지 예측

# %%

