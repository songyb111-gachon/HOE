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
DATA_PATH = 'data/forward_intensity_tiles'
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
VISUALIZE_FREQ = 10                 # N epoch마다 예측 시각화

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
print(f"   체크포인트 저장: {SAVE_FREQ} epoch마다")
print(f"   예측 시각화: {VISUALIZE_FREQ} epoch마다")
print(f"   Device: {device}")

# %% [markdown]
# ## 3. 데이터 로더 생성

# %%
print("📂 데이터 로딩 중...")

# 직접 train/val 데이터셋 생성 (02번에서 이미 나눈 것 사용)
from torch.utils.data import DataLoader

train_dataset = ForwardPhaseDataset(
    data_path=f'{DATA_PATH}/train',
    normalize=False
)

val_dataset = ForwardPhaseDataset(
    data_path=f'{DATA_PATH}/val',
    normalize=False
)

# 데이터 로더 생성
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

print("\n✅ 데이터 로더 생성 완료!")
print(f"   훈련 샘플: {len(train_dataset)} ({len(train_loader)} 배치)")
print(f"   검증 샘플: {len(val_dataset)} ({len(val_loader)} 배치)")

# 샘플 데이터 확인
sample = next(iter(train_loader))
print(f"\n📊 배치 크기:")
print(f"   입력: {sample['image'].shape}  # (batch, C, H, W)")
print(f"   출력: {sample['target'].shape}")
print(f"   입력 범위: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
print(f"   출력 범위: [{sample['target'].min():.2f}, {sample['target'].max():.2f}]")

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
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    experiment_name=EXPERIMENT_NAME,
    visualize_freq=VISUALIZE_FREQ
)

# 학습 실행
trainer.train(
    num_epochs=NUM_EPOCHS,
    save_freq=SAVE_FREQ
)

# 학습 히스토리 구성
history = {
    'train_loss': trainer.train_losses,
    'val_loss': trainer.val_losses,
    'train_mse': trainer.train_mse,
    'val_mse': trainer.val_mse,
    'train_psnr': trainer.train_psnr,
    'val_psnr': trainer.val_psnr,
    'learning_rate': [optimizer.param_groups[0]['lr']] * NUM_EPOCHS  # 간단한 구현
}

print("\n" + "="*80)
print("🎉 학습 완료!")
print("="*80)

# %% [markdown]
# ## 7. 학습 곡선 시각화

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss 곡선
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MSE 곡선
axes[0, 1].plot(history['train_mse'], label='Train MSE', linewidth=2, color='orange')
axes[0, 1].plot(history['val_mse'], label='Val MSE', linewidth=2, color='red')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Training and Validation MSE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# PSNR 곡선
axes[0, 2].plot(history['train_psnr'], label='Train PSNR', linewidth=2, color='purple')
axes[0, 2].plot(history['val_psnr'], label='Val PSNR', linewidth=2, color='magenta')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('PSNR (dB)')
axes[0, 2].set_title('Training and Validation PSNR')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Learning rate 곡선
axes[1, 0].plot(history['learning_rate'], linewidth=2, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_yscale('log')

# 최종 메트릭 요약
axes[1, 1].axis('off')
summary_text = f"""
📊 학습 최종 결과

Loss:
  • 최종 Train Loss: {history['train_loss'][-1]:.6f}
  • 최종 Val Loss: {history['val_loss'][-1]:.6f}
  • 최고 Val Loss: {trainer.best_val_loss:.6f}

MSE:
  • 최종 Train MSE: {history['train_mse'][-1]:.6f}
  • 최종 Val MSE: {history['val_mse'][-1]:.6f}
  • 최고 Val MSE: {min(history['val_mse']):.6f}

PSNR:
  • 최종 Train PSNR: {history['train_psnr'][-1]:.2f} dB
  • 최종 Val PSNR: {history['val_psnr'][-1]:.2f} dB
  • 최고 Val PSNR: {max(history['val_psnr']):.2f} dB

학습률:
  • 최종 LR: {history['learning_rate'][-1]:.2e}
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')

# 빈 공간
axes[1, 2].axis('off')

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

