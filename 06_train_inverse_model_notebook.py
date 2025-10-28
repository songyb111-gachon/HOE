# %% [markdown]
# # 🎓 Inverse Design U-Net 모델 학습
#
# 256×256 타일로 Inverse Design U-Net 모델을 학습합니다.
#
# **모델:**
# - Input: EM Intensity Map (목표 위상 맵)
# - Output: Pillar Pattern (그것을 만들어낼 필러 패턴)
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

from models import InverseUNet
from datasets import InverseDesignDataset, create_dataloaders
from utils import Trainer

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
DATA_PATH = 'data/inverse_tiles/train'
BATCH_SIZE = 16                    # 타일 기반이므로 더 큰 배치 사용 가능
NUM_WORKERS = 4                     # 데이터 로딩 워커

# ==================== 모델 파라미터 ====================
LAYER_NUM = 5                       # U-Net 레이어 수
BASE_FEATURES = 64                  # 기본 feature 수
DROPOUT_RATE = 0.2                  # Dropout 비율
USE_BATCHNORM = True                # BatchNorm 사용 여부

# ==================== 학습 파라미터 ====================
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
LOSS_TYPE = 'weighted_bce'          # 'bce', 'weighted_bce'
PILLAR_WEIGHT = 2.0                 # Pillar 클래스 가중치 (pillar가 더 중요)

# ==================== 체크포인트 파라미터 ====================
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
EXPERIMENT_NAME = 'inverse_design_basic_tiles'
SAVE_FREQ = 5                       # N epoch마다 저장

# 디렉토리 생성
Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
Path(LOG_DIR).mkdir(exist_ok=True)

print("✅ 파라미터 설정 완료!")
print(f"\n📊 학습 설정:")
print(f"   데이터 경로: {DATA_PATH}")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   학습률: {LEARNING_RATE}")
print(f"   손실 함수: {LOSS_TYPE}")
print(f"   Pillar 가중치: {PILLAR_WEIGHT}")
print(f"   Device: {device}")

# %% [markdown]
# ## 3. 데이터 로더 생성

# %%
print("📂 데이터 로딩 중...")

from torch.utils.data import DataLoader

# 데이터셋 생성 (이미 train/val로 나뉘어져 있음)
train_dataset = InverseDesignDataset(
    data_path='data/inverse_tiles/train',
    input_extension='npy',
    output_extension='png',
    normalize=False
)

val_dataset = InverseDesignDataset(
    data_path='data/inverse_tiles/val',
    input_extension='npy',
    output_extension='png',
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
sample_batch = next(iter(train_loader))
print(f"\n📊 샘플 배치 크기:")
print(f"   Input (EM Intensity Map): {sample_batch['image'].shape}")  # [B, 1, H, W]
print(f"   Target (Pillar): {sample_batch['target'].shape}")    # [B, 1, H, W]
print(f"   Input range: [{sample_batch['image'].min():.2f}, {sample_batch['image'].max():.2f}]")
print(f"   Target range: [{sample_batch['target'].min():.2f}, {sample_batch['target'].max():.2f}]")

# %% [markdown]
# ## 4. 모델 생성

# %%
print("🔨 모델 생성 중...")

# Inverse Design U-Net 모델
model = InverseUNet(
    in_channels=1,
    out_channels=[1],
    layer_num=LAYER_NUM,
    base_features=BASE_FEATURES,
    dropout_rate=DROPOUT_RATE,
    output_activations=['linear'],  # BCEWithLogitsLoss를 위해 linear 사용
    use_batchnorm=USE_BATCHNORM
).to(device)

print(f"\n✅ 모델 생성 완료!")
print(f"   모델: InverseUNet")
print(f"   레이어 수: {LAYER_NUM}")
print(f"   기본 features: {BASE_FEATURES}")
print(f"   Dropout: {DROPOUT_RATE}")
print(f"   BatchNorm: {USE_BATCHNORM}")

# 모델 요약 출력
model.get_model_summary()

# 모델 파라미터 수 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 모델 파라미터:")
print(f"   총 파라미터: {total_params:,}")
print(f"   학습 가능 파라미터: {trainable_params:,}")

# %% [markdown]
# ## 5. 손실 함수 및 옵티마이저 설정

# %%
# 손실 함수
if LOSS_TYPE == 'weighted_bce':
    # pos_weight: pillar 클래스(1)에 더 높은 가중치 부여
    pos_weight = torch.tensor([PILLAR_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"✅ 손실 함수: Weighted BCE Loss (pillar_weight={PILLAR_WEIGHT})")
else:
    criterion = nn.BCEWithLogitsLoss()
    print(f"✅ 손실 함수: BCE Loss")

# 옵티마이저
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(f"✅ 옵티마이저: Adam (lr={LEARNING_RATE})")

# Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=10, 
    verbose=True
)
print(f"✅ Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")

# %% [markdown]
# ## 6. 학습 시작

# %%
print("\n" + "="*80)
print("🚀 학습 시작!")
print("="*80)

# Trainer를 사용한 학습
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    checkpoint_dir=CHECKPOINT_DIR,
    log_dir=LOG_DIR,
    experiment_name=EXPERIMENT_NAME
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
    'learning_rate': [optimizer.param_groups[0]['lr']] * NUM_EPOCHS
}

print("\n" + "="*80)
print("✅ 학습 완료!")
print("="*80)
print(f"   최고 검증 손실: {trainer.best_val_loss:.6f}")
print(f"   모델 저장 위치: {trainer.checkpoint_dir}")

# %% [markdown]
# ## 7. 학습 곡선 시각화

# %%
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss (BCE)
axes[0, 0].plot(history['train_loss'], label='Train Loss (BCE)', linewidth=2)
axes[0, 0].plot(history['val_loss'], label='Val Loss (BCE)', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and Validation Loss (BCE)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# MSE
axes[0, 1].plot(history['train_mse'], label='Train MSE', linewidth=2, color='orange')
axes[0, 1].plot(history['val_mse'], label='Val MSE', linewidth=2, color='red')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('Training and Validation MSE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# PSNR
axes[0, 2].plot(history['train_psnr'], label='Train PSNR', linewidth=2, color='purple')
axes[0, 2].plot(history['val_psnr'], label='Val PSNR', linewidth=2, color='magenta')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('PSNR (dB)')
axes[0, 2].set_title('Training and Validation PSNR')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Learning Rate
axes[1, 0].plot(history['learning_rate'], linewidth=2, color='green')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Learning Rate')
axes[1, 0].set_title('Learning Rate Schedule')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# 최종 메트릭 요약
axes[1, 1].axis('off')
summary_text = f"""
📊 학습 최종 결과

BCE Loss:
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
plt.savefig(trainer.checkpoint_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ 학습 곡선 저장: {trainer.checkpoint_dir / 'training_curves.png'}")

# %% [markdown]
# ## 8. 검증 세트에서 예측 시각화

# %%
print("\n" + "="*80)
print("📊 검증 세트 예측 시각화")
print("="*80)

# 최고 성능 모델 로드
checkpoint = torch.load(trainer.checkpoint_dir / 'best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 검증 샘플 가져오기
val_samples = next(iter(val_loader))
inputs = val_samples['image'].to(device)
targets = val_samples['target'].to(device)

with torch.no_grad():
    outputs = model(inputs)
    predictions = torch.sigmoid(outputs)  # [0, 1] 확률로 변환
    binary_predictions = (predictions > 0.5).float()  # 이진화

# 시각화 (4개 샘플)
num_samples = min(4, inputs.shape[0])
fig, axes = plt.subplots(num_samples, 4, figsize=(16, num_samples*4))

if num_samples == 1:
    axes = axes.reshape(1, -1)

for i in range(num_samples):
    # Input: Phase map
    input_img = inputs[i, 0].cpu().numpy()
    axes[i, 0].imshow(input_img, cmap='twilight')
    axes[i, 0].set_title(f'Input: EM Intensity Map\nRange: [{input_img.min():.2f}, {input_img.max():.2f}]')
    axes[i, 0].axis('off')
    
    # Target: Pillar pattern
    target_img = targets[i, 0].cpu().numpy()
    axes[i, 1].imshow(target_img, cmap='gray')
    axes[i, 1].set_title(f'Target: Pillar Pattern\nRange: [{target_img.min():.2f}, {target_img.max():.2f}]')
    axes[i, 1].axis('off')
    
    # Prediction: Probability map
    pred_prob = predictions[i, 0].cpu().numpy()
    axes[i, 2].imshow(pred_prob, cmap='gray', vmin=0, vmax=1)
    axes[i, 2].set_title(f'Prediction: Probability\nRange: [{pred_prob.min():.2f}, {pred_prob.max():.2f}]')
    axes[i, 2].axis('off')
    
    # Prediction: Binary
    pred_binary = binary_predictions[i, 0].cpu().numpy()
    axes[i, 3].imshow(pred_binary, cmap='gray')
    axes[i, 3].set_title(f'Prediction: Binary (>0.5)\nPillar ratio: {pred_binary.mean():.2%}')
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig(trainer.checkpoint_dir / 'validation_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ 검증 예측 저장: {trainer.checkpoint_dir / 'validation_predictions.png'}")

# %% [markdown]
# ## 9. 완료!
#
# Inverse Design 모델 학습이 완료되었습니다! 🎉
#
# **다음 단계:**
# - `07_inverse_design_notebook.py`: 원하는 phase map으로부터 pillar pattern 설계

# %%
print("\n" + "="*80)
print("🎉 Inverse Design 모델 학습 완료!")
print("="*80)
print(f"\n📂 저장된 파일:")
print(f"   {trainer.checkpoint_dir / 'best_model.pth'}")
print(f"   {trainer.checkpoint_dir / 'training_curves.png'}")
print(f"   {trainer.checkpoint_dir / 'validation_predictions.png'}")
print(f"\n🚀 다음 단계: 07_inverse_design_notebook.py 실행!")

