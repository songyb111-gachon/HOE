# %% [markdown]
# # 🎓 Inverse Design U-Net 모델 학습
#
# 256×256 타일로 Inverse Design U-Net 모델을 학습합니다.
#
# **모델:**
# - Input: Phase Map (목표 위상 맵)
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

from models import InverseDesignUNet
from datasets import InverseDesignDataset, create_dataloaders
from utils import WeightedBCELoss, Trainer

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

# 데이터 로더 생성
train_loader, val_loader, test_loader = create_dataloaders(
    dataset_path=DATA_PATH,
    dataset_type='inverse',
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
sample_batch = next(iter(train_loader))
print(f"\n📊 샘플 배치 크기:")
print(f"   Input (Phase Map): {sample_batch['image'].shape}")  # [B, 1, H, W]
print(f"   Target (Pillar): {sample_batch['target'].shape}")    # [B, 1, H, W]
print(f"   Input range: [{sample_batch['image'].min():.2f}, {sample_batch['image'].max():.2f}]")
print(f"   Target range: [{sample_batch['target'].min():.2f}, {sample_batch['target'].max():.2f}]")

# %% [markdown]
# ## 4. 모델 생성

# %%
print("🔨 모델 생성 중...")

# Inverse Design U-Net 모델
model = InverseDesignUNet(
    in_channels=1,
    out_channels=1,
    layer_num=LAYER_NUM,
    base_features=BASE_FEATURES,
    dropout_rate=DROPOUT_RATE,
    use_batchnorm=USE_BATCHNORM
).to(device)

print(f"\n✅ 모델 생성 완료!")
print(f"   모델: InverseDesignUNet")
print(f"   레이어 수: {LAYER_NUM}")
print(f"   기본 features: {BASE_FEATURES}")
print(f"   Dropout: {DROPOUT_RATE}")
print(f"   BatchNorm: {USE_BATCHNORM}")

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
    criterion = WeightedBCELoss(pillar_weight=PILLAR_WEIGHT).to(device)
    print(f"✅ 손실 함수: Weighted BCE Loss (pillar_weight={PILLAR_WEIGHT})")
else:
    criterion = nn.BCEWithLogitsLoss().to(device)
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

# 체크포인트 디렉토리
checkpoint_dir = Path(CHECKPOINT_DIR) / EXPERIMENT_NAME
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# 학습 히스토리
history = {
    'train_loss': [],
    'val_loss': [],
    'learning_rate': []
}

best_val_loss = float('inf')

# %% [markdown]
# ### 학습 루프

# %%
for epoch in range(NUM_EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    print(f"{'='*80}")
    
    # ==================== 훈련 ====================
    model.train()
    train_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # 데이터 이동
        inputs = batch['image'].to(device)      # Phase map
        targets = batch['target'].to(device)    # Pillar pattern
        
        # Forward
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Loss
        loss = criterion(outputs, targets)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # 진행상황 출력
        if (batch_idx + 1) % 50 == 0:
            print(f"  [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    avg_train_loss = train_loss / len(train_loader)
    history['train_loss'].append(avg_train_loss)
    
    # ==================== 검증 ====================
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['image'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    history['val_loss'].append(avg_val_loss)
    history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    # Learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # 결과 출력
    print(f"\n  📊 Epoch {epoch+1} 결과:")
    print(f"     Train Loss: {avg_train_loss:.6f}")
    print(f"     Val Loss:   {avg_val_loss:.6f}")
    print(f"     LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # 최고 성능 모델 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
        }, checkpoint_dir / 'best_model.pth')
        print(f"     ✅ 최고 성능 모델 저장! (Val Loss: {avg_val_loss:.6f})")
    
    # 주기적 체크포인트 저장
    if (epoch + 1) % SAVE_FREQ == 0:
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'train_loss': avg_train_loss,
        }, checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth')
        print(f"     💾 체크포인트 저장: epoch_{epoch+1}")

print("\n" + "="*80)
print("✅ 학습 완료!")
print("="*80)
print(f"   최고 검증 손실: {best_val_loss:.6f}")
print(f"   모델 저장 위치: {checkpoint_dir}")

# %% [markdown]
# ## 7. 학습 곡선 시각화

# %%
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Learning Rate
axes[1].plot(history['learning_rate'], linewidth=2, color='orange')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].set_title('Learning Rate Schedule')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(checkpoint_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✅ 학습 곡선 저장: {checkpoint_dir / 'training_curves.png'}")

# %% [markdown]
# ## 8. 검증 세트에서 예측 시각화

# %%
print("\n" + "="*80)
print("📊 검증 세트 예측 시각화")
print("="*80)

# 최고 성능 모델 로드
checkpoint = torch.load(checkpoint_dir / 'best_model.pth')
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
    axes[i, 0].set_title(f'Input: Phase Map\nRange: [{input_img.min():.2f}, {input_img.max():.2f}]')
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
plt.savefig(checkpoint_dir / 'validation_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ 검증 예측 저장: {checkpoint_dir / 'validation_predictions.png'}")

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
print(f"   {checkpoint_dir / 'best_model.pth'}")
print(f"   {checkpoint_dir / 'training_curves.png'}")
print(f"   {checkpoint_dir / 'validation_predictions.png'}")
print(f"\n🚀 다음 단계: 07_inverse_design_notebook.py 실행!")

