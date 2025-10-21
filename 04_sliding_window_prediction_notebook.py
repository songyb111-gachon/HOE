# %% [markdown]
# # 🔮 Sliding Window Prediction
#
# 학습된 모델로 대형 이미지 (4096×4096)를 슬라이딩 윈도우 방식으로 예측합니다.
#
# ## 📋 목차
# 1. 환경 설정 및 임포트
# 2. 파라미터 설정
# 3. 모델 로드
# 4. 슬라이딩 윈도우 예측
# 5. 결과 시각화 및 저장

# %% [markdown]
# ## 1. 환경 설정 및 임포트

# %%
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm

# PyTorch 코드 경로 추가
sys.path.append('pytorch_codes')

from models import ForwardPhaseUNet, MultiScalePhaseUNet, PhaseAmplitudeUNet

# GPU 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ 환경 설정 완료!")
print(f"   Device: {device}")
print(f"   PyTorch 버전: {torch.__version__}")

# %% [markdown]
# ## 2. 파라미터 설정

# %%
# ==================== 입력 파라미터 ====================
INPUT_MASK_PATH = 'data/forward_phase/inputs/sample_0000.png'  # 예측할 이미지
CHECKPOINT_PATH = 'checkpoints/forward_phase_basic_tiles/best_model.pth'  # 학습된 모델
OUTPUT_DIR = 'predictions'                                      # 출력 디렉토리

# ==================== 모델 파라미터 ====================
MODEL_TYPE = 'basic'            # 'basic', 'multiscale', 'phase_amplitude'

# ==================== 슬라이딩 윈도우 파라미터 ====================
TILE_SIZE = 256                 # 타일 크기 (학습 시와 동일해야 함)
STRIDE = 64                     # 슬라이딩 stride (작을수록 더 정확, 느림)

# 출력 디렉토리 생성
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("✅ 파라미터 설정 완료!")
print(f"\n📊 예측 설정:")
print(f"   입력 이미지: {INPUT_MASK_PATH}")
print(f"   체크포인트: {CHECKPOINT_PATH}")
print(f"   타일 크기: {TILE_SIZE}×{TILE_SIZE}")
print(f"   Stride: {STRIDE}")
print(f"   Device: {device}")

# %% [markdown]
# ## 3. 입력 이미지 로드

# %%
print("\n📂 입력 이미지 로딩 중...")

input_mask = cv2.imread(INPUT_MASK_PATH, cv2.IMREAD_GRAYSCALE)

if input_mask is None:
    raise ValueError(f"Failed to load image: {INPUT_MASK_PATH}")

h, w = input_mask.shape

print(f"✅ 입력 이미지 로드 완료!")
print(f"   크기: {w}×{h}")
print(f"   Fill ratio: {np.sum(input_mask > 128) / input_mask.size * 100:.1f}%")

# 입력 이미지 시각화
plt.figure(figsize=(10, 10))
plt.imshow(input_mask, cmap='gray')
plt.title(f'Input Mask\n{w}×{h}')
plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. 모델 로드

# %%
print("\n📥 모델 로딩 중...")

# 모델 생성
if MODEL_TYPE == 'basic':
    model = ForwardPhaseUNet(in_channels=1, out_channels=1)
elif MODEL_TYPE == 'multiscale':
    model = MultiScalePhaseUNet(in_channels=1, out_channels=1)
elif MODEL_TYPE == 'phase_amplitude':
    model = PhaseAmplitudeUNet(in_channels=1)

# 체크포인트 로드
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

print(f"✅ 모델 로드 완료!")
print(f"   모델 타입: {MODEL_TYPE}")
print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
print(f"   Val Loss: {checkpoint.get('val_loss', 'unknown')}")

# %% [markdown]
# ## 5. 슬라이딩 윈도우 예측

# %%
print("\n" + "="*80)
print("🔍 슬라이딩 윈도우 예측 시작")
print("="*80)

# 예측 맵 및 카운트 맵 초기화
prediction_map = np.zeros((h, w), dtype=np.float32)
count_map = np.zeros((h, w), dtype=np.int32)

# 타일 개수 계산
n_tiles_y = (h - TILE_SIZE) // STRIDE + 1
n_tiles_x = (w - TILE_SIZE) // STRIDE + 1
total_tiles = n_tiles_y * n_tiles_x

print(f"\n📐 슬라이딩 윈도우 정보:")
print(f"   이미지 크기: {w}×{h}")
print(f"   타일 크기: {TILE_SIZE}×{TILE_SIZE}")
print(f"   Stride: {STRIDE}")
print(f"   타일 개수: {n_tiles_y}×{n_tiles_x} = {total_tiles:,}")
print(f"\n진행 중...\n")

# 슬라이딩 윈도우 예측
with torch.no_grad():
    pbar = tqdm(total=total_tiles, desc="Processing tiles")
    
    for i in range(n_tiles_y):
        for j in range(n_tiles_x):
            # 타일 위치 계산
            top = i * STRIDE
            left = j * STRIDE
            bottom = min(top + TILE_SIZE, h)
            right = min(left + TILE_SIZE, w)
            
            # 경계 처리
            if bottom - top < TILE_SIZE:
                top = max(0, bottom - TILE_SIZE)
            if right - left < TILE_SIZE:
                left = max(0, right - TILE_SIZE)
            
            # 타일 추출
            tile = input_mask[top:bottom, left:right]
            
            # 텐서로 변환
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float()
            tile_tensor = tile_tensor / 255.0  # Normalize to [0, 1]
            tile_tensor = tile_tensor.to(device)
            
            # 예측
            output = model(tile_tensor)
            
            # 결과 추출
            pred = output.squeeze().cpu().numpy()
            
            # 예측 맵에 추가
            prediction_map[top:bottom, left:right] += pred
            count_map[top:bottom, left:right] += 1
            
            pbar.update(1)
    
    pbar.close()

# 평균화
count_map = np.maximum(count_map, 1)  # 0으로 나누기 방지
prediction_map = prediction_map / count_map

print(f"\n✅ 예측 완료!")
print(f"   픽셀당 평균 예측 횟수: {np.mean(count_map):.1f}")
print(f"   최소 예측 횟수: {np.min(count_map)}")
print(f"   최대 예측 횟수: {np.max(count_map)}")

# %% [markdown]
# ## 6. 결과 저장

# %%
print("\n💾 결과 저장 중...")

output_path = Path(OUTPUT_DIR)

# 예측 위상맵 저장
phase_path = output_path / 'predicted_phase_map.npy'
np.save(phase_path, prediction_map.astype(np.float32))
print(f"   ✓ Phase map: {phase_path}")

# 카운트 맵 저장
count_path = output_path / 'count_map.npy'
np.save(count_path, count_map.astype(np.int32))
print(f"   ✓ Count map: {count_path}")

print("\n✅ 결과 저장 완료!")

# %% [markdown]
# ## 7. 결과 시각화

# %%
print("\n🎨 결과 시각화 중...\n")

fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# 입력 마스크
axes[0, 0].imshow(input_mask, cmap='gray')
axes[0, 0].set_title('Input: Random Pillar Mask', fontsize=14, fontweight='bold')
axes[0, 0].axis('off')

# 예측 위상맵
im1 = axes[0, 1].imshow(prediction_map, cmap='hsv', vmin=-np.pi, vmax=np.pi)
axes[0, 1].set_title('Predicted Phase Map', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')
plt.colorbar(im1, ax=axes[0, 1], label='Phase (rad)', fraction=0.046)

# 카운트 맵 (overlap 정보)
im2 = axes[1, 0].imshow(count_map, cmap='viridis')
axes[1, 0].set_title('Prediction Count Map\n(Overlapping Predictions)', fontsize=14, fontweight='bold')
axes[1, 0].axis('off')
plt.colorbar(im2, ax=axes[1, 0], label='Count', fraction=0.046)

# 위상 히스토그램
axes[1, 1].hist(prediction_map.flatten(), bins=50, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Phase (rad)', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)
axes[1, 1].set_title('Phase Distribution', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=np.mean(prediction_map), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(prediction_map):.3f}')
axes[1, 1].legend()

plt.tight_layout()

# 시각화 저장
vis_path = output_path / 'prediction_visualization.png'
plt.savefig(vis_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Visualization: {vis_path}")

plt.show()

print("\n✅ 시각화 완료!")

# %% [markdown]
# ## 8. 통계 정보

# %%
print("\n" + "="*80)
print("📊 예측 통계")
print("="*80)
print(f"\n위상맵:")
print(f"   평균: {np.mean(prediction_map):.4f} rad ({np.mean(prediction_map)/np.pi:.2f}π)")
print(f"   표준편차: {np.std(prediction_map):.4f} rad ({np.std(prediction_map)/np.pi:.2f}π)")
print(f"   최소값: {np.min(prediction_map):.4f} rad ({np.min(prediction_map)/np.pi:.2f}π)")
print(f"   최대값: {np.max(prediction_map):.4f} rad ({np.max(prediction_map)/np.pi:.2f}π)")
print(f"   범위: {np.max(prediction_map) - np.min(prediction_map):.4f} rad")

print(f"\n카운트 맵:")
print(f"   평균 예측 횟수: {np.mean(count_map):.1f}")
print(f"   최소 예측 횟수: {np.min(count_map)}")
print(f"   최대 예측 횟수: {np.max(count_map)}")

print(f"\n출력 파일:")
print(f"   📁 {output_path}/")
print(f"      ├── predicted_phase_map.npy")
print(f"      ├── count_map.npy")
print(f"      └── prediction_visualization.png")

# %% [markdown]
# ## 9. Ground Truth와 비교 (선택사항)
#
# Ground Truth가 있는 경우 비교합니다.

# %%
# Ground Truth 경로
gt_path = INPUT_MASK_PATH.replace('inputs', 'outputs').replace('.png', '.npy')

if Path(gt_path).exists():
    print("\n📊 Ground Truth와 비교 중...")
    
    # Ground Truth 로드
    ground_truth = np.load(gt_path)
    
    # 크기가 다르면 조정
    if ground_truth.shape != prediction_map.shape:
        print(f"   ⚠️  크기 불일치: GT {ground_truth.shape} vs Pred {prediction_map.shape}")
        print(f"   Ground Truth를 예측 크기로 리사이즈합니다...")
        from scipy import ndimage
        zoom_factors = (prediction_map.shape[0] / ground_truth.shape[0],
                       prediction_map.shape[1] / ground_truth.shape[1])
        ground_truth = ndimage.zoom(ground_truth, zoom_factors, order=1)
    
    # 에러 계산
    mae = np.mean(np.abs(prediction_map - ground_truth))
    mse = np.mean((prediction_map - ground_truth)**2)
    rmse = np.sqrt(mse)
    
    print(f"\n   에러 메트릭:")
    print(f"      MAE:  {mae:.4f} rad ({mae/np.pi:.3f}π)")
    print(f"      MSE:  {mse:.4f}")
    print(f"      RMSE: {rmse:.4f} rad ({rmse/np.pi:.3f}π)")
    
    # 비교 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Ground Truth
    im1 = axes[0, 0].imshow(ground_truth, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0, 0].set_title('Ground Truth (MEEP)', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], label='Phase (rad)', fraction=0.046)
    
    # Prediction
    im2 = axes[0, 1].imshow(prediction_map, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[0, 1].set_title('Prediction (U-Net)', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], label='Phase (rad)', fraction=0.046)
    
    # Error map
    error = np.abs(prediction_map - ground_truth)
    im3 = axes[1, 0].imshow(error, cmap='hot')
    axes[1, 0].set_title(f'Absolute Error\nMAE = {mae:.4f} rad', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], label='Error (rad)', fraction=0.046)
    
    # Error histogram
    axes[1, 1].hist(error.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Absolute Error (rad)', fontsize=12)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].axvline(x=mae, color='red', linestyle='--', linewidth=2, label=f'MAE: {mae:.3f}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 비교 저장
    comparison_path = output_path / 'comparison_with_gt.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n   ✓ 비교 시각화: {comparison_path}")
    
    plt.show()
    
    print("\n✅ Ground Truth 비교 완료!")
else:
    print(f"\n⚠️  Ground Truth를 찾을 수 없습니다: {gt_path}")

# %% [markdown]
# ## 10. 완료!
#
# 예측이 완료되었습니다! 🎉
#
# 결과 파일은 `predictions/` 디렉토리에 저장되었습니다.

# %%

