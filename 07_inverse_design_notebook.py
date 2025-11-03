# %% [markdown]
# # 🔮 Inverse Design Prediction (Sliding Window)
#
# 학습된 Inverse Design 모델로 대형 intensity map (2048×2048)으로부터  
# pillar pattern을 설계합니다.
#
# **데이터 흐름:**
# - Input: 목표 Phase Map (2048×2048 .npy)
# - Output: 설계된 Pillar Pattern (2048×2048 PNG)
#
# ## 📋 목차
# 1. 환경 설정 및 임포트
# 2. 파라미터 설정
# 3. 모델 로드
# 4. 슬라이딩 윈도우 예측
# 5. 이진화 및 결과 시각화

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

from models import InverseUNet

# GPU 확인
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ 환경 설정 완료!")
print(f"   Device: {device}")
print(f"   PyTorch 버전: {torch.__version__}")

# %% [markdown]
# ## 2. 파라미터 설정

# %%
# ==================== 입력 파라미터 ====================
INPUT_PHASE_PATH = 'data/forward_intensity/outputs/sample_0000.npy'  # 목표 intensity map
OUTPUT_DIR = 'predictions/inverse'                               # 출력 디렉토리

# ==================== 모델 파라미터 (학습 시와 동일해야 함) ====================
LAYER_NUM = 5                       # U-Net 레이어 수
BASE_FEATURES = 64                  # 기본 feature 수
DROPOUT_RATE = 0.2                  # Dropout 비율
USE_BATCHNORM = True                # BatchNorm 사용 여부
EXPERIMENT_NAME = 'inverse_design_basic_tiles'
CHECKPOINT_PATH = f'checkpoints/{EXPERIMENT_NAME}/best_model.pth'  # 학습된 모델

# ==================== 슬라이딩 윈도우 파라미터 ====================
TILE_SIZE = 256                     # 타일 크기 (학습 시와 동일해야 함)
STRIDE = 64                         # 슬라이딩 stride (작을수록 더 정확, 느림)

# ==================== 이진화 파라미터 ====================
THRESHOLD = 0.5                     # Pillar 확률 임계값 (논문 기준)

# 출력 디렉토리 생성
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

print("✅ 파라미터 설정 완료!")
print(f"\n📊 Inverse Design 설정:")
print(f"   입력 Phase Map: {INPUT_PHASE_PATH}")
print(f"   실험 이름: {EXPERIMENT_NAME}")
print(f"   체크포인트: {CHECKPOINT_PATH}")
print(f"   타일 크기: {TILE_SIZE}×{TILE_SIZE}")
print(f"   Stride: {STRIDE}")
print(f"   이진화 임계값: {THRESHOLD}")
print(f"   Device: {device}")

# %% [markdown]
# ## 3. 입력 Phase Map 로드

# %%
print("\n📂 입력 Phase Map 로딩 중...")

input_phase = np.load(INPUT_PHASE_PATH)

if input_phase is None:
    raise ValueError(f"Failed to load intensity map: {INPUT_PHASE_PATH}")

h, w = input_phase.shape

print(f"✅ 입력 Phase Map 로드 완료!")
print(f"   크기: {w}×{h}")
print(f"   Phase 범위: [{input_phase.min():.2f}, {input_phase.max():.2f}]")

# 입력 Phase Map 시각화
plt.figure(figsize=(10, 10))
plt.imshow(input_phase, cmap='twilight')
plt.colorbar(label='Phase')
plt.title(f'Input Phase Map\n{w}×{h}\nRange: [{input_phase.min():.2f}, {input_phase.max():.2f}]')
plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. 모델 로드

# %%
print("\n📥 Inverse Design 모델 로딩 중...")

# 모델 생성 (학습 시와 동일한 파라미터 사용)
model = InverseUNet(
    in_channels=1,
    out_channels=[1],
    layer_num=LAYER_NUM,
    base_features=BASE_FEATURES,
    dropout_rate=DROPOUT_RATE,
    output_activations=['linear'],  # BCEWithLogitsLoss를 위해 linear 사용
    use_batchnorm=USE_BATCHNORM
).to(device)

# 체크포인트 로드
checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ 모델 로드 완료!")
print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")

# Val loss 출력 (있는 경우에만)
if 'val_loss' in checkpoint:
    print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
    
# 모델 파라미터 수
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total params: {total_params:,}")

# %% [markdown]
# ## 5. Phase Map 정규화

# %%
print("\n🔧 Phase Map 정규화 중...")

# Phase map을 [0, 1]로 정규화
phase_normalized = (input_phase - input_phase.min()) / (input_phase.max() - input_phase.min() + 1e-8)

print(f"✅ 정규화 완료!")
print(f"   정규화 범위: [{phase_normalized.min():.4f}, {phase_normalized.max():.4f}]")

# %% [markdown]
# ## 6. 슬라이딩 윈도우 예측

# %%
print("\n🔮 슬라이딩 윈도우 예측 시작...")

# 예측 결과를 누적할 배열
prediction_sum = np.zeros((h, w), dtype=np.float32)
count_map = np.zeros((h, w), dtype=np.int32)

# 슬라이딩 윈도우로 타일 추출 및 예측
tiles_processed = 0
total_tiles = ((h - TILE_SIZE) // STRIDE + 1) * ((w - TILE_SIZE) // STRIDE + 1)

print(f"   총 예측 타일 수: {total_tiles:,}")
print(f"   (이미지 크기 {h}×{w}, 타일 {TILE_SIZE}×{TILE_SIZE}, stride {STRIDE})")

with torch.no_grad():
    # Y 방향 슬라이딩
    for top in tqdm(range(0, h - TILE_SIZE + 1, STRIDE), desc="Y position"):
        # X 방향 슬라이딩
        for left in range(0, w - TILE_SIZE + 1, STRIDE):
            # 타일 추출
            tile = phase_normalized[top:top+TILE_SIZE, left:left+TILE_SIZE]
            
            # Tensor로 변환 [1, 1, H, W]
            tile_tensor = torch.from_numpy(tile).float().unsqueeze(0).unsqueeze(0).to(device)
            
            # 예측 (logits)
            output = model(tile_tensor)
            
            # Sigmoid로 확률 변환
            prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # 누적
            prediction_sum[top:top+TILE_SIZE, left:left+TILE_SIZE] += prob
            count_map[top:top+TILE_SIZE, left:left+TILE_SIZE] += 1
            
            tiles_processed += 1

print(f"\n✅ 예측 완료! ({tiles_processed:,} 타일 처리)")

# %% [markdown]
# ## 7. Overlap Averaging

# %%
print("\n📊 Overlap Averaging 중...")

# 평균 계산
average_prob_map = prediction_sum / np.maximum(count_map, 1)

print(f"✅ Averaging 완료!")
print(f"   확률 범위: [{average_prob_map.min():.4f}, {average_prob_map.max():.4f}]")
print(f"   평균 확률: {average_prob_map.mean():.4f}")

# Count map 확인
unique_counts = np.unique(count_map)
print(f"   Overlap 횟수: {unique_counts.min()} ~ {unique_counts.max()}")

# %% [markdown]
# ## 8. 이진화 (Binarization)

# %%
print(f"\n🔨 이진화 중 (threshold={THRESHOLD})...")

# 0.5 임계값으로 이진화 (논문 방식)
binary_pillar_pattern = (average_prob_map > THRESHOLD).astype(np.uint8) * 255

pillar_ratio = np.sum(binary_pillar_pattern > 0) / binary_pillar_pattern.size
print(f"✅ 이진화 완료!")
print(f"   Pillar 비율: {pillar_ratio * 100:.2f}%")

# %% [markdown]
# ## 9. 결과 저장

# %%
print("\n💾 결과 저장 중...")

# 파일 이름 생성
input_name = Path(INPUT_PHASE_PATH).stem
output_prefix = f"{input_name}_inverse"

# 1. 확률 맵 저장 (.npy)
np.save(Path(OUTPUT_DIR) / f"{output_prefix}_prob_map.npy", average_prob_map)
print(f"   ✅ 확률 맵: {output_prefix}_prob_map.npy")

# 2. 이진화된 pillar pattern 저장 (.png)
cv2.imwrite(str(Path(OUTPUT_DIR) / f"{output_prefix}_pillar_pattern.png"), binary_pillar_pattern)
print(f"   ✅ Pillar Pattern: {output_prefix}_pillar_pattern.png")

# 3. Count map 저장 (.npy)
np.save(Path(OUTPUT_DIR) / f"{output_prefix}_count_map.npy", count_map)
print(f"   ✅ Count Map: {output_prefix}_count_map.npy")

print(f"\n✅ 모든 결과 저장 완료!")

# %% [markdown]
# ## 10. 결과 시각화

# %%
print("\n" + "="*80)
print("📊 결과 시각화")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Input Phase Map
axes[0, 0].imshow(input_phase, cmap='twilight')
axes[0, 0].set_title(f'Input: Phase Map\nRange: [{input_phase.min():.2f}, {input_phase.max():.2f}]', fontsize=12)
axes[0, 0].axis('off')

# 2. Normalized Phase Map
axes[0, 1].imshow(phase_normalized, cmap='twilight')
axes[0, 1].set_title(f'Normalized Phase Map\nRange: [{phase_normalized.min():.4f}, {phase_normalized.max():.4f}]', fontsize=12)
axes[0, 1].axis('off')

# 3. Count Map
im3 = axes[0, 2].imshow(count_map, cmap='viridis')
axes[0, 2].set_title(f'Count Map (Overlap)\nRange: {unique_counts.min()} ~ {unique_counts.max()}', fontsize=12)
axes[0, 2].axis('off')
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

# 4. Average Probability Map
im4 = axes[1, 0].imshow(average_prob_map, cmap='gray', vmin=0, vmax=1)
axes[1, 0].set_title(f'Probability Map\nMean: {average_prob_map.mean():.4f}', fontsize=12)
axes[1, 0].axis('off')
plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

# 5. Binary Pillar Pattern
axes[1, 1].imshow(binary_pillar_pattern, cmap='gray')
axes[1, 1].set_title(f'Binary Pillar Pattern (>{THRESHOLD})\nPillar: {pillar_ratio*100:.2f}%', fontsize=12)
axes[1, 1].axis('off')

# 6. Histogram
axes[1, 2].hist(average_prob_map.flatten(), bins=100, color='blue', alpha=0.7)
axes[1, 2].axvline(THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Threshold={THRESHOLD}')
axes[1, 2].set_xlabel('Probability', fontsize=11)
axes[1, 2].set_ylabel('Frequency', fontsize=11)
axes[1, 2].set_title('Probability Distribution', fontsize=12)
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / f"{output_prefix}_visualization.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ 시각화 저장: {output_prefix}_visualization.png")

# %% [markdown]
# ## 11. 상세 비교 (줌인)

# %%
print("\n📊 상세 영역 비교 (중앙 512×512)")

# 중앙 영역 추출
center_y, center_x = h // 2, w // 2
crop_size = 512
y1, y2 = center_y - crop_size//2, center_y + crop_size//2
x1, x2 = center_x - crop_size//2, center_x + crop_size//2

input_crop = input_phase[y1:y2, x1:x2]
prob_crop = average_prob_map[y1:y2, x1:x2]
binary_crop = binary_pillar_pattern[y1:y2, x1:x2]

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].imshow(input_crop, cmap='twilight')
axes[0].set_title(f'Input Phase Map\n(Center 512×512)', fontsize=14)
axes[0].axis('off')

axes[1].imshow(prob_crop, cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f'Probability Map\n(Center 512×512)', fontsize=14)
axes[1].axis('off')

axes[2].imshow(binary_crop, cmap='gray')
axes[2].set_title(f'Binary Pillar Pattern\n(Center 512×512)', fontsize=14)
axes[2].axis('off')

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / f"{output_prefix}_zoom.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ 줌인 이미지 저장: {output_prefix}_zoom.png")

# %% [markdown]
# ## 12. 통계 정보

# %%
print("\n" + "="*80)
print("📊 Inverse Design 결과 통계")
print("="*80)

print(f"\n【Input Phase Map】")
print(f"  크기: {input_phase.shape}")
print(f"  범위: [{input_phase.min():.6f}, {input_phase.max():.6f}]")
print(f"  평균: {input_phase.mean():.6f}")
print(f"  표준편차: {input_phase.std():.6f}")

print(f"\n【Prediction Probability Map】")
print(f"  크기: {average_prob_map.shape}")
print(f"  범위: [{average_prob_map.min():.6f}, {average_prob_map.max():.6f}]")
print(f"  평균: {average_prob_map.mean():.6f}")
print(f"  표준편차: {average_prob_map.std():.6f}")

print(f"\n【Binary Pillar Pattern】")
print(f"  크기: {binary_pillar_pattern.shape}")
print(f"  Pillar 픽셀 수: {np.sum(binary_pillar_pattern > 0):,}")
print(f"  Pillar 비율: {pillar_ratio * 100:.2f}%")
print(f"  이진화 임계값: {THRESHOLD}")

print(f"\n【Sliding Window】")
print(f"  타일 크기: {TILE_SIZE}×{TILE_SIZE}")
print(f"  Stride: {STRIDE}")
print(f"  처리된 타일 수: {tiles_processed:,}")
print(f"  Overlap 범위: {unique_counts.min()} ~ {unique_counts.max()}")

# %% [markdown]
# ## 13. 완료!
#
# Inverse Design이 완료되었습니다! 🎉
#
# **생성된 파일:**
# - `*_prob_map.npy`: 평균 확률 맵
# - `*_pillar_pattern.png`: 이진화된 pillar pattern (0.5 threshold)
# - `*_count_map.npy`: Overlap count map
# - `*_visualization.png`: 전체 결과 시각화
# - `*_zoom.png`: 중앙 영역 확대
#
# **다음 단계:**
# - 설계된 pillar pattern을 MEEP으로 시뮬레이션하여 검증
# - 다양한 intensity map으로 추가 설계 테스트

# %%
print("\n" + "="*80)
print("🎉 Inverse Design 완료!")
print("="*80)
print(f"\n📂 생성된 파일:")
print(f"   {Path(OUTPUT_DIR) / f'{output_prefix}_prob_map.npy'}")
print(f"   {Path(OUTPUT_DIR) / f'{output_prefix}_pillar_pattern.png'}")
print(f"   {Path(OUTPUT_DIR) / f'{output_prefix}_count_map.npy'}")
print(f"   {Path(OUTPUT_DIR) / f'{output_prefix}_visualization.png'}")
print(f"   {Path(OUTPUT_DIR) / f'{output_prefix}_zoom.png'}")
print(f"\n💡 설계된 pillar pattern을 MEEP으로 검증해보세요!")

