# %% [markdown]
# # 📦 Training Tiles Generation (Sliding Window)
#
# 대형 샘플 (2048×2048)에서 256×256 타일을 추출합니다.
#
# ## 📋 목차
# 1. 환경 설정 및 임포트
# 2. 파라미터 설정
# 3. 타일 생성
# 4. 생성된 타일 확인 및 시각화

# %% [markdown]
# ## 1. 환경 설정 및 임포트

# %%
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random
from tqdm import tqdm

# 시각화 설정
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

print("✅ 모든 라이브러리 임포트 완료!")

# %% [markdown]
# ## 2. 파라미터 설정

# %%
# ==================== 타일 생성 파라미터 ====================
DATA_DIR = 'data/forward_intensity'          # 대형 샘플 디렉토리
OUTPUT_DIR = 'data/forward_intensity_tiles'  # 타일 출력 디렉토리
TILE_SIZE = 256                          # 타일 크기
NUM_TILES_PER_SAMPLE = 1000              # 샘플당 타일 개수
TRAIN_SAMPLES = 8                        # 훈련용 샘플 개수
VAL_SAMPLES = 2                          # 검증용 샘플 개수
RANDOM_SEED = 42                         # 랜덤 시드

# 랜덤 시드 설정
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("✅ 파라미터 설정 완료!")
print(f"\n📊 타일 생성 정보:")
print(f"   소스 디렉토리: {DATA_DIR}")
print(f"   출력 디렉토리: {OUTPUT_DIR}")
print(f"   타일 크기: {TILE_SIZE}×{TILE_SIZE}")
print(f"   샘플당 타일 개수: {NUM_TILES_PER_SAMPLE}")
print(f"   훈련 샘플: {TRAIN_SAMPLES} → {TRAIN_SAMPLES * NUM_TILES_PER_SAMPLE:,} 타일")
print(f"   검증 샘플: {VAL_SAMPLES} → {VAL_SAMPLES * NUM_TILES_PER_SAMPLE:,} 타일")
print(f"   총 타일: {(TRAIN_SAMPLES + VAL_SAMPLES) * NUM_TILES_PER_SAMPLE:,}")

# %% [markdown]
# ## 3. 타일 추출 함수 정의

# %%
def extract_random_tile(image, tile_size):
    """Extract a random tile from the image"""
    h, w = image.shape
    
    if h < tile_size or w < tile_size:
        raise ValueError(f"Image size {image.shape} is too small for tile size {tile_size}")
    
    max_y = h - tile_size
    max_x = w - tile_size
    
    top_y = random.randint(0, max_y)
    top_x = random.randint(0, max_x)
    
    tile = image[top_y:top_y+tile_size, top_x:top_x+tile_size]
    
    return tile, (top_y, top_x)

print("✅ 타일 추출 함수 정의 완료!")

# %% [markdown]
# ## 4. 데이터 확인

# %%
data_dir = Path(DATA_DIR)
input_dir = data_dir / 'inputs'
output_dir_src = data_dir / 'outputs'

# 사용 가능한 샘플 확인
all_samples = sorted(list(input_dir.glob('*.png')))

print(f"📂 사용 가능한 샘플: {len(all_samples)}개")

if len(all_samples) < TRAIN_SAMPLES + VAL_SAMPLES:
    print(f"\n⚠️  경고: 필요한 샘플 수({TRAIN_SAMPLES + VAL_SAMPLES})보다 적습니다!")
    print(f"   사용 가능: {len(all_samples)}개")
else:
    print(f"   → 충분한 샘플이 있습니다!")

# %% [markdown]
# ## 5. 타일 생성 - 훈련 세트

# %%
# 출력 디렉토리 생성
output_path = Path(OUTPUT_DIR)
for split in ['train', 'val']:
    (output_path / split / 'inputs').mkdir(parents=True, exist_ok=True)
    (output_path / split / 'outputs').mkdir(parents=True, exist_ok=True)

# 샘플 분할
random.shuffle(all_samples)
train_sample_files = all_samples[:TRAIN_SAMPLES]
val_sample_files = all_samples[TRAIN_SAMPLES:TRAIN_SAMPLES+VAL_SAMPLES]

print("="*80)
print("🔨 훈련 타일 생성 중...")
print("="*80)

tile_idx = 0
for sample_file in tqdm(train_sample_files, desc="Training samples"):
    # 입력/출력 로드
    input_path = data_dir / 'inputs' / sample_file.name
    output_npy_path = data_dir / 'outputs' / (sample_file.stem + '.npy')
    
    input_img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    output_intensity = np.load(output_npy_path)
    
    if input_img is None:
        print(f"  ⚠️  Failed to load {input_path}")
        continue
    
    # 크기가 다르면 조정
    if input_img.shape != output_intensity.shape:
        input_img = cv2.resize(input_img, (output_intensity.shape[1], output_intensity.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # 타일 추출
    for _ in range(NUM_TILES_PER_SAMPLE):
        try:
            # 입력 타일
            input_tile, (top_y, top_x) = extract_random_tile(input_img, TILE_SIZE)
            
            # 출력 타일
            output_tile = output_intensity[top_y:top_y+TILE_SIZE, top_x:top_x+TILE_SIZE]
            
            # 저장
            tile_name = f"tile_{tile_idx:06d}"
            cv2.imwrite(str(output_path / 'train' / 'inputs' / f"{tile_name}.png"), input_tile)
            np.save(str(output_path / 'train' / 'outputs' / f"{tile_name}.npy"), output_tile)
            
            tile_idx += 1
            
        except Exception as e:
            print(f"  ⚠️  Failed to extract tile: {e}")
            continue

print(f"\n✅ 훈련 타일 {tile_idx}개 생성 완료!")

# %% [markdown]
# ## 6. 타일 생성 - 검증 세트

# %%
print("\n" + "="*80)
print("🔨 검증 타일 생성 중...")
print("="*80)

tile_idx = 0
for sample_file in tqdm(val_sample_files, desc="Validation samples"):
    # 입력/출력 로드
    input_path = data_dir / 'inputs' / sample_file.name
    output_npy_path = data_dir / 'outputs' / (sample_file.stem + '.npy')
    
    input_img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    output_intensity = np.load(output_npy_path)
    
    if input_img is None:
        print(f"  ⚠️  Failed to load {input_path}")
        continue
    
    # 크기가 다르면 조정
    if input_img.shape != output_intensity.shape:
        input_img = cv2.resize(input_img, (output_intensity.shape[1], output_intensity.shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # 타일 추출
    for _ in range(NUM_TILES_PER_SAMPLE):
        try:
            # 입력 타일
            input_tile, (top_y, top_x) = extract_random_tile(input_img, TILE_SIZE)
            
            # 출력 타일
            output_tile = output_intensity[top_y:top_y+TILE_SIZE, top_x:top_x+TILE_SIZE]
            
            # 저장
            tile_name = f"tile_{tile_idx:06d}"
            cv2.imwrite(str(output_path / 'val' / 'inputs' / f"{tile_name}.png"), input_tile)
            np.save(str(output_path / 'val' / 'outputs' / f"{tile_name}.npy"), output_tile)
            
            tile_idx += 1
            
        except Exception as e:
            print(f"  ⚠️  Failed to extract tile: {e}")
            continue

print(f"\n✅ 검증 타일 {tile_idx}개 생성 완료!")

# %% [markdown]
# ## 7. 메타데이터 저장

# %%
metadata = {
    'tile_size': TILE_SIZE,
    'num_tiles_per_sample': NUM_TILES_PER_SAMPLE,
    'train_samples': TRAIN_SAMPLES,
    'val_samples': VAL_SAMPLES,
    'train_total_tiles': TRAIN_SAMPLES * NUM_TILES_PER_SAMPLE,
    'val_total_tiles': VAL_SAMPLES * NUM_TILES_PER_SAMPLE,
    'train_sample_files': [str(f.name) for f in train_sample_files],
    'val_sample_files': [str(f.name) for f in val_sample_files],
    'random_seed': RANDOM_SEED
}

metadata_path = output_path / 'tiles_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ 메타데이터 저장 완료!")

# %% [markdown]
# ## 8. 생성된 타일 확인

# %%
# 파일 개수 확인
train_input_tiles = list((output_path / 'train' / 'inputs').glob('*.png'))
train_output_tiles = list((output_path / 'train' / 'outputs').glob('*.npy'))
val_input_tiles = list((output_path / 'val' / 'inputs').glob('*.png'))
val_output_tiles = list((output_path / 'val' / 'outputs').glob('*.npy'))

print("\n" + "="*80)
print("🎉 타일 생성 완료!")
print("="*80)
print(f"\n📊 생성된 타일:")
print(f"   훈련 세트:")
print(f"     • 입력: {len(train_input_tiles):,}개")
print(f"     • 출력: {len(train_output_tiles):,}개")
print(f"   검증 세트:")
print(f"     • 입력: {len(val_input_tiles):,}개")
print(f"     • 출력: {len(val_output_tiles):,}개")
print(f"   총 타일: {len(train_input_tiles) + len(val_input_tiles):,}개")

# %% [markdown]
# ## 9. 타일 시각화

# %%
# 랜덤하게 6개 타일 시각화
num_to_show = 6
sample_indices = random.sample(range(len(train_input_tiles)), num_to_show)

fig, axes = plt.subplots(num_to_show, 3, figsize=(12, 4*num_to_show))

for idx, tile_idx in enumerate(sample_indices):
    # 타일 로드
    input_tile_path = train_input_tiles[tile_idx]
    output_tile_path = output_path / 'train' / 'outputs' / (input_tile_path.stem + '.npy')
    
    input_tile = cv2.imread(str(input_tile_path), cv2.IMREAD_GRAYSCALE)
    intensity_tile = np.load(output_tile_path)
    
    # 입력 타일
    axes[idx, 0].imshow(input_tile, cmap='gray')
    axes[idx, 0].set_title(f'Tile {tile_idx}: Input\n{input_tile.shape}')
    axes[idx, 0].axis('off')
    
    # Intensity 타일
    im = axes[idx, 1].imshow(intensity_tile, cmap='hot')
    axes[idx, 1].set_title(f'Tile {tile_idx}: EM Intensity\n{intensity_tile.shape}')
    axes[idx, 1].axis('off')
    plt.colorbar(im, ax=axes[idx, 1], fraction=0.046)
    
    # 히스토그램
    axes[idx, 2].hist(intensity_tile.flatten(), bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[idx, 2].set_xlabel('Intensity')
    axes[idx, 2].set_ylabel('Count')
    axes[idx, 2].set_title(f'Tile {tile_idx}: Distribution')
    axes[idx, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✅ {num_to_show}개 타일 시각화 완료!")

# %% [markdown]
# ## 10. 다음 단계
#
# 타일 생성이 완료되었습니다! 다음 노트북으로 이동하세요:
#
# **`03_train_model_notebook.py`**: U-Net 모델 학습

# %%

