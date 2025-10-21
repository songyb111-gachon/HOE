# %% [markdown]
# # 📦 Inverse Design Tiles Generation
#
# Forward 데이터를 역순으로 사용하여 Inverse Design용 타일을 생성합니다.
#
# **데이터 방향:**
# - Input: Phase Map (4096×4096 .npy) ← Forward의 outputs
# - Output: Pillar Pattern (4096×4096 .png) ← Forward의 inputs
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
DATA_DIR = 'data/forward_phase'          # Forward 데이터 디렉토리
OUTPUT_DIR = 'data/inverse_tiles'        # Inverse 타일 출력 디렉토리
TILE_SIZE = 256                          # 타일 크기
NUM_TILES_PER_SAMPLE = 1000              # 샘플당 타일 개수
TRAIN_SAMPLES = 8                        # 훈련용 샘플 개수
VAL_SAMPLES = 2                          # 검증용 샘플 개수
RANDOM_SEED = 42                         # 랜덤 시드

# 랜덤 시드 설정
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("✅ 파라미터 설정 완료!")
print(f"\n📊 Inverse Tiles 생성 정보:")
print(f"   소스 디렉토리: {DATA_DIR}")
print(f"   출력 디렉토리: {OUTPUT_DIR}")
print(f"   타일 크기: {TILE_SIZE}×{TILE_SIZE}")
print(f"   샘플당 타일 개수: {NUM_TILES_PER_SAMPLE}")
print(f"   훈련 샘플: {TRAIN_SAMPLES} → {TRAIN_SAMPLES * NUM_TILES_PER_SAMPLE:,} 타일")
print(f"   검증 샘플: {VAL_SAMPLES} → {VAL_SAMPLES * NUM_TILES_PER_SAMPLE:,} 타일")
print(f"   총 타일: {(TRAIN_SAMPLES + VAL_SAMPLES) * NUM_TILES_PER_SAMPLE:,}")
print("\n🔄 데이터 방향 (Inverse):")
print("   Input:  Phase Map (.npy) ← Forward의 outputs")
print("   Output: Pillar Pattern (.png) ← Forward의 inputs")

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
# Inverse는 Forward의 outputs를 input으로 사용
phase_dir = data_dir / 'outputs'  # Phase maps (.npy)
pillar_dir = data_dir / 'inputs'  # Pillar patterns (.png)

# 사용 가능한 샘플 확인 (phase map 기준)
all_samples = sorted(list(phase_dir.glob('*.npy')))

print(f"📂 사용 가능한 샘플: {len(all_samples)}개")

if len(all_samples) < TRAIN_SAMPLES + VAL_SAMPLES:
    print(f"\n⚠️  경고: 필요한 샘플 수({TRAIN_SAMPLES + VAL_SAMPLES})보다 적습니다!")
    print(f"   사용 가능: {len(all_samples)}개")
else:
    print(f"   ✅ 충분한 샘플이 있습니다!")

# 첫 샘플 확인
if len(all_samples) > 0:
    sample_phase = np.load(all_samples[0])
    sample_pillar_path = pillar_dir / (all_samples[0].stem + '.png')
    sample_pillar = cv2.imread(str(sample_pillar_path), cv2.IMREAD_GRAYSCALE)
    
    print(f"\n📊 샘플 크기:")
    print(f"   Phase Map: {sample_phase.shape} (range: {sample_phase.min():.2f} ~ {sample_phase.max():.2f})")
    if sample_pillar is not None:
        print(f"   Pillar Pattern: {sample_pillar.shape} (range: {sample_pillar.min()} ~ {sample_pillar.max()})")

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
print("🔨 Inverse 훈련 타일 생성 중...")
print("="*80)

tile_idx = 0
train_stats = {'phase_min': [], 'phase_max': [], 'pillar_min': [], 'pillar_max': []}

for sample_file in tqdm(train_sample_files, desc="Training samples"):
    # Inverse: phase map이 input, pillar pattern이 output
    input_phase_path = phase_dir / sample_file.name  # .npy
    output_pillar_path = pillar_dir / (sample_file.stem + '.png')  # .png
    
    # 로드
    input_phase = np.load(input_phase_path)  # Phase map (input)
    output_pillar = cv2.imread(str(output_pillar_path), cv2.IMREAD_GRAYSCALE)  # Pillar (output)
    
    if output_pillar is None:
        print(f"  ⚠️  Failed to load {output_pillar_path}")
        continue
    
    # 크기가 다르면 조정
    if input_phase.shape != output_pillar.shape:
        output_pillar = cv2.resize(output_pillar, (input_phase.shape[1], input_phase.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
    
    # 타일 추출
    for _ in range(NUM_TILES_PER_SAMPLE):
        try:
            # Phase map 타일 (input)
            input_tile, (top_y, top_x) = extract_random_tile(input_phase, TILE_SIZE)
            
            # Pillar pattern 타일 (output)
            output_tile = output_pillar[top_y:top_y+TILE_SIZE, top_x:top_x+TILE_SIZE]
            
            # 저장
            tile_name = f"tile_{tile_idx:06d}"
            np.save(str(output_path / 'train' / 'inputs' / f"{tile_name}.npy"), input_tile)
            cv2.imwrite(str(output_path / 'train' / 'outputs' / f"{tile_name}.png"), output_tile)
            
            # 통계
            train_stats['phase_min'].append(input_tile.min())
            train_stats['phase_max'].append(input_tile.max())
            train_stats['pillar_min'].append(output_tile.min())
            train_stats['pillar_max'].append(output_tile.max())
            
            tile_idx += 1
            
        except Exception as e:
            print(f"  ⚠️  Error extracting tile: {e}")
            continue

print(f"\n✅ 훈련 타일 생성 완료: {tile_idx:,}개")

# %% [markdown]
# ## 6. 타일 생성 - 검증 세트

# %%
print("="*80)
print("🔨 Inverse 검증 타일 생성 중...")
print("="*80)

val_tile_idx = 0
val_stats = {'phase_min': [], 'phase_max': [], 'pillar_min': [], 'pillar_max': []}

for sample_file in tqdm(val_sample_files, desc="Validation samples"):
    # Inverse: phase map이 input, pillar pattern이 output
    input_phase_path = phase_dir / sample_file.name
    output_pillar_path = pillar_dir / (sample_file.stem + '.png')
    
    # 로드
    input_phase = np.load(input_phase_path)
    output_pillar = cv2.imread(str(output_pillar_path), cv2.IMREAD_GRAYSCALE)
    
    if output_pillar is None:
        print(f"  ⚠️  Failed to load {output_pillar_path}")
        continue
    
    # 크기가 다르면 조정
    if input_phase.shape != output_pillar.shape:
        output_pillar = cv2.resize(output_pillar, (input_phase.shape[1], input_phase.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
    
    # 타일 추출
    for _ in range(NUM_TILES_PER_SAMPLE):
        try:
            # Phase map 타일 (input)
            input_tile, (top_y, top_x) = extract_random_tile(input_phase, TILE_SIZE)
            
            # Pillar pattern 타일 (output)
            output_tile = output_pillar[top_y:top_y+TILE_SIZE, top_x:top_x+TILE_SIZE]
            
            # 저장
            tile_name = f"tile_{val_tile_idx:06d}"
            np.save(str(output_path / 'val' / 'inputs' / f"{tile_name}.npy"), input_tile)
            cv2.imwrite(str(output_path / 'val' / 'outputs' / f"{tile_name}.png"), output_tile)
            
            # 통계
            val_stats['phase_min'].append(input_tile.min())
            val_stats['phase_max'].append(input_tile.max())
            val_stats['pillar_min'].append(output_tile.min())
            val_stats['pillar_max'].append(output_tile.max())
            
            val_tile_idx += 1
            
        except Exception as e:
            print(f"  ⚠️  Error extracting tile: {e}")
            continue

print(f"\n✅ 검증 타일 생성 완료: {val_tile_idx:,}개")

# %% [markdown]
# ## 7. 메타데이터 저장

# %%
metadata = {
    'data_source': DATA_DIR,
    'tile_size': TILE_SIZE,
    'num_tiles_per_sample': NUM_TILES_PER_SAMPLE,
    'train_samples': TRAIN_SAMPLES,
    'val_samples': VAL_SAMPLES,
    'total_train_tiles': tile_idx,
    'total_val_tiles': val_tile_idx,
    'random_seed': RANDOM_SEED,
    'data_direction': 'inverse',  # Phase map → Pillar pattern
    'input_type': 'phase_map',    # .npy
    'output_type': 'pillar_pattern',  # .png
    'train_stats': {
        'phase_min': float(np.min(train_stats['phase_min'])),
        'phase_max': float(np.max(train_stats['phase_max'])),
        'pillar_min': int(np.min(train_stats['pillar_min'])),
        'pillar_max': int(np.max(train_stats['pillar_max']))
    },
    'val_stats': {
        'phase_min': float(np.min(val_stats['phase_min'])),
        'phase_max': float(np.max(val_stats['phase_max'])),
        'pillar_min': int(np.min(val_stats['pillar_min'])),
        'pillar_max': int(np.max(val_stats['pillar_max']))
    }
}

metadata_path = output_path / 'inverse_tiles_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ 메타데이터 저장: {metadata_path}")
print(f"\n📊 최종 통계:")
print(f"   훈련 타일: {tile_idx:,}개")
print(f"   검증 타일: {val_tile_idx:,}개")
print(f"   총 타일: {tile_idx + val_tile_idx:,}개")
print(f"\n   Phase Map 범위 (train): {metadata['train_stats']['phase_min']:.2f} ~ {metadata['train_stats']['phase_max']:.2f}")
print(f"   Pillar Pattern 범위 (train): {metadata['train_stats']['pillar_min']} ~ {metadata['train_stats']['pillar_max']}")

# %% [markdown]
# ## 8. 생성된 타일 시각화

# %%
print("="*80)
print("📊 생성된 타일 시각화")
print("="*80)

# 훈련 타일 중 랜덤 샘플 시각화
num_samples = min(4, tile_idx)
sample_indices = random.sample(range(tile_idx), num_samples)

fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*5))
if num_samples == 1:
    axes = axes.reshape(1, -1)

for i, idx in enumerate(sample_indices):
    tile_name = f"tile_{idx:06d}"
    
    # Input: Phase map
    input_tile = np.load(output_path / 'train' / 'inputs' / f"{tile_name}.npy")
    
    # Output: Pillar pattern
    output_tile = cv2.imread(str(output_path / 'train' / 'outputs' / f"{tile_name}.png"), 
                            cv2.IMREAD_GRAYSCALE)
    
    # Plot
    axes[i, 0].imshow(input_tile, cmap='twilight')
    axes[i, 0].set_title(f'Input: Phase Map (Tile {idx})\nRange: [{input_tile.min():.2f}, {input_tile.max():.2f}]')
    axes[i, 0].axis('off')
    
    axes[i, 1].imshow(output_tile, cmap='gray')
    axes[i, 1].set_title(f'Output: Pillar Pattern (Tile {idx})\nRange: [{output_tile.min()}, {output_tile.max()}]')
    axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig(output_path / 'inverse_tiles_sample.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✅ 타일 시각화 완료!")
print(f"   저장 위치: {output_path / 'inverse_tiles_sample.png'}")

# %% [markdown]
# ## 9. 완료!
#
# Inverse Design용 타일 생성이 완료되었습니다! 🎉
#
# **다음 단계:**
# - `06_train_inverse_model_notebook.py`: Inverse 모델 학습
# - `07_inverse_design_notebook.py`: Inverse 예측 (원하는 phase → pillar 설계)

# %%
print("\n" + "="*80)
print("✅ Inverse Tiles 생성 완료!")
print("="*80)
print(f"\n📂 생성된 데이터:")
print(f"   {output_path / 'train' / 'inputs'}/     ← {tile_idx:,}개 phase map tiles (.npy)")
print(f"   {output_path / 'train' / 'outputs'}/    ← {tile_idx:,}개 pillar pattern tiles (.png)")
print(f"   {output_path / 'val' / 'inputs'}/       ← {val_tile_idx:,}개 phase map tiles (.npy)")
print(f"   {output_path / 'val' / 'outputs'}/      ← {val_tile_idx:,}개 pillar pattern tiles (.png)")
print(f"\n🚀 다음 단계: 06_train_inverse_model_notebook.py 실행!")

