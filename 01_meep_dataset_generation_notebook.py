# %% [markdown]
# # 🔬 MEEP Phase Map Dataset Generation
#
# 이 노트북은 MEEP 시뮬레이션을 실행하여 학습용 데이터셋을 생성합니다.
#
# ## 📋 목차
# 1. 환경 설정 및 임포트
# 2. 파라미터 설정
# 3. 단일 샘플 테스트
# 4. 데이터셋 생성 (10개 샘플)

# %% [markdown]
# ## 1. 환경 설정 및 임포트

# %%
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import os
import cv2
from pathlib import Path
from datetime import datetime
import json
from tqdm import tqdm
from IPython.display import display, Image as IPImage

# 시각화 설정
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("✅ 모든 라이브러리 임포트 완료!")
print(f"   MEEP 버전: {mp.__version__ if hasattr(mp, '__version__') else 'unknown'}")

# %% [markdown]
# ## 2. 파라미터 설정
#
# 모든 시뮬레이션 파라미터를 여기서 설정합니다.

# %%
# ==================== 데이터셋 생성 파라미터 ====================
NUM_SAMPLES = 10              # 생성할 샘플 개수
OUTPUT_DIR = 'data/forward_phase'  # 출력 디렉토리
SAVE_VISUALIZATIONS = True    # 시각화 저장 여부

# ==================== Random Pillar 파라미터 ====================
PILLAR_PARAMS = {
    'domain_size': (4096, 4096),        # 시뮬레이션 영역 (nm)
    'pillar_radius': 10.0,              # 기둥 반지름 (nm, 샘플마다 8-12로 랜덤)
    'min_edge_distance': 5.0,           # 최소 edge-to-edge 거리 (nm)
    'initial_density': 100.0,           # 초기 밀도 (pillars/μm², 샘플마다 80-120으로 랜덤)
    'max_attempts': 10000
}

# ==================== MEEP 시뮬레이션 파라미터 ====================
SIMULATION_PARAMS = {
    'resolution_nm': 1.0,               # 해상도 (pixels/nm) - 1:1 매칭
    'pml_nm': 1500.0,                   # PML 두께 (nm)
    'size_x_nm': 20000.0,               # x 방향 크기 (nm)
    'pillar_height_nm': 600.0,          # 기둥 높이 (nm)
    'pillar_x_center': 0.0,             # 기둥 x 중심 (nm)
    'incident_deg': 0.0,                # 입사각 (도)
    'wavelength_nm': 535.0,             # 파장 (nm)
    'n_base': 1.5,                      # 기본 굴절률
    'delta_n': 0.04,                    # 굴절률 변조
    'cell_size_scale': 1.0,
    'auto_terminate': True,
    'decay_threshold': 1e-4,
    'source_width_factor': 10
}

print("✅ 파라미터 설정 완료!")
print(f"\n📊 데이터셋 정보:")
print(f"   샘플 개수: {NUM_SAMPLES}")
print(f"   출력 디렉토리: {OUTPUT_DIR}")
print(f"   도메인 크기: {PILLAR_PARAMS['domain_size'][0]}×{PILLAR_PARAMS['domain_size'][1]} nm")
print(f"   해상도: {SIMULATION_PARAMS['resolution_nm']} pixels/nm")
print(f"   파장: {SIMULATION_PARAMS['wavelength_nm']} nm")

# %% [markdown]
# ## 3. 헬퍼 함수 로드

# %%
# 기존 모듈에서 함수 임포트
from meep_phase_simulation import (
    generate_single_training_sample,
    generate_training_dataset
)

from random_pillar_generator import RandomPillarGenerator

print("✅ 헬퍼 함수 로드 완료!")

# %% [markdown]
# ## 4. 단일 샘플 테스트 (선택사항)
#
# 전체 데이터셋 생성 전에 한 개의 샘플만 테스트합니다.

# %%
# 테스트용 출력 디렉토리
test_output_dir = Path('data/test_sample')
test_output_dir.mkdir(parents=True, exist_ok=True)

print("🧪 테스트 샘플 생성 중...")
print("⏰ 예상 시간: 30분 ~ 2시간 (시스템에 따라 다름)\n")

# 단일 샘플 생성
success, sample_info = generate_single_training_sample(
    sample_idx=0,
    output_dir=test_output_dir,
    pillar_params=PILLAR_PARAMS,
    simulation_params=SIMULATION_PARAMS,
    visualize=True
)

if success:
    print("\n✅ 테스트 샘플 생성 성공!")
    print(f"\n📊 샘플 정보:")
    print(f"   입력 크기: {sample_info['input_shape']}")
    print(f"   출력 크기: {sample_info['output_shape']}")
    print(f"   Fill ratio: {sample_info['fill_ratio']:.1f}%")
    print(f"   Pillar 개수: {sample_info['num_pillars']}")
    print(f"   Phase 평균: {sample_info['phase_mean']:.3f} rad")
    print(f"   Phase 범위: [{sample_info['phase_min']:.3f}, {sample_info['phase_max']:.3f}] rad")
    
    # 시각화 표시
    if SAVE_VISUALIZATIONS:
        vis_path = test_output_dir / 'visualizations' / 'sample_0000_vis.png'
        if vis_path.exists():
            img = plt.imread(str(vis_path))
            plt.figure(figsize=(15, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Test Sample Visualization')
            plt.tight_layout()
            plt.show()
else:
    print("\n❌ 테스트 샘플 생성 실패")

# %% [markdown]
# ## 5. 전체 데이터셋 생성
#
# ⚠️ **주의**: 이 셀은 오래 걸립니다 (5~20시간)
#
# 중간에 중단하고 싶으면 커널을 interrupt하세요.

# %%
print("="*80)
print("🚀 전체 데이터셋 생성 시작")
print("="*80)
print(f"\n샘플 개수: {NUM_SAMPLES}")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print(f"\n⏰ 예상 소요 시간: {NUM_SAMPLES * 0.5}~{NUM_SAMPLES * 2}시간")
print(f"\n진행 상황은 실시간으로 표시됩니다...\n")

# 데이터셋 생성
metadata = generate_training_dataset(
    num_samples=NUM_SAMPLES,
    output_dir=OUTPUT_DIR,
    pillar_params=PILLAR_PARAMS,
    simulation_params=SIMULATION_PARAMS,
    visualize_samples=SAVE_VISUALIZATIONS,
    start_idx=0
)

print("\n" + "="*80)
print("🎉 데이터셋 생성 완료!")
print("="*80)

# %% [markdown]
# ## 6. 생성된 데이터 확인

# %%
output_path = Path(OUTPUT_DIR)

# 파일 개수 확인
input_files = list((output_path / 'inputs').glob('*.png'))
output_files = list((output_path / 'outputs').glob('*.npy'))

print(f"📁 생성된 파일:")
print(f"   입력 마스크: {len(input_files)}개")
print(f"   출력 위상맵: {len(output_files)}개")

if SAVE_VISUALIZATIONS:
    vis_files = list((output_path / 'visualizations').glob('*.png'))
    print(f"   시각화: {len(vis_files)}개")

# 메타데이터 로드
metadata_path = output_path / 'dataset_metadata.json'
if metadata_path.exists():
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n📊 메타데이터:")
    print(f"   성공: {metadata['successful_samples']}")
    print(f"   실패: {metadata['failed_samples']}")
    print(f"   생성 시간: {metadata['generation_date']}")

# %% [markdown]
# ## 7. 샘플 시각화

# %%
# 처음 3개 샘플 시각화
num_to_show = min(3, len(input_files))

fig, axes = plt.subplots(num_to_show, 3, figsize=(15, 5*num_to_show))
if num_to_show == 1:
    axes = axes.reshape(1, -1)

for idx in range(num_to_show):
    # 입력 마스크 로드
    input_path = output_path / 'inputs' / f'sample_{idx:04d}.png'
    output_npy_path = output_path / 'outputs' / f'sample_{idx:04d}.npy'
    
    input_mask = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    phase_map = np.load(output_npy_path)
    
    # 입력 마스크
    axes[idx, 0].imshow(input_mask, cmap='gray')
    axes[idx, 0].set_title(f'Sample {idx}: Input Mask\n{input_mask.shape}')
    axes[idx, 0].axis('off')
    
    # 위상맵
    im = axes[idx, 1].imshow(phase_map, cmap='hsv', vmin=-np.pi, vmax=np.pi)
    axes[idx, 1].set_title(f'Sample {idx}: Phase Map\n{phase_map.shape}')
    axes[idx, 1].axis('off')
    plt.colorbar(im, ax=axes[idx, 1], label='Phase (rad)')
    
    # 히스토그램
    axes[idx, 2].hist(phase_map.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[idx, 2].set_xlabel('Phase (rad)')
    axes[idx, 2].set_ylabel('Count')
    axes[idx, 2].set_title(f'Sample {idx}: Phase Distribution')
    axes[idx, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✅ {num_to_show}개 샘플 시각화 완료!")

# %% [markdown]
# ## 8. 다음 단계
#
# 데이터셋 생성이 완료되었습니다! 다음 노트북으로 이동하세요:
#
# 1. **`02_create_training_tiles_notebook.py`**: 대형 샘플에서 256×256 타일 추출
# 2. **`03_train_model_notebook.py`**: U-Net 모델 학습
# 3. **`04_sliding_window_prediction_notebook.py`**: 새로운 패턴에 대한 예측

# %%

