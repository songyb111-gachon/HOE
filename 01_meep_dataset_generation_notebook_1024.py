# %% [markdown]
# # 🔬 MEEP EM Near-Field Intensity Map Dataset Generation - 1024×1024 Version
#
# 메모리 효율적인 전략: 작은 샘플(5μm×5μm, 1024×1024 px) 40개 생성
#
# **Output**: EM Near-Field Intensity Map (|Ex|² + |Ey|² + |Ez|²)
#
# ## 📋 목차
# 1. 환경 설정 및 임포트
# 2. 파라미터 설정
# 3. 단일 샘플 테스트
# 4. 데이터셋 생성 (40개 샘플)

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
NUM_SAMPLES = 40              # 생성할 샘플 개수 (10개 → 40개로 증가)
                              # 전략: 작은 샘플을 많이 생성하여 메모리 절약
OUTPUT_DIR = 'data/forward_intensity_1024'  # 출력 디렉토리
SAVE_VISUALIZATIONS = True    # 시각화 저장 여부

# ==================== Random Pillar 파라미터 ====================
# 메모리 절약 전략: 5000×5000 nm (5 μm × 5 μm) × 40 샘플
# 총 면적: 25 μm² × 40 = 1000 μm² (원래와 동일)
# 목표: 샘플당 평균 738 ± 3개 기둥 (29.5 /μm² × 25 μm²)
PILLAR_PARAMS = {
    'domain_size': (5000, 5000),        # 시뮬레이션 영역 (nm) - 5 μm × 5 μm
                                        # 5000×5000 nm = 25 μm² (원래의 1/4)
                                        # 시뮬레이션 그리드: 1024×1024 pixels
                                        # Resolution: 5000/1024 ≈ 4.88 nm/pixel
    'pillar_radius': 45.0,              # 기둥 반지름 (nm) - 유지
    'min_edge_distance': 5.0,           # 최소 edge-to-edge 거리 (nm)
    'initial_density': 29.5,            # 초기 밀도 (pillars/μm²)
                                        # 예상 기둥 개수: 25 μm² × 29.5 = 738개/샘플
                                        # 총 pillars: 738 × 40 = 29,520개 (원래와 동일)
    'max_attempts': 10000
}

# ==================== MEEP 시뮬레이션 파라미터 ====================
# 논문 방식: 평면파 광원을 pillar 근처에 배치하여 X축 최소화 ⚡⚡⚡
SIMULATION_PARAMS = {
    'resolution_nm': 1024.0 / 5000.0,  # 해상도 (pixels/nm) ≈ 0.2048
                                        # 5000 nm → 1024 pixels
                                        # 픽셀 크기: ~4.88 nm/pixel
                                        # 메모리: ~4 GB/샘플 (17 GB에서 75% 감소)
    'pml_nm': 500.0,                    # PML 두께 (nm) - 파장(535nm)과 비슷하면 충분!
    'size_x_nm': 2000.0,                # x 방향 크기 (nm) - 최소화! ⚡⚡⚡
                                        # 2000nm = 2μm
                                        # Pillar(600) + 여유(400×2) + PML(500×2) = 1900nm
    'pillar_height_nm': 600.0,          # 기둥 높이 (nm) - pillar 두께
    'pillar_x_center': 0.0,             # 기둥 x 중심 (nm)
    'incident_deg': 0.0,                # 입사각 (도) - 수직 입사
    'wavelength_nm': 535.0,             # 파장 (nm) - 535nm 녹색
    'n_base': 1.5,                      # 기본 굴절률
    'delta_n': 0.04,                    # 굴절률 변조
    'cell_size_scale': 1.0,
    'auto_terminate': True,
    'decay_threshold': 1e-6,            # 논문과 동일: 1e-6
    'source_width_factor': 10
}

print("✅ 파라미터 설정 완료!")
print(f"\n📊 데이터셋 정보 (1024×1024 버전):")
print(f"   샘플 개수: {NUM_SAMPLES}")
print(f"   출력 디렉토리: {OUTPUT_DIR}")
print(f"   도메인 크기: {PILLAR_PARAMS['domain_size'][0]}×{PILLAR_PARAMS['domain_size'][1]} nm")
print(f"   출력 해상도: 1024×1024 pixels")
print(f"   해상도: {SIMULATION_PARAMS['resolution_nm']:.4f} pixels/nm")
print(f"   파장: {SIMULATION_PARAMS['wavelength_nm']} nm")
print(f"   예상 pillar: ~{int(PILLAR_PARAMS['domain_size'][0] * PILLAR_PARAMS['domain_size'][1] / 1e6 * PILLAR_PARAMS['initial_density'])}개/샘플")
print(f"\n💡 전략: 작은 샘플(1024×1024)을 많이(40개) 생성하여 메모리 절약")
print(f"   • 샘플당 메모리: ~4 GB (이전 대비 75% 감소)")
print(f"   • 총 데이터 면적: 1000 μm² (동일)")
print(f"   • 패턴 다양성: 4배 증가 (10개 → 40개)")

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
test_output_dir = Path('data/test_sample_1024')
test_output_dir.mkdir(parents=True, exist_ok=True)

print("🧪 테스트 샘플 생성 중 (1024×1024)...")
print("⏰ 예상 시간: 15분 ~ 1시간 (시스템에 따라 다름)")
print(f"   메모리 사용량: ~4 GB (이전 대비 75% 감소)\n")

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
    print(f"   Intensity 평균: {sample_info.get('intensity_mean', 0):.3e}")
    print(f"   Intensity 범위: [{sample_info.get('intensity_min', 0):.3e}, {sample_info.get('intensity_max', 0):.3e}]")
    
    # 시각화 표시
    if SAVE_VISUALIZATIONS:
        vis_path = test_output_dir / 'visualizations' / 'sample_0000_vis.png'
        if vis_path.exists():
            img = plt.imread(str(vis_path))
            plt.figure(figsize=(15, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title('Test Sample Visualization (1024×1024)')
            plt.tight_layout()
            plt.show()
else:
    print("\n❌ 테스트 샘플 생성 실패")

# %% [markdown]
# ## 5. 전체 데이터셋 생성
#
# ⚠️ **주의**: 이 셀은 오래 걸립니다 (3~10시간)
#
# 중간에 중단하고 싶으면 커널을 interrupt하세요.

# %%
print("="*80)
print("🚀 전체 데이터셋 생성 시작 (1024×1024 버전)")
print("="*80)
print(f"\n샘플 개수: {NUM_SAMPLES}")
print(f"출력 디렉토리: {OUTPUT_DIR}")
print(f"\n⏰ 예상 소요 시간: {NUM_SAMPLES * 0.25}~{NUM_SAMPLES * 1}시간")
print(f"   (샘플당 15분~1시간, 이전 대비 2~4배 빠름)")
print(f"\n💾 예상 메모리 사용량: ~4 GB/샘플 (이전 대비 75% 감소)")
print(f"   병렬 실행 가능: 여러 샘플 동시 생성 가능!")
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
print(f"   출력 Intensity 맵: {len(output_files)}개")

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
    intensity_map = np.load(output_npy_path)
    
    # 입력 마스크
    axes[idx, 0].imshow(input_mask, cmap='gray')
    axes[idx, 0].set_title(f'Sample {idx}: Input Mask (1024×1024)\n{input_mask.shape}')
    axes[idx, 0].axis('off')
    
    # Intensity 맵
    im = axes[idx, 1].imshow(intensity_map, cmap='hot')
    axes[idx, 1].set_title(f'Sample {idx}: EM Intensity Map\n{intensity_map.shape}')
    axes[idx, 1].axis('off')
    plt.colorbar(im, ax=axes[idx, 1], label='Intensity')
    
    # 히스토그램
    axes[idx, 2].hist(intensity_map.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[idx, 2].set_xlabel('Intensity')
    axes[idx, 2].set_ylabel('Count')
    axes[idx, 2].set_title(f'Sample {idx}: Intensity Distribution')
    axes[idx, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✅ {num_to_show}개 샘플 시각화 완료!")

# %% [markdown]
# ## 8. 다음 단계
#
# 데이터셋 생성이 완료되었습니다! 다음 노트북으로 이동하세요:
#
# 1. **`02_create_training_tiles_notebook_1024.py`**: 1024×1024 샘플에서 256×256 타일 추출
# 2. **`03_train_model_notebook_1024.py`**: U-Net 모델 학습
# 3. **`04_sliding_window_prediction_notebook_1024.py`**: 새로운 패턴에 대한 예측

# %%


