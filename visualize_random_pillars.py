"""
랜덤 필러 생성 시각화 도구

MEEP 시뮬레이션을 실행하기 전에 필러 패턴이 제대로 생성되는지 확인합니다.
여러 샘플을 생성하고 통계를 시각화하여 파라미터가 적절한지 검증합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from random_pillar_generator import RandomPillarGenerator
from datetime import datetime
import warnings
import cv2

# matplotlib 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def generate_multiple_samples(params, num_samples=10, random_seed=None):
    """
    여러 샘플을 생성하고 통계 수집
    
    Parameters:
    -----------
    params : dict
        pillar 생성 파라미터
    num_samples : int
        생성할 샘플 개수
    random_seed : int or None
        재현성을 위한 시드 (None이면 랜덤)
    
    Returns:
    --------
    samples : list of dict
        각 샘플의 pillars와 통계 정보
    """
    samples = []
    
    print("\n" + "="*80)
    print(f"랜덤 필러 샘플 {num_samples}개 생성 시작")
    print("="*80)
    
    for i in range(num_samples):
        print(f"\n[샘플 {i+1}/{num_samples}] 생성 중...")
        print("-"*80)
        
        # 랜덤 시드 설정 (각 샘플마다 다른 시드)
        if random_seed is not None:
            np.random.seed(random_seed + i)
        
        # 생성기 초기화
        generator = RandomPillarGenerator(
            pillar_radius=params['pillar_radius'],
            min_edge_distance=params['min_edge_distance'],
            domain_size=params['domain_size'],
            initial_density=params['initial_density'],
            max_attempts=params['max_attempts']
        )
        
        # 필러 생성
        pillars = generator.generate_pillars()
        
        # 통계 계산
        stats = generator.get_statistics()
        
        samples.append({
            'pillars': pillars,
            'generator': generator,
            'stats': stats
        })
        
        # 간단한 요약 출력
        print(f"✓ 샘플 {i+1} 완료: {len(pillars)}개 필러 생성")
    
    print("\n" + "="*80)
    print(f"전체 {num_samples}개 샘플 생성 완료!")
    print("="*80 + "\n")
    
    return samples


def plot_single_sample(sample, params, save_path='pillar_single.png', num_crops=6, crop_size=256, target_size=(2048, 2048)):
    """
    단일 샘플의 필러 패턴을 크게 시각화
    
    Parameters:
    -----------
    sample : dict
        단일 샘플 정보 (pillars, generator, stats)
    params : dict
        pillar 생성 파라미터
    save_path : str
        저장할 이미지 파일 경로
    num_crops : int
        랜덤 크롭할 개수
    crop_size : int
        크롭 크기 (정사각형)
    target_size : tuple
        최종 리사이즈할 크기 (width, height)
    """
    pillars = sample['pillars']
    stats = sample['stats']
    
    # Figure 생성 (모든 단계를 표시하기 위한 큰 레이아웃)
    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3)
    
    # 제목
    fig.suptitle(
        f'Random Pillar Generation Pipeline (All Steps Visualization)\n'
        f'Original Domain: {params["domain_size"][0]}×{params["domain_size"][1]} nm² → '
        f'Resized: {target_size[0]}×{target_size[1]} px → '
        f'Cropped: {crop_size}×{crop_size} px | '
        f'Pillar Count: {len(pillars)} | '
        f'Radius: {params["pillar_radius"]} nm | '
        f'Min Distance: {params["min_edge_distance"]} nm',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Step 1: 원본 마스크 생성 (도메인 크기) - 최적화된 버전
    width, height = params['domain_size']
    mask_original = np.zeros((height, width), dtype=np.uint8)
    
    radius = params['pillar_radius']
    radius_px = int(radius)
    
    print(f"\n원본 마스크 생성 중 ({len(pillars)} pillars)...")
    
    # 최적화: 각 pillar의 bounding box만 계산
    for idx, (cx, cy) in enumerate(pillars):
        cx_px = int(cx)
        cy_px = int(cy)
        
        # Bounding box 계산 (pillar 주변만)
        x_min = max(0, cx_px - radius_px)
        x_max = min(width, cx_px + radius_px + 1)
        y_min = max(0, cy_px - radius_px)
        y_max = min(height, cy_px + radius_px + 1)
        
        # 작은 영역에서만 거리 계산
        y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
        distances = np.sqrt((x_indices - cx_px)**2 + (y_indices - cy_px)**2)
        
        # 해당 영역에만 마스크 적용
        mask_original[y_min:y_max, x_min:x_max][distances <= radius_px] = 1
        
        # 진행상황 표시 (100개마다)
        if (idx + 1) % 500 == 0 or (idx + 1) == len(pillars):
            print(f"  {idx + 1}/{len(pillars)} pillars processed...")
    
    print(f"✓ 원본 마스크 생성 완료 ({width}×{height})")
    
    # 원본 마스크 저장 (10000×10000)
    original_save_path = save_path.replace('.png', '_original_10000x10000.png')
    cv2.imwrite(original_save_path, mask_original * 255)
    print(f"  → 원본 저장: {original_save_path}")
    
    # Step 2: 2048×2048로 리사이즈 (MEEP 시뮬레이션과 동일)
    print(f"\n리사이즈 중: {width}×{height} → {target_size[0]}×{target_size[1]}...")
    mask_resized = cv2.resize(mask_original, target_size, interpolation=cv2.INTER_NEAREST)
    print("✓ 리사이즈 완료")
    
    # 리사이즈된 마스크 저장 (2048×2048)
    resized_save_path = save_path.replace('.png', '_resized_2048x2048.png')
    cv2.imwrite(resized_save_path, mask_resized * 255)
    print(f"  → 리사이즈 저장: {resized_save_path}")
    
    # ========================================
    # 첫 번째 행: 각 단계별 출력 시각화
    # ========================================
    
    # 1-1. STEP 1: Original Domain (10000×10000 nm)
    ax1 = fig.add_subplot(gs[0, 0:2])
    im1 = ax1.imshow(mask_original, cmap='viridis', interpolation='nearest')
    ax1.set_title(
        f'STEP 1: Original Domain\n'
        f'{params["domain_size"][0]}×{params["domain_size"][1]} nm² | {len(pillars)} pillars',
        fontsize=12, fontweight='bold', pad=10, 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3)
    )
    ax1.set_xlabel('X (nm)', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Y (nm)', fontsize=9, fontweight='bold')
    ax1.grid(False)
    
    # 원본 통계
    orig_pillar_pixels = np.sum(mask_original)
    orig_total_pixels = mask_original.size
    orig_fill_ratio = (orig_pillar_pixels / orig_total_pixels) * 100
    
    orig_stats_text = f'Pillar pixels: {orig_pillar_pixels:,}\nFill ratio: {orig_fill_ratio:.2f}%'
    ax1.text(0.02, 0.98, orig_stats_text,
             transform=ax1.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 1-2. STEP 2: Resized (2048×2048 pixels)
    ax2 = fig.add_subplot(gs[0, 2:4])
    im2 = ax2.imshow(mask_resized, cmap='viridis', interpolation='nearest')
    ax2.set_title(
        f'STEP 2: Resized (cv2.resize)\n'
        f'{target_size[0]}×{target_size[1]} pixels | MEEP simulation output size',
        fontsize=12, fontweight='bold', pad=10,
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3)
    )
    ax2.set_xlabel('X (pixel)', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Y (pixel)', fontsize=9, fontweight='bold')
    ax2.grid(False)
    
    # 리사이즈된 통계
    resized_pillar_pixels = np.sum(mask_resized)
    resized_total_pixels = mask_resized.size
    resized_fill_ratio = (resized_pillar_pixels / resized_total_pixels) * 100
    
    resized_stats_text = f'Pillar pixels: {resized_pillar_pixels:,}\nFill ratio: {resized_fill_ratio:.2f}%'
    ax2.text(0.02, 0.98, resized_stats_text,
             transform=ax2.transAxes,
             fontsize=9,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 이후 작업은 리사이즈된 이미지 사용
    mask = mask_resized
    width, height = target_size
    
    # 1-3. Statistics Table
    ax3 = fig.add_subplot(gs[0, 4])
    ax3.axis('off')
    
    target_count = 2951
    target_tolerance = 14
    is_target_met = abs(len(pillars) - target_count) <= target_tolerance
    
    status_symbol = "✓" if is_target_met else "✗"
    status_text = "Target Met" if is_target_met else "Not Met"
    status_color = "lightgreen" if is_target_met else "lightcoral"
    
    stats_text = f"""
╔═════════════════════════════════╗
║      Pillar Statistics          ║
╠═════════════════════════════════╣
║ Count:    {stats['기둥 개수']:6d}             ║
║ Density:  {stats['기둥 밀도 (/μm²)']:6.2f} /um^2      ║
║ Fill:     {stats['충진율 (%)']:6.2f} %            ║
║ Min dist: {stats['최소 edge-to-edge 거리 (nm)']:6.2f} nm         ║
╠═════════════════════════════════╣
║         Parameters              ║
╠═════════════════════════════════╣
║ Init density: {params['initial_density']:5.1f} /um^2  ║
║ Radius:       {params['pillar_radius']:6.1f} nm      ║
║ Min distance: {params['min_edge_distance']:6.1f} nm      ║
╠═════════════════════════════════╣
║      Target Evaluation          ║
╠═════════════════════════════════╣
║ Target: {target_count} ± {target_tolerance}             ║
║ Actual: {stats['기둥 개수']:6d}                ║
║ Status: {status_symbol} {status_text:15s}   ║
╚═════════════════════════════════╝
"""
    
    ax3.text(0.5, 0.5, stats_text,
             transform=ax3.transAxes,
             fontsize=8,
             verticalalignment='center',
             horizontalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
    
    # ========================================
    # 두 번째-네 번째 행: STEP 3 - 랜덤 크롭들
    # ========================================
    
    print(f"\nSTEP 3: 리사이즈된 이미지({width}×{height})에서 {crop_size}×{crop_size} 랜덤 크롭 생성 중...")
    
    # 크롭 위치 결정
    crop_positions = []
    for i in range(num_crops):
        # 랜덤 시작 위치 선택 (크롭이 이미지 범위 내에 있도록)
        max_x = width - crop_size
        max_y = height - crop_size
        
        if max_x <= 0 or max_y <= 0:
            print(f"경고: 리사이즈된 크기({width}×{height})가 크롭 크기({crop_size}×{crop_size})보다 작습니다.")
            break
        
        start_x = np.random.randint(0, max_x)
        start_y = np.random.randint(0, max_y)
        
        crop_positions.append((start_x, start_y))
    
    # 크롭 이미지 시각화
    for idx, (start_x, start_y) in enumerate(crop_positions):
        # 레이아웃: 1행 5개, 2행 5개, 3행 5개 (최대 15개 크롭 가능)
        if idx < 5:
            row = 1
            col = idx
        elif idx < 10:
            row = 2
            col = idx - 5
        else:
            row = 3
            col = idx - 10
        
        ax = fig.add_subplot(gs[row, col])
        
        # 크롭
        cropped = mask[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # 시각화
        ax.imshow(cropped, cmap='viridis', interpolation='nearest')
        
        # 크롭 영역 내 통계
        pillar_pixels = np.sum(cropped)
        total_pixels = cropped.size
        fill_ratio = (pillar_pixels / total_pixels) * 100
        
        # 제목 색상 (STEP 3 강조)
        title_color = 'lightyellow' if idx == 0 else None
        
        ax.set_title(
            f'{"STEP 3: " if idx == 0 else ""}Training Tile #{idx+1}\n'
            f'Position [{start_x}:{start_x+crop_size}, {start_y}:{start_y+crop_size}]\n'
            f'Pillars: {pillar_pixels}px ({fill_ratio:.1f}%)',
            fontsize=9, fontweight='bold', pad=8,
            bbox=dict(boxstyle='round', facecolor=title_color, alpha=0.3) if title_color else None
        )
        ax.set_xlabel('X', fontsize=7)
        ax.set_ylabel('Y', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(False)
        
        # 크롭 영역을 원본에 표시하기 위한 사각형
        if idx == 0:
            # 첫 번째 크롭 위치를 STEP 2 이미지에 표시
            from matplotlib.patches import Rectangle
            rect = Rectangle((start_x, start_y), crop_size, crop_size, 
                           linewidth=2, edgecolor='red', facecolor='none')
            ax2.add_patch(rect)
            ax2.text(start_x, start_y-10, f'Crop #1', 
                    color='red', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    print(f"✓ STEP 3 완료: {len(crop_positions)}개 학습 타일 크롭 생성")
    
    # 크롭된 이미지들 개별 저장 (256×256)
    print(f"\n크롭된 타일 개별 저장 중...")
    for idx, (start_x, start_y) in enumerate(crop_positions):
        cropped = mask[start_y:start_y+crop_size, start_x:start_x+crop_size]
        crop_save_path = save_path.replace('.png', f'_crop_{idx+1}_256x256.png')
        cv2.imwrite(crop_save_path, cropped * 255)
        if idx == 0:
            print(f"  → 크롭 타일들 저장 중... (예: {crop_save_path})")
    print(f"  → 총 {len(crop_positions)}개 크롭 타일 저장 완료")
    
    # 전체 파이프라인 시각화 저장 및 표시
    print(f"\n전체 파이프라인 시각화 저장 중...")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  → 전체 시각화 저장: {save_path}")
    print("\n✓ 화면에 시각화 표시 중...")
    plt.show()
    plt.close()
    
    # 저장된 파일 요약
    print(f"\n" + "="*80)
    print("💾 저장된 파일 요약")
    print("="*80)
    print(f"1. 원본 마스크 (10000×10000):  {original_save_path}")
    print(f"2. 리사이즈 (2048×2048):       {resized_save_path}")
    print(f"3. 크롭 타일 (256×256):        {len(crop_positions)}개 파일")
    print(f"4. 전체 시각화:                {save_path}")
    print("="*80)
    
    # 콘솔 요약 출력
    print("\n" + "="*80)
    print("필러 생성 결과")
    print("="*80)
    print(f"필러 개수:      {stats['기둥 개수']}")
    print(f"밀도 (/μm²):   {stats['기둥 밀도 (/μm²)']:.2f}")
    print(f"충진율 (%):    {stats['충진율 (%)']:.2f}")
    print(f"최소 거리 (nm): {stats['최소 edge-to-edge 거리 (nm)']:.2f}")
    print(f"평균 거리 (nm): {stats['평균 최근접 이웃 거리 (nm)']:.2f}")
    print("-"*80)
    print(f"목표:          {target_count} ± {target_tolerance} 개")
    print(f"상태:          {status_symbol} {status_text}")
    print("="*80 + "\n")


def plot_pillar_statistics(samples, params, save_path='pillar_statistics.png'):
    """
    여러 샘플의 통계를 시각화
    
    Parameters:
    -----------
    samples : list of dict
        generate_multiple_samples()의 출력
    params : dict
        pillar 생성 파라미터
    save_path : str
        저장할 이미지 파일 경로
    """
    num_samples = len(samples)
    
    # 통계 수집
    pillar_counts = [s['stats']['기둥 개수'] for s in samples]
    densities = [s['stats']['기둥 밀도 (/μm²)'] for s in samples]
    fill_ratios = [s['stats']['충진율 (%)'] for s in samples]
    min_distances = [s['stats']['최소 edge-to-edge 거리 (nm)'] for s in samples]
    avg_distances = [s['stats']['평균 최근접 이웃 거리 (nm)'] for s in samples]
    
    # 통계값 계산
    mean_count = np.mean(pillar_counts)
    std_count = np.std(pillar_counts)
    mean_density = np.mean(densities)
    std_density = np.std(densities)
    mean_fill = np.mean(fill_ratios)
    std_fill = np.std(fill_ratios)
    mean_min_dist = np.mean(min_distances)
    std_min_dist = np.std(min_distances)
    mean_avg_dist = np.mean(avg_distances)
    std_avg_dist = np.std(avg_distances)
    
    # Figure 생성
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # 제목
    fig.suptitle(
        f'Random Pillar Generation Statistical Analysis (Samples: {num_samples})\n'
        f'Domain: {params["domain_size"][0]}×{params["domain_size"][1]} nm² | '
        f'Pillar Radius: {params["pillar_radius"]} nm | '
        f'Min Distance: {params["min_edge_distance"]} nm',
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # ========================================
    # 첫 번째 행: 샘플 이미지 (4개)
    # ========================================
    sample_indices = [0, num_samples//3, 2*num_samples//3, num_samples-1] if num_samples >= 4 else list(range(num_samples))
    
    for idx, sample_idx in enumerate(sample_indices[:4]):
        if sample_idx < num_samples:
            ax = fig.add_subplot(gs[0, idx])
            
            # 마스크 생성 (최적화)
            width, height = params['domain_size']
            mask = np.zeros((height, width), dtype=np.uint8)
            
            pillars = samples[sample_idx]['pillars']
            radius = params['pillar_radius']
            radius_px = int(radius)
            
            for (cx, cy) in pillars:
                cx_px = int(cx)
                cy_px = int(cy)
                
                # Bounding box만 계산
                x_min = max(0, cx_px - radius_px)
                x_max = min(width, cx_px + radius_px + 1)
                y_min = max(0, cy_px - radius_px)
                y_max = min(height, cy_px + radius_px + 1)
                
                y_indices, x_indices = np.ogrid[y_min:y_max, x_min:x_max]
                distances = np.sqrt((x_indices - cx_px)**2 + (y_indices - cy_px)**2)
                mask[y_min:y_max, x_min:x_max][distances <= radius_px] = 1
            
            # 시각화
            ax.imshow(mask, cmap='viridis', interpolation='nearest')
            ax.set_title(
                f'Sample {sample_idx+1}\nPillars: {len(pillars)}',
                fontsize=11, fontweight='bold'
            )
            ax.set_xlabel('X (nm)', fontsize=9)
            ax.set_ylabel('Y (nm)', fontsize=9)
            ax.grid(False)
    
    # ========================================
    # 두 번째 행: 히스토그램 (4개)
    # ========================================
    
    # 2-1: Pillar count distribution
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(pillar_counts, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.1f}')
    ax1.axvline(mean_count - std_count, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ: {std_count:.1f}')
    ax1.axvline(mean_count + std_count, color='orange', linestyle=':', linewidth=1.5)
    ax1.set_xlabel('Pillar Count', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax1.set_title('Pillar Count Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2-2: Density distribution
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.hist(densities, bins=15, color='green', edgecolor='black', alpha=0.7)
    ax2.axvline(mean_density, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_density:.2f}')
    ax2.axvline(mean_density - std_density, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ: {std_density:.2f}')
    ax2.axvline(mean_density + std_density, color='orange', linestyle=':', linewidth=1.5)
    ax2.set_xlabel('Density (/um^2)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax2.set_title('Pillar Density Distribution', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 2-3: Fill ratio distribution
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.hist(fill_ratios, bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax3.axvline(mean_fill, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_fill:.2f}%')
    ax3.axvline(mean_fill - std_fill, color='orange', linestyle=':', linewidth=1.5, label=f'±1σ: {std_fill:.2f}%')
    ax3.axvline(mean_fill + std_fill, color='orange', linestyle=':', linewidth=1.5)
    ax3.set_xlabel('Fill Ratio (%)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax3.set_title('Fill Ratio Distribution', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # 2-4: Min distance distribution
    ax4 = fig.add_subplot(gs[1, 3])
    ax4.hist(min_distances, bins=15, color='purple', edgecolor='black', alpha=0.7)
    ax4.axvline(mean_min_dist, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_min_dist:.2f}')
    ax4.axvline(params['min_edge_distance'], color='darkgreen', linestyle='-', linewidth=2, label=f'Min allowed: {params["min_edge_distance"]}')
    ax4.set_xlabel('Min edge-to-edge distance (nm)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=10, fontweight='bold')
    ax4.set_title('Min Pillar Distance Distribution', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # ========================================
    # 세 번째 행: 요약 통계 및 샘플별 비교
    # ========================================
    
    # 3-1: 요약 통계 테이블
    ax5 = fig.add_subplot(gs[2, 0:2])
    ax5.axis('off')
    
    summary_text = f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    Summary Statistics (N = {num_samples})                           ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Metric                       Mean             Std Dev        Range    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Pillar Count             {mean_count:8.1f}         ±{std_count:6.1f}     [{min(pillar_counts)}, {max(pillar_counts)}]
║  Density (/um^2)          {mean_density:8.2f}         ±{std_density:6.2f}     [{min(densities):.2f}, {max(densities):.2f}]
║  Fill Ratio (%)           {mean_fill:8.2f}         ±{std_fill:6.2f}     [{min(fill_ratios):.2f}, {max(fill_ratios):.2f}]
║  Min Distance (nm)        {mean_min_dist:8.2f}         ±{std_min_dist:6.2f}     [{min(min_distances):.2f}, {max(min_distances):.2f}]
║  Avg Nearest Dist (nm)    {mean_avg_dist:8.2f}         ±{std_avg_dist:6.2f}     [{min(avg_distances):.2f}, {max(avg_distances):.2f}]
╠═══════════════════════════════════════════════════════════════════════╣
║  Parameter Settings                                                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Initial density:    {params['initial_density']:.1f} /um^2                                       ║
║  Min allowed dist:   {params['min_edge_distance']:.1f} nm                                          ║
║  Pillar radius:      {params['pillar_radius']:.1f} nm                                          ║
║  Domain size:        {params['domain_size'][0]} × {params['domain_size'][1]} nm^2                      ║
╚═══════════════════════════════════════════════════════════════════════╝
"""
    
    ax5.text(0.5, 0.5, summary_text, 
             transform=ax5.transAxes,
             fontsize=10,
             verticalalignment='center',
             horizontalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # 3-2: Per-sample pillar count comparison
    ax6 = fig.add_subplot(gs[2, 2])
    sample_nums = list(range(1, num_samples + 1))
    ax6.plot(sample_nums, pillar_counts, marker='o', linestyle='-', linewidth=2, markersize=6, color='steelblue')
    ax6.axhline(mean_count, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_count:.1f}')
    ax6.fill_between(sample_nums, 
                     mean_count - std_count, 
                     mean_count + std_count, 
                     alpha=0.2, color='orange', label=f'±1σ')
    ax6.set_xlabel('Sample Number', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Pillar Count', fontsize=10, fontweight='bold')
    ax6.set_title('Pillar Count per Sample', fontsize=11, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # 3-3: Per-sample density comparison
    ax7 = fig.add_subplot(gs[2, 3])
    ax7.plot(sample_nums, densities, marker='s', linestyle='-', linewidth=2, markersize=6, color='green')
    ax7.axhline(mean_density, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_density:.2f}')
    ax7.fill_between(sample_nums, 
                     mean_density - std_density, 
                     mean_density + std_density, 
                     alpha=0.2, color='orange', label=f'±1σ')
    ax7.set_xlabel('Sample Number', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Density (/um^2)', fontsize=10, fontweight='bold')
    ax7.set_title('Pillar Density per Sample', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    # 저장 및 표시
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ 통계 시각화 저장: {save_path}")
    print("✓ 화면에 시각화 표시 중...")
    plt.show()
    plt.close()
    
    # 콘솔 요약 출력
    print("\n" + "="*80)
    print("통계 요약")
    print("="*80)
    print(f"필러 개수:      {mean_count:.1f} ± {std_count:.1f}  (범위: {min(pillar_counts)} ~ {max(pillar_counts)})")
    print(f"밀도 (/μm²):   {mean_density:.2f} ± {std_density:.2f}  (범위: {min(densities):.2f} ~ {max(densities):.2f})")
    print(f"충진율 (%):    {mean_fill:.2f} ± {std_fill:.2f}  (범위: {min(fill_ratios):.2f} ~ {max(fill_ratios):.2f})")
    print(f"최소 거리 (nm): {mean_min_dist:.2f} ± {std_min_dist:.2f}  (범위: {min(min_distances):.2f} ~ {max(min_distances):.2f})")
    print("="*80)
    
    # 목표 달성 여부 확인
    target_count = 2951
    target_tolerance = 14
    
    print("\n" + "="*80)
    print("목표 달성 평가")
    print("="*80)
    print(f"목표 필러 개수:      {target_count} ± {target_tolerance}")
    print(f"실제 평균 필러 개수: {mean_count:.1f} ± {std_count:.1f}")
    
    if abs(mean_count - target_count) <= target_tolerance and std_count <= target_tolerance * 1.5:
        print("✓ 목표 달성! 파라미터가 적절합니다.")
    elif abs(mean_count - target_count) <= target_tolerance * 2:
        print("⚠ 거의 목표에 근접했습니다. 미세 조정이 필요할 수 있습니다.")
    else:
        if mean_count < target_count:
            print(f"✗ 필러 개수가 부족합니다. initial_density를 약 {params['initial_density'] * target_count / mean_count:.1f}로 증가시키세요.")
        else:
            print(f"✗ 필러 개수가 너무 많습니다. initial_density를 약 {params['initial_density'] * target_count / mean_count:.1f}로 감소시키세요.")
    
    print("="*80 + "\n")


def main():
    """메인 실행 함수"""
    
    # ========================================
    # 파라미터 설정 (01_meep_dataset_generation_notebook.py와 동일)
    # ========================================
    
    PARAMS = {
        'pillar_radius': 45.0,           # nm
        'min_edge_distance': 5.0,        # nm
        'domain_size': (10000, 10000),   # nm (2048x2048 pixel에 대응)
        'initial_density': 29.5,         # pillars per μm²
        'max_attempts': 10000
    }
    
    NUM_SAMPLES = 1  # 생성할 샘플 개수
    RANDOM_SEED = None  # 재현성을 위한 시드 (None이면 랜덤)
    
    # 리사이즈 및 크롭 설정
    TARGET_SIZE = (2048, 2048)  # 최종 저장될 이미지 크기 (MEEP 시뮬과 동일)
    NUM_CROPS = 10  # 랜덤 크롭할 개수 (최대 15개 가능, 5×3 레이아웃)
    CROP_SIZE = 256  # 크롭 크기 (256×256, 학습에 사용될 타일 크기)
    
    # 출력 파일명
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = f'pillar_visualization_{timestamp}.png'
    
    # ========================================
    
    print("\n" + "="*80)
    print("랜덤 필러 시각화 도구")
    print("="*80)
    print(f"파라미터 설정:")
    print(f"  - 필러 반지름:     {PARAMS['pillar_radius']} nm")
    print(f"  - 최소 간격:       {PARAMS['min_edge_distance']} nm")
    print(f"  - 도메인 크기:     {PARAMS['domain_size'][0]} × {PARAMS['domain_size'][1]} nm²")
    print(f"  - 초기 밀도:       {PARAMS['initial_density']} /μm²")
    print(f"  - 리사이즈 크기:   {TARGET_SIZE[0]} × {TARGET_SIZE[1]} pixels")
    print(f"  - 샘플 개수:       {NUM_SAMPLES}")
    print(f"  - 학습 타일 크롭:  {NUM_CROPS}개, {CROP_SIZE}×{CROP_SIZE} pixels")
    print(f"  - 랜덤 시드:       {RANDOM_SEED if RANDOM_SEED is not None else '랜덤'}")
    print(f"  - 출력 파일:       {OUTPUT_FILE}")
    print("="*80)
    
    # 여러 샘플 생성
    samples = generate_multiple_samples(
        params=PARAMS,
        num_samples=NUM_SAMPLES,
        random_seed=RANDOM_SEED
    )
    
    # 시각화 (샘플 개수에 따라 다른 방식 사용)
    if NUM_SAMPLES == 1:
        # 단일 샘플: 큰 이미지로 상세히 표시
        plot_single_sample(
            sample=samples[0],
            params=PARAMS,
            save_path=OUTPUT_FILE,
            num_crops=NUM_CROPS,
            crop_size=CROP_SIZE,
            target_size=TARGET_SIZE
        )
    else:
        # 여러 샘플: 통계 분석 및 비교
        plot_pillar_statistics(
            samples=samples,
            params=PARAMS,
            save_path=OUTPUT_FILE
        )
    
    print("\n" + "="*80)
    print("✓ 모든 작업 완료!")
    print(f"✓ 결과 이미지: {OUTPUT_FILE}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

