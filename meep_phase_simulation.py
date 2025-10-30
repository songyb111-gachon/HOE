#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 Random Pillar Pattern + Plane Wave + EM Near-Field Intensity Map Simulation
================================================================================

3D MEEP simulation with random pillar pattern from random_pillar_generator.py
- Random pillar pattern (not periodic grating)
- 3D plane wave source (using amp_func)
- EM near-field intensity map calculation and analysis
- Total intensity: |Ex|² + |Ey|² + |Ez|²
- Based on HOE simulation structure
"""

# ============================================================================
# 로그 저장 코드
# ============================================================================
import sys
import os
from datetime import datetime

class AutoLogger:
    def __init__(self, filename="meep_simulation_log.txt"):
        if not os.path.exists("logs"):
            os.makedirs("logs")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name, ext = os.path.splitext(filename)
        self.log_path = os.path.join("logs", f"{name}_{timestamp}{ext}")
        self.terminal = sys.stdout
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        header = f"{'='*80}\nRandom Pillar EM Near-Field Intensity Map Simulation Log\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n\n"
        self.log_file.write(header)
        self.log_file.flush()
        print(f"Log file started: {self.log_path}")
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Start logging
_auto_logger = AutoLogger("random_pillar_intensity_simulation.txt")
sys.stdout = _auto_logger

import atexit
def _close_auto_log():
    if '_auto_logger' in globals():
        _auto_logger.log_file.write(f"\n{'='*80}\nLog End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n")
        _auto_logger.log_file.close()
atexit.register(_close_auto_log)
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import math
import cmath

# Font settings
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ================== Simulation Parameters ==================
# 모든 단위를 nm로 통일 (random_pillar_generator와 동일)

# Resolution and PML (HOE 물리 파라미터, nm 단위로 변환)
# ⚠️ Resolution = 1.0 pixels/nm으로 설정하여 마스크 픽셀과 1:1 매칭 (Binary pattern 유지)
RESOLUTION_NM = 1.0         # 해상도 (pixels/nm) - 1 픽셀 = 1 nm (마스크와 동일)
PML_NM = 1500.0            # PML 두께 (nm) = 1.5 μm

# Simulation cell size (nm)
SIZE_X_NM = 20000.0        # x 방향 (전파 방향, nm) = 20 μm
# SIZE_Y_NM, SIZE_Z_NM은 마스크 크기에서 자동 계산 (1 픽셀 = 1 nm)

# Random pillar structure parameters (nm)
PILLAR_HEIGHT_NM = 600.0   # 기둥(필름) 두께 (nm) = 0.6 μm
PILLAR_X_CENTER = 0.0      # 기둥 중심 x 위치 (nm) - 셀 중앙

# Optical parameters (nm)
WAVELENGTH_NM = 535.0      # 파장 (nm) - 535nm 녹색 레이저
INCIDENT_DEG = 0.0         # 입사각 (도) - 수직 입사

# Material properties (HOE 표준)
N_BASE = 1.5               # 기본 굴절률
DELTA_N = 0.04             # 굴절률 변조 (현실적)

# Auto-termination settings
AUTO_TERMINATE = True      # 자동 종료 활성화
DECAY_THRESHOLD = 1e-4     # 필드 감쇠 임계값 (상대값, 정상 상태 판단)
SOURCE_WIDTH_FACTOR = 10   # GaussianSource 폭 (파장 배수)

# Single parameter (nm 단위) - 단일 파장만 사용
PARAMETER_SWEEP = {
    'pillar_height_nm': [600.0],  # 기둥(필름) 두께 (nm)
    'wavelength_nm': [535.0],  # 단일 파장: 535nm 녹색 (RGB sweep 제거)
    'delta_n': [0.04],  # 굴절률 변조
    'incident_deg': [0.0]  # 입사각
}

# Input file
MASK_FILE = 'random_pillar_slice_mask.npy'  # Binary mask from random_pillar_generator.py

# Cell size scaling factor (optional, 1.0 = use mask size as-is)
CELL_SIZE_SCALE = 1.0      # 패턴 크기 스케일 조정 (필요시)

# ============================================================


def pw_amp(k, x0):
    """Plane wave phase function (from MEEP example)"""
    def _amp(x):
        return cmath.exp(1j * 2*math.pi * k.dot(x + x0))
    return _amp


def load_random_pillar_mask(mask_file):
    """Load random pillar binary mask
    
    Parameters:
    -----------
    mask_file : str
        Path to binary mask numpy file (.npy)
        
    Returns:
    --------
    mask : 2D numpy array
        Binary pattern (0 or 1)
    mask_info : dict
        Mask information
    """
    print(f"=== Loading Random Pillar Mask ===")
    print(f"Mask file: {mask_file}")
    
    # Load mask
    mask = np.load(mask_file)
    
    print(f"Mask size: {mask.shape} (height × width)")
    print(f"Data type: {mask.dtype}")
    
    # Analyze mask
    total_pixels = mask.size
    pillar_pixels = np.sum(mask == 1)
    background_pixels = total_pixels - pillar_pixels
    fill_ratio = pillar_pixels / total_pixels * 100
    
    mask_info = {
        'shape': mask.shape,
        'total_pixels': total_pixels,
        'pillar_pixels': pillar_pixels,
        'background_pixels': background_pixels,
        'fill_ratio': fill_ratio,
        'pattern_type': 'random_pillar'
    }
    
    print(f"Mask statistics:")
    print(f"  • Total pixels: {total_pixels:,}")
    print(f"  • Pillar pixels (1): {pillar_pixels:,}")
    print(f"  • Background pixels (0): {background_pixels:,}")
    print(f"  • Fill ratio: {fill_ratio:.1f}%")
    print(f"  • Pattern type: Random pillar (non-periodic)")
    
    return mask, mask_info


def resample_mask_to_cell_size(mask, target_ny, target_nz):
    """Resample mask to MEEP grid resolution
    
    Parameters:
    -----------
    mask : 2D numpy array
        Original binary mask from generator (height × width)
    target_ny : int
        Target grid points in y direction (horizontal)
    target_nz : int
        Target grid points in z direction (vertical)
        
    Returns:
    --------
    resampled_mask : 2D numpy array
        Resampled binary mask (nz × ny) format for MEEP
    
    Note:
    -----
    - Input mask: (height, width) from random_pillar_generator
    - Output mask: (nz, ny) for MEEP coordinates (z=height, y=width)
    - Uses nearest neighbor interpolation to preserve binary nature
    """
    from scipy import ndimage
    
    original_shape = mask.shape  # (height, width) from generator
    # zoom_factors: (height_scale, width_scale) = (nz/original_height, ny/original_width)
    zoom_factors = (target_nz / original_shape[0], target_ny / original_shape[1])
    
    print(f"\n  📐 Resampling mask to MEEP grid:")
    print(f"    • Original mask: {original_shape} pixels (height × width)")
    print(f"    • Target MEEP grid: ({target_nz} × {target_ny}) points (z × y)")
    print(f"    • Zoom factors: (z={zoom_factors[0]:.4f}, y={zoom_factors[1]:.4f})")
    
    # Use nearest neighbor (order=0) to preserve binary nature
    resampled = ndimage.zoom(mask, zoom_factors, order=0)
    
    # Ensure binary (0 or 1)
    resampled = (resampled > 0.5).astype(np.uint8)
    
    # Check fill ratio preservation
    fill_before = np.sum(mask) / mask.size * 100
    fill_after = np.sum(resampled) / resampled.size * 100
    
    print(f"    • Fill ratio: {fill_before:.1f}% → {fill_after:.1f}%")
    print(f"    • Resampled shape: {resampled.shape} (nz × ny)")
    
    return resampled


def create_random_pillar_geometry(mask, cell_size_x, cell_size_y, cell_size_z,
                                  n_base=1.5, delta_n=0.04, thickness_um=600.0, 
                                  pillar_x_center=0.0):
    """Convert random pillar mask to MEEP geometry (HOE-style, nm units)
    
    HOE 코드의 create_binary_grating_geometry 방식을 따름:
    - YZ 평면에 패턴 배치
    - 픽셀별로 Block 생성
    - 모든 단위는 nm
    
    Refractive index mapping (X 방향 전체 두께에 적용):
    - Pattern = 0 (background) → X 방향 전체(thickness) n = n_base (default: 1.5)
    - Pattern = 1 (pillar)     → X 방향 전체(thickness) n = n_base + delta_n (default: 1.54)
    
    Parameters:
    -----------
    mask : 2D numpy array
        Binary pattern (0 or 1), shape (nz, ny) = (z, y) in MEEP coordinates
        0 = background (n_base), 1 = pillar (n_base + delta_n)
    cell_size_x, cell_size_y, cell_size_z : float
        Cell size (nm)
    n_base : float
        Base refractive index for background (pattern = 0)
    delta_n : float
        Refractive index modulation (pattern = 1 → n = n_base + delta_n)
    thickness_um : float
        Pillar thickness in x direction (nm)
    pillar_x_center : float
        Pillar center position in x direction (nm)
        
    Returns:
    --------
    geometry : list
        MEEP geometry objects (Blocks for pattern = 1)
    background_material : mp.Medium
        Background material (n = n_base, for pattern = 0)
    """
    print(f"\n=== Generating Random Pillar Geometry (HOE-style, nm units) ===")
    print(f"Mask size: {mask.shape} (nz × ny)")
    print(f"\n📊 Refractive index mapping:")
    print(f"  • Pattern = 0 (background) → X 방향 {thickness_um:.0f} nm 전체가 n = {n_base:.2f}")
    print(f"  • Pattern = 1 (pillar)     → X 방향 {thickness_um:.0f} nm 전체가 n = {n_base + delta_n:.2f}")
    print(f"  • Δn = {delta_n:.3f}")
    print(f"\nPillar structure:")
    print(f"  • Film thickness (X direction): {thickness_um:.0f} nm (전체 두께)")
    print(f"  • Pillar x center: {pillar_x_center:.0f} nm")
    print(f"  • YZ 평면에서 패턴에 따라 X 방향 {thickness_um:.0f} nm 전체 굴절률 변조")
    
    # Materials (HOE standard)
    # Pattern 0 (background) = n_base
    # Pattern 1 (pillar) = n_base + delta_n
    background_material = mp.Medium(index=n_base)
    pillar_material = mp.Medium(index=n_base + delta_n)
    
    # Generate geometry
    geometry = []
    
    nz, ny = mask.shape  # (z, y) format
    z_coords = np.linspace(-cell_size_z/2, cell_size_z/2, nz)  # z coordinates
    y_coords = np.linspace(-cell_size_y/2, cell_size_y/2, ny)  # y coordinates
    
    print(f"Coordinates:")
    print(f"  • y range: {y_coords[0]:.0f} to {y_coords[-1]:.0f} nm ({ny} points)")
    print(f"  • z range: {z_coords[0]:.0f} to {z_coords[-1]:.0f} nm ({nz} points)")
    
    # Pixel sizes (nm)
    pixel_size_y = cell_size_y / ny
    pixel_size_z = cell_size_z / nz
    
    print(f"Pixel size: {pixel_size_y:.2f} × {pixel_size_z:.2f} nm (y × z)")
    
    # Create blocks (HOE method: analyze pattern and create blocks)
    # Pattern = 0 (background) → use background_material (default)
    # Pattern = 1 (pillar) → create Block with pillar_material
    # 
    # ⚠️ 중요: X 방향 thickness_um(600 nm) 전체가 굴절률 변조됨
    #   - Pattern = 0 위치: X 방향 600 nm 전체가 n = 1.5 (background)
    #   - Pattern = 1 위치: X 방향 600 nm 전체가 n = 1.54 (Block)
    pillar_count = 0
    for j in range(nz):  # z direction (vertical in pattern)
        z_pos = z_coords[j]
        
        # Check pattern at this z position
        for i in range(ny):  # y direction (horizontal in pattern)
            if mask[j, i] > 0.5:  # Pattern = 1 (pillar pixel)
                y_pos = y_coords[i]
                
                # Create block with pillar_material (n = n_base + delta_n)
                # X 방향 전체 thickness_um (600 nm)가 n = 1.54
                block = mp.Block(
                    size=mp.Vector3(thickness_um, pixel_size_y, pixel_size_z),  # X: 600 nm 전체
                    center=mp.Vector3(pillar_x_center, y_pos, z_pos),  # (x, y, z)
                    material=pillar_material  # n = 1.54
                )
                geometry.append(block)
                pillar_count += 1
            # else: Pattern = 0 (background), X 방향 600 nm 전체가 n = 1.5 (default_material)
    
    print(f"Generated Block count: {len(geometry):,}")
    print(f"  • Pillar pixels: {pillar_count:,}")
    print(f"  • Block size: {thickness_um:.0f} × {pixel_size_y:.2f} × {pixel_size_z:.2f} nm")
    
    # Debug: Check first few blocks
    if len(geometry) > 0:
        print(f"\n  🔍 Debug - First 3 blocks:")
        for idx, block in enumerate(geometry[:3]):
            print(f"    Block {idx+1}: center=({block.center.x:.1f}, {block.center.y:.1f}, {block.center.z:.1f}) nm, "
                  f"size=({block.size.x:.1f}, {block.size.y:.2f}, {block.size.z:.2f}) nm")
        print(f"    ...")
        print(f"    ⚠️ 중요: 모든 Block의 X center = {pillar_x_center:.1f}, X size = {thickness_um:.0f} nm")
        print(f"    ⚠️ 즉, 모든 Block은 X 방향 [{pillar_x_center-thickness_um/2:.1f}, {pillar_x_center+thickness_um/2:.1f}] nm 범위")
    
    return geometry, background_material


def visualize_actual_meep_pattern(sim, size_x_nm, size_y_nm, size_z_nm, 
                                  pillar_x_center, pillar_thickness_nm,
                                  wavelength_nm=535, delta_n=0.04, n_base=1.5, 
                                  resolution_nm=0.03, incident_deg=0):
    """Visualize actual refractive index distribution used in MEEP simulation (nm units)"""
    
    print(f"\n  📐 Extracting actual MEEP refractive index distribution...")
    
    # YZ plane (pillar pattern at x=pillar_x_center)
    yz_plane_center = mp.Vector3(pillar_x_center, 0, 0)
    yz_plane_size = mp.Vector3(0, size_y_nm, size_z_nm)
    
    eps_data_yz = sim.get_array(center=yz_plane_center, size=yz_plane_size, 
                                component=mp.Dielectric)
    n_data_yz = np.sqrt(np.real(eps_data_yz))
    print(f"    YZ plane: shape={n_data_yz.shape}, center=({pillar_x_center:.1f}, 0, 0), size=(0, {size_y_nm:.0f}, {size_z_nm:.0f})")
    
    # XZ plane (side view at y=0) - focus on pillar region only
    # Pillar region: pillar_x_center ± pillar_thickness/2
    xz_view_size_x = pillar_thickness_nm * 2.0  # Show 2x pillar thickness for context
    xz_plane_center = mp.Vector3(pillar_x_center, 0, 0)
    xz_plane_size = mp.Vector3(xz_view_size_x, 0, size_z_nm)
    
    eps_data_xz = sim.get_array(center=xz_plane_center, size=xz_plane_size, 
                                component=mp.Dielectric)
    n_data_xz = np.sqrt(np.real(eps_data_xz))
    print(f"    XZ plane: shape={n_data_xz.shape}, center=({pillar_x_center:.1f}, 0, 0), size=({xz_view_size_x:.0f}, 0, {size_z_nm:.0f})")
    print(f"      ⚠️ 예상: X 방향 각 위치에서 모든 Z가 같은 굴절률 (X 방향 600 nm 전체가 uniform)")
    
    # XY plane (top view at z=0) - focus on pillar region only
    xy_plane_center = mp.Vector3(pillar_x_center, 0, 0)
    xy_plane_size = mp.Vector3(xz_view_size_x, size_y_nm, 0)
    
    eps_data_xy = sim.get_array(center=xy_plane_center, size=xy_plane_size, 
                                component=mp.Dielectric)
    n_data_xy = np.sqrt(np.real(eps_data_xy))
    print(f"    XY plane: shape={n_data_xy.shape}, center=({pillar_x_center:.1f}, 0, 0), size=({xz_view_size_x:.0f}, {size_y_nm:.0f}, 0)")
    print(f"      ⚠️ 예상: X 방향 각 위치에서 모든 Y가 같은 굴절률 (X 방향 600 nm 전체가 uniform)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # YZ plane (actual pillar pattern)
    ax1 = axes[0, 0]
    extent_yz = [-size_z_nm*0.5, size_z_nm*0.5, -size_y_nm*0.5, size_y_nm*0.5]
    im1 = ax1.imshow(n_data_yz, extent=extent_yz, cmap='viridis', origin='lower')
    ax1.set_title(f'YZ Plane: Actual MEEP Refractive Index\\n(x = {pillar_x_center} nm, Random Pillar Pattern)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('z (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im1, ax=ax1, label='Refractive Index n')
    ax1.grid(True, alpha=0.3)
    
    # Statistics
    n_min, n_max = np.min(n_data_yz), np.max(n_data_yz)
    ax1.text(0.05, 0.95, f'n_min: {n_min:.3f}\\nn_max: {n_max:.3f}\\nΔn: {n_max-n_min:.3f}', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # XZ plane (side view) - focused on pillar region
    ax2 = axes[0, 1]
    extent_xz = [-xz_view_size_x/2, xz_view_size_x/2, -size_z_nm*0.5, size_z_nm*0.5]
    im2 = ax2.imshow(n_data_xz.T, extent=extent_xz, cmap='viridis', origin='lower')
    ax2.set_title(f'XZ Plane: Side View (y = 0 nm)\n[Focused on Pillar Region]', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('z (nm)')
    plt.colorbar(im2, ax=ax2, label='Refractive Index n')
    ax2.grid(True, alpha=0.3)
    
    # Mark pillar region
    pillar_x_min = -pillar_thickness_nm/2
    pillar_x_max = +pillar_thickness_nm/2
    ax2.axvline(x=pillar_x_min, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axvline(x=pillar_x_max, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.text(0, size_z_nm*0.4, f'Pillar\\n{pillar_thickness_nm:.0f} nm', ha='center', va='center',
             color='red', fontweight='bold', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # XY plane (top view) - focused on pillar region
    ax3 = axes[1, 0]
    extent_xy = [-xz_view_size_x/2, xz_view_size_x/2, -size_y_nm*0.5, size_y_nm*0.5]
    im3 = ax3.imshow(n_data_xy.T, extent=extent_xy, cmap='viridis', origin='lower')
    ax3.set_title(f'XY Plane: Top View (z = 0 nm)\n[Focused on Pillar Region]', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3, label='Refractive Index n')
    ax3.grid(True, alpha=0.3)
    
    # Mark pillar region
    ax3.axvline(x=-pillar_thickness_nm/2, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax3.axvline(x=+pillar_thickness_nm/2, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # YZ plane histogram
    ax4 = axes[1, 1]
    n_flat = n_data_yz.flatten()
    ax4.hist(n_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Refractive Index n')
    ax4.set_ylabel('Pixel Count')
    ax4.set_title('YZ Plane Refractive Index Distribution\n(Intermediate values from MEEP subpixel averaging)', 
                  fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Statistics
    n_mean = np.mean(n_flat)
    n_std = np.std(n_flat)
    ax4.axvline(x=n_mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {n_mean:.3f}')
    ax4.axvline(x=n_mean+n_std, color='orange', linestyle='--', linewidth=2, label=f'+1σ: {n_mean+n_std:.3f}')
    ax4.axvline(x=n_mean-n_std, color='orange', linestyle='--', linewidth=2, label=f'-1σ: {n_mean-n_std:.3f}')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save with parameters in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = (f"wl{wavelength_nm:.0f}nm_"
                f"h{pillar_thickness_nm:.0f}nm_"
                f"dn{delta_n:.3f}_"
                f"nb{n_base:.2f}_"
                f"res{resolution_nm:.3f}_"
                f"inc{incident_deg:.0f}deg_"
                f"size{int(size_y_nm)}x{int(size_z_nm)}nm")
    
    filename = f"meep_refractive_index_{param_str}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 MEEP Refractive Index Analysis:")
    print(f"    • Min: {np.min(n_data_yz):.4f}")
    print(f"    • Max: {np.max(n_data_yz):.4f}")
    print(f"    • Mean: {np.mean(n_data_yz):.4f} ± {np.std(n_data_yz):.4f}")
    print(f"    • Δn: {np.max(n_data_yz) - np.min(n_data_yz):.4f}")
    print(f"  💾 Saved: {filename}")


def calculate_intensity_map_from_monitors(back_monitors, size_y_nm, size_z_nm, wavelength_nm):
    """Calculate EM near-field intensity map from back monitors (nm units)
    
    Intensity: Total electric field intensity = |Ex|² + |Ey|² + |Ez|²
    
    Parameters:
    -----------
    back_monitors : dict
        Back monitor data with Ez, Ex, Ey fields
    size_y_nm, size_z_nm : float
        Monitor size (nm)
    wavelength_nm : float
        Wavelength (nm)
        
    Returns:
    --------
    intensity_analysis : dict
        Intensity map analysis results (includes phase and amplitude for reference)
    """
    print(f"\n📊 Calculating EM near-field intensity map from transmitted field...")
    
    try:
        # Use BackNear monitor
        back_near_data = list(back_monitors.values())[0]
        ez_field = back_near_data['ez']
        ex_field = back_near_data['ex']
        ey_field = back_near_data['ey']
        
        print(f"  • Field size: {ez_field.shape}")
        print(f"  • Monitor size: {size_y_nm:.0f} × {size_z_nm:.0f} nm")
        
        # Total intensity (all components) - PRIMARY OUTPUT
        intensity_map = np.abs(ez_field)**2 + np.abs(ex_field)**2 + np.abs(ey_field)**2
        
        # Calculate phase from Ez field (for reference)
        phase_map = np.angle(ez_field)  # Phase in radians (-π to π)
        
        # Calculate amplitude (for reference)
        amplitude_map = np.abs(ez_field)
        
        # Intensity statistics (PRIMARY)
        intensity_mean = np.mean(intensity_map)
        intensity_std = np.std(intensity_map)
        intensity_min = np.min(intensity_map)
        intensity_max = np.max(intensity_map)
        intensity_range = intensity_max - intensity_min
        
        # Phase statistics (reference)
        phase_mean = np.mean(phase_map)
        phase_std = np.std(phase_map)
        phase_min = np.min(phase_map)
        phase_max = np.max(phase_map)
        phase_range = phase_max - phase_min
        
        # Amplitude statistics (reference)
        amplitude_mean = np.mean(amplitude_map)
        amplitude_std = np.std(amplitude_map)
        
        intensity_analysis = {
            'intensity_map': intensity_map,          # PRIMARY OUTPUT
            'phase_map': phase_map,                  # Reference
            'amplitude_map': amplitude_map,          # Reference
            'intensity_mean': intensity_mean,
            'intensity_std': intensity_std,
            'intensity_min': intensity_min,
            'intensity_max': intensity_max,
            'intensity_range': intensity_range,
            'phase_mean': phase_mean,
            'phase_std': phase_std,
            'phase_min': phase_min,
            'phase_max': phase_max,
            'phase_range': phase_range,
            'amplitude_mean': amplitude_mean,
            'amplitude_std': amplitude_std,
            'wavelength_nm': wavelength_nm,
            'size_y_nm': size_y_nm,
            'size_z_nm': size_z_nm
        }
        
        print(f"  📐 Intensity map statistics (PRIMARY OUTPUT):")
        print(f"    • Mean intensity: {intensity_mean:.4e}")
        print(f"    • Std intensity: {intensity_std:.4e}")
        print(f"    • Intensity range: [{intensity_min:.4e}, {intensity_max:.4e}]")
        print(f"  📐 Phase map statistics (reference):")
        print(f"    • Mean phase: {phase_mean:.4f} rad ({phase_mean/np.pi:.2f}π)")
        print(f"    • Phase range: {phase_range:.4f} rad ({phase_range/np.pi:.2f}π)")
        print(f"  📐 Amplitude statistics (reference):")
        print(f"    • Mean amplitude: {amplitude_mean:.4e}")
        
        return intensity_analysis
        
    except Exception as e:
        print(f"    ⚠️ Intensity map calculation failed: {e}")
        return {}


def visualize_intensity_map(intensity_analysis, mask_info, wavelength_nm=535, pillar_height_nm=600, 
                        delta_n=0.04, n_base=1.5, resolution_nm=0.03, incident_deg=0):
    """Visualize EM near-field intensity map results with parameters in filename
    
    Primary output: Intensity map (|Ex|² + |Ey|² + |Ez|²)
    Reference: Phase and amplitude maps
    """
    
    if not intensity_analysis:
        print(f"⚠️ No intensity map data to visualize.")
        return
    
    print(f"\n🎨 Generating intensity map visualization...")
    
    intensity_map = intensity_analysis['intensity_map']  # PRIMARY
    phase_map = intensity_analysis['phase_map']          # Reference
    amplitude_map = intensity_analysis['amplitude_map']  # Reference
    size_y_nm = intensity_analysis['size_y_nm']
    size_z_nm = intensity_analysis['size_z_nm']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    extent = [-size_z_nm*0.5, size_z_nm*0.5, -size_y_nm*0.5, size_y_nm*0.5]
    
    # 1. Intensity map (PRIMARY OUTPUT)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(intensity_map, extent=extent, cmap='hot', origin='lower')
    ax1.set_title('EM Near-Field Intensity Map\n|Ex|² + |Ey|² + |Ez|² (PRIMARY OUTPUT)', 
                  fontsize=14, fontweight='bold', color='darkred')
    ax1.set_xlabel('z (nm)')
    ax1.set_ylabel('y (nm)')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    ax1.grid(True, alpha=0.3)
    
    # 2. Phase map (reference)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(phase_map, extent=extent, cmap='hsv', origin='lower',
                     vmin=-np.pi, vmax=np.pi)
    ax2.set_title('Phase Map (reference)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('z (nm)')
    ax2.set_ylabel('y (nm)')
    cbar2 = plt.colorbar(im2, ax=ax2, label='Phase (rad)')
    cbar2.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar2.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax2.grid(True, alpha=0.3)
    
    # 3. Amplitude map (reference)
    ax3 = axes[0, 2]
    im3 = ax3.imshow(amplitude_map, extent=extent, cmap='viridis', origin='lower')
    ax3.set_title('Amplitude Map |Ez| (reference)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('z (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3, label='Amplitude')
    ax3.grid(True, alpha=0.3)
    
    # 4. Intensity histogram (PRIMARY)
    ax4 = axes[1, 0]
    ax4.hist(intensity_map.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
    ax4.set_xlabel('Intensity')
    ax4.set_ylabel('Count')
    ax4.set_title('Intensity Distribution (PRIMARY)', fontsize=14, fontweight='bold', color='darkred')
    ax4.axvline(x=intensity_analysis['intensity_mean'], color='darkred', linestyle='--', 
                linewidth=2, label=f"Mean: {intensity_analysis['intensity_mean']:.2e}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Phase histogram (reference)
    ax5 = axes[1, 1]
    ax5.hist(phase_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax5.set_xlabel('Phase (rad)')
    ax5.set_ylabel('Count')
    ax5.set_title('Phase Distribution (reference)', fontsize=14, fontweight='bold')
    ax5.axvline(x=intensity_analysis['phase_mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {intensity_analysis['phase_mean']:.2f}")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Intensity profile (center line)
    ax6 = axes[1, 2]
    center_idx = intensity_map.shape[0] // 2
    intensity_profile = intensity_map[center_idx, :]
    z_coords = np.linspace(-size_z_nm*0.5, size_z_nm*0.5, len(intensity_profile))
    
    ax6.plot(z_coords, intensity_profile, 'r-', linewidth=2, label='Intensity profile (y=0)')
    ax6.set_xlabel('z (nm)')
    ax6.set_ylabel('Intensity')
    ax6.set_title('Intensity Profile at y=0 (PRIMARY)', fontsize=14, fontweight='bold', color='darkred')
    ax6.axhline(y=intensity_analysis['intensity_mean'], color='darkred', linestyle='--', alpha=0.5, label=f'Mean')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with parameters in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = (f"wl{wavelength_nm:.0f}nm_"
                f"h{pillar_height_nm:.0f}nm_"
                f"dn{delta_n:.3f}_"
                f"nb{n_base:.2f}_"
                f"res{resolution_nm:.3f}_"
                f"inc{incident_deg:.0f}deg_"
                f"size{int(size_y_nm)}x{int(size_z_nm)}nm")
    
    filename = f"intensity_map_analysis_{param_str}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {filename}")


def run_random_pillar_simulation(mask_file=MASK_FILE, resolution_nm=RESOLUTION_NM, 
                                 pml_nm=PML_NM, size_x_nm=SIZE_X_NM,
                                 pillar_height_nm=PILLAR_HEIGHT_NM, 
                                 pillar_x_center=PILLAR_X_CENTER,
                                 incident_deg=INCIDENT_DEG, wavelength_nm=WAVELENGTH_NM,
                                 n_base=N_BASE, delta_n=DELTA_N,
                                 cell_size_scale=CELL_SIZE_SCALE,
                                 auto_terminate=AUTO_TERMINATE,
                                 decay_threshold=DECAY_THRESHOLD,
                                 source_width_factor=SOURCE_WIDTH_FACTOR):
    """Run random pillar + plane wave simulation and calculate phase map
    
    HOE 시뮬레이션 코드 구조 기반, 모든 단위 nm로 통일:
    - 물리적 파라미터는 HOE 표준 (해상도, PML, 파장, 굴절률)
    - 셀 크기(y, z)는 실제 패턴 크기에 맞춤 (왜곡 방지)
    - 랜덤 필러 패턴 (바이너리 그레이팅 대신)
    - 위상맵 계산 (회절 효율 대신)
    - 모든 거리 단위는 nm
    """
    
    print("=" * 80)
    print("🔬 Random Pillar + Plane Wave + Phase Map Simulation (nm units)")
    print("=" * 80)
    
    # Load mask
    mask, mask_info = load_random_pillar_mask(mask_file)
    
    # Calculate cell size from mask (1 pixel = 1 nm)
    mask_height_nm, mask_width_nm = mask.shape
    size_z_nm_raw = mask_height_nm * cell_size_scale  # nm (already in nm)
    size_y_nm_raw = mask_width_nm * cell_size_scale   # nm (already in nm)
    
    # MEEP grid resolution (pixels/nm) - round to nearest integer
    target_ny = int(np.round(size_y_nm_raw * resolution_nm))  # y direction grid points
    target_nz = int(np.round(size_z_nm_raw * resolution_nm))  # z direction grid points
    
    # Adjust cell size to ensure integer number of pixels (avoid MEEP warning)
    size_y_nm = target_ny / resolution_nm  # Adjusted to exact grid
    size_z_nm = target_nz / resolution_nm  # Adjusted to exact grid
    
    print(f"\n📐 Cell size from mask:")
    print(f"  • Mask size: {mask.shape} pixels (height × width)")
    print(f"  • 1 pixel = 1 nm")
    print(f"  • Raw cell size: {size_y_nm_raw:.0f} × {size_z_nm_raw:.0f} nm (Y × Z)")
    print(f"  • Adjusted cell size: {size_y_nm:.2f} × {size_z_nm:.2f} nm (Y × Z)")
    print(f"  • Adjustment: {abs(size_y_nm-size_y_nm_raw):.2f} nm ({abs(size_y_nm-size_y_nm_raw)/size_y_nm_raw*100:.3f}%)")
    print(f"  • Scale factor: {cell_size_scale}")
    
    print(f"\n📐 MEEP grid size (integer pixels):")
    print(f"  • ny (y direction): {target_ny} points ({size_y_nm:.2f} nm × {resolution_nm} pixels/nm = {target_ny})")
    print(f"  • nz (z direction): {target_nz} points ({size_z_nm:.2f} nm × {resolution_nm} pixels/nm = {target_nz})")
    
    # Resample mask to MEEP grid resolution
    resampled_mask = resample_mask_to_cell_size(mask, target_ny, target_nz)
    
    print(f"\n📋 Simulation parameters (all in nm):")
    print(f"  • Cell size: {size_x_nm:.0f} × {size_y_nm:.0f} × {size_z_nm:.0f} nm (X × Y × Z)")
    print(f"  • Pillar thickness: {pillar_height_nm:.0f} nm")
    print(f"  • Resolution: {resolution_nm} pixels/nm")
    print(f"  • Wavelength: {wavelength_nm} nm")
    print(f"  • Incident angle: {incident_deg}° (normal incidence)")
    print(f"\n  Refractive index:")
    print(f"  • Pattern 0 (background): n = {n_base:.2f}")
    print(f"  • Pattern 1 (pillar):     n = {n_base + delta_n:.2f}")
    print(f"  • Δn: {delta_n:.3f} (HOE standard)")
    print(f"  • Pattern type: Random pillar (non-periodic)")
    
    # Physical parameters (MEEP uses normalized units, but we convert)
    incident_angle = math.radians(incident_deg)
    frequency = 1.0 / wavelength_nm  # in 1/nm
    
    # Create geometry (HOE-style, nm units)
    geometry, default_material = create_random_pillar_geometry(
        resampled_mask, size_x_nm, size_y_nm, size_z_nm,
        n_base=n_base, delta_n=delta_n, 
        thickness_um=pillar_height_nm,
        pillar_x_center=pillar_x_center
    )
    
    # Cell and boundary (nm units)
    # Ensure total cell size has integer number of pixels
    total_x_nm = size_x_nm + 2*pml_nm
    n_x = int(np.round(total_x_nm * resolution_nm))
    total_x_nm_adjusted = n_x / resolution_nm
    
    cell_size = mp.Vector3(total_x_nm_adjusted, size_y_nm, size_z_nm)
    pml_layers = [mp.PML(thickness=pml_nm, direction=mp.X)]
    
    print(f"\n📐 Total cell size (with PML, adjusted for integer pixels):")
    print(f"  • X: {total_x_nm_adjusted:.2f} nm ({n_x} pixels)")
    print(f"  • Y: {size_y_nm:.2f} nm ({target_ny} pixels)")
    print(f"  • Z: {size_z_nm:.2f} nm ({target_nz} pixels)")
    
    # k-vector for plane wave
    k_vec = mp.Vector3(n_base*frequency, 0, 0)
    k_point = k_vec
    
    print(f"\n🌊 Plane wave setup:")
    print(f"  • k-vector: ({k_vec.x:.6f}, {k_vec.y:.6f}, {k_vec.z:.6f})")
    print(f"  • Frequency: {frequency:.6f} (1/nm)")
    
    # Plane wave source (nm units)
    x_src = -0.4*size_x_nm
    src_center = mp.Vector3(x_src, 0, 0)
    src_size = mp.Vector3(0, size_y_nm, size_z_nm)
    
    # Use GaussianSource for better auto-termination
    # Width: several wavelengths for smooth turn-on
    src_width = wavelength_nm * source_width_factor
    
    sources = [
        mp.Source(
            src=mp.GaussianSource(frequency=frequency, width=src_width),
            component=mp.Ez,
            center=src_center,
            size=src_size,
            amp_func=pw_amp(k_point, src_center)
        )
    ]
    
    print(f"  • Source type: GaussianSource (for auto-termination)")
    print(f"  • Source width: {src_width:.0f} nm/c (~{src_width/wavelength_nm:.1f} periods)")
    print(f"  • Source position: x = {x_src:.0f} nm")
    print(f"  • Source size: {src_size.y:.0f} × {src_size.z:.0f} nm")
    
    # Monitors (front and back, nm units)
    print(f"\n📡 Setting up monitors...")
    
    front_monitor_positions = [
        pillar_x_center - pillar_height_nm/2 - 300.0,  # 300 nm = 0.3 μm
        pillar_x_center - pillar_height_nm/2 - 100.0   # 100 nm = 0.1 μm
    ]
    front_monitor_names = ["FrontFar", "FrontNear"]
    
    back_monitor_positions = [
        pillar_x_center + pillar_height_nm/2 + 100.0,   # 100 nm = 0.1 μm
        pillar_x_center + pillar_height_nm/2 + 300.0    # 300 nm = 0.3 μm
    ]
    back_monitor_names = ["BackNear", "BackFar"]
    
    all_monitor_positions = front_monitor_positions + back_monitor_positions
    all_monitor_names = front_monitor_names + back_monitor_names
    monitor_volumes = []
    
    print(f"  📥 Front monitors:")
    for x_pos, name in zip(front_monitor_positions, front_monitor_names):
        monitor_vol = mp.Volume(
            center=mp.Vector3(x_pos, 0, 0),
            size=mp.Vector3(0, size_y_nm, size_z_nm)  # 100% of cell size
        )
        monitor_volumes.append((monitor_vol, name, x_pos, "front"))
        print(f"    • {name}: x = {x_pos:.0f} nm (100% of cell size)")
    
    print(f"  📤 Back monitors:")
    for x_pos, name in zip(back_monitor_positions, back_monitor_names):
        monitor_vol = mp.Volume(
            center=mp.Vector3(x_pos, 0, 0),
            size=mp.Vector3(0, size_y_nm, size_z_nm)  # 100% of cell size
        )
        monitor_volumes.append((monitor_vol, name, x_pos, "back"))
        print(f"    • {name}: x = {x_pos:.0f} nm (100% of cell size)")
    
    # Create simulation (nm units)
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution_nm,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k_point,
        default_material=default_material,
        geometry=geometry,
        symmetries=[]
    )
    
    # Run simulation
    if auto_terminate:
        print(f"\n🚀 Running simulation with auto-termination...")
        print(f"  • Geometry count: {len(geometry)}")
        print(f"  • Monitor count: {len(monitor_volumes)}")
        
        # Auto-termination settings
        # Monitor position: use the farthest back monitor for field decay check
        farthest_back_monitor_x = max(back_monitor_positions)
        
        print(f"\n📊 Auto-termination settings:")
        print(f"  • Monitor position: x = {farthest_back_monitor_x:.0f} nm (farthest back monitor)")
        print(f"  • Decay threshold: {decay_threshold:.0e} (relative)")
        print(f"  • Component monitored: Ez")
        print(f"  • Source width: {src_width:.0f} nm/c (~{source_width_factor:.1f} periods)")
        print(f"  • Auto-stop when steady state reached")
        
        # Run with auto-termination
        # Stop when Ez field at the back monitor decays to threshold
        sim.run(
            mp.at_beginning(mp.output_epsilon),
            until_after_sources=mp.stop_when_fields_decayed(
                dt=wavelength_nm / 20.0,  # Check every ~1/20 wavelength time
                c=mp.Ez,  # Monitor Ez component
                pt=mp.Vector3(farthest_back_monitor_x, 0, 0),  # At back monitor
                decay_by=decay_threshold  # Stop when field decays to this level
            )
        )
        
        final_time = sim.meep_time()
        print(f"\n✅ Simulation complete!")
        print(f"  • Final time: {final_time:.0f} nm/c")
        print(f"  • Steady state reached (field decayed to {decay_threshold:.0e})")
    
    else:
        # Manual termination with fixed time
        print(f"\n🚀 Running simulation with fixed time...")
        print(f"  • Geometry count: {len(geometry)}")
        print(f"  • Monitor count: {len(monitor_volumes)}")
        
        # Calculate required simulation time (nm/c units)
        max_distance = abs(x_src) + max(abs(pos) for pos in all_monitor_positions) + pillar_height_nm/2
        travel_time = max_distance * n_base
        extra_time = wavelength_nm * 5
        total_time = travel_time + extra_time
        
        print(f"  • Max distance: {max_distance:.0f} nm")
        print(f"  • Travel time: {travel_time:.0f} nm/c")
        print(f"  • Total simulation time: {total_time:.0f} nm/c")
        
        sim.run(until=total_time)
        
        print(f"\n✅ Simulation complete!")
    
    # Visualize refractive index
    visualize_actual_meep_pattern(sim, size_x_nm, size_y_nm, size_z_nm, 
                                  pillar_x_center, pillar_height_nm,
                                  wavelength_nm=wavelength_nm,
                                  delta_n=delta_n,
                                  n_base=n_base,
                                  resolution_nm=resolution_nm,
                                  incident_deg=incident_deg)
    
    # Collect monitor data
    print(f"\n📊 Collecting monitor data...")
    
    monitor_data = {}
    front_monitors = {}
    back_monitors = {}
    
    for monitor_vol, name, x_pos, position_type in monitor_volumes:
        print(f"  • {name} monitor (x = {x_pos:.0f} nm)...")
        
        ez_data = sim.get_array(center=monitor_vol.center, size=monitor_vol.size, 
                               component=mp.Ez)
        ex_data = sim.get_array(center=monitor_vol.center, size=monitor_vol.size,
                               component=mp.Ex)
        ey_data = sim.get_array(center=monitor_vol.center, size=monitor_vol.size,
                               component=mp.Ey)
        
        intensity = np.abs(ez_data)**2 + np.abs(ex_data)**2 + np.abs(ey_data)**2
        
        monitor_info = {
            'ez': ez_data,
            'ex': ex_data,
            'ey': ey_data,
            'intensity': intensity,
            'x_pos': x_pos,
            'position_type': position_type,
            'extent': [-size_z_nm*0.5, size_z_nm*0.5, -size_y_nm*0.5, size_y_nm*0.5]
        }
        
        monitor_data[name] = monitor_info
        
        if position_type == "front":
            front_monitors[name] = monitor_info
        elif position_type == "back":
            back_monitors[name] = monitor_info
    
    # Calculate EM near-field intensity map
    intensity_analysis = calculate_intensity_map_from_monitors(back_monitors, size_y_nm, 
                                                                size_z_nm, wavelength_nm)
    
    # Visualize intensity map
    if intensity_analysis:
        visualize_intensity_map(intensity_analysis, mask_info, 
                               wavelength_nm=wavelength_nm,
                               pillar_height_nm=pillar_height_nm,
                               delta_n=delta_n,
                               n_base=n_base,
                               resolution_nm=resolution_nm,
                               incident_deg=incident_deg)
        
        # Save intensity map (PRIMARY) with all parameters in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("meep_output", exist_ok=True)
        
        # Create descriptive filename with all parameters
        param_str = (f"wl{wavelength_nm:.0f}nm_"
                    f"h{pillar_height_nm:.0f}nm_"
                    f"dn{delta_n:.3f}_"
                    f"nb{n_base:.2f}_"
                    f"res{resolution_nm:.3f}_"
                    f"inc{incident_deg:.0f}deg_"
                    f"size{int(size_y_nm)}x{int(size_z_nm)}nm")
        
        filename_base = f"intensity_map_{param_str}_{timestamp}"
        
        # Save PRIMARY output (intensity) and reference data (phase, amplitude)
        np.save(f"meep_output/{filename_base}.npy", intensity_analysis['intensity_map'])
        np.save(f"meep_output/phase_map_{param_str}_{timestamp}.npy", intensity_analysis['phase_map'])
        np.save(f"meep_output/amplitude_map_{param_str}_{timestamp}.npy", intensity_analysis['amplitude_map'])
        print(f"\n💾 Intensity map saved (PRIMARY): meep_output/{filename_base}.npy")
        print(f"   Phase & Amplitude maps saved (reference)")
        print(f"   Parameters: wavelength={wavelength_nm}nm, height={pillar_height_nm}nm, Δn={delta_n}, n_base={n_base}")
    
    # Basic visualization
    print(f"\n🎨 Generating field visualizations...")
    
    # XY plane
    fig, ax = plt.subplots(figsize=(14, 6))
    output_plane_xy = mp.Volume(center=mp.Vector3(0, 0, 0),
                                size=mp.Vector3(size_x_nm, size_y_nm, 0))
    sim.plot2D(fields=mp.Ez, output_plane=output_plane_xy, ax=ax)
    ax.set_title(f"Ez Field @ z=0, Random Pillar Pattern", fontsize=14, fontweight='bold')
    
    # Mark monitors
    for monitor_vol, name, x_pos, pos_type in monitor_volumes:
        color = 'blue' if pos_type == "front" else 'red'
        ax.axvline(x=x_pos, color=color, linestyle='-', alpha=0.8, linewidth=2)
    
    # Mark pillar
    ax.axvline(x=pillar_x_center, color='yellow', linestyle='-', alpha=0.8, linewidth=4)
    ax.text(pillar_x_center, 0, 'Pillar', ha='center', va='center', 
            color='yellow', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    # Save XY field with parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    param_str = (f"wl{wavelength_nm:.0f}nm_"
                f"h{pillar_height_nm:.0f}nm_"
                f"dn{delta_n:.3f}_"
                f"nb{n_base:.2f}_"
                f"res{resolution_nm:.3f}_"
                f"inc{incident_deg:.0f}deg_"
                f"size{int(size_y_nm)}x{int(size_z_nm)}nm")
    
    field_xy_filename = f"field_xy_{param_str}_{timestamp}.png"
    plt.savefig(field_xy_filename, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"\n🎉 Random pillar phase map simulation complete!")
    print(f"📁 Output files (all with parameters in filename):")
    print(f"  • meep_refractive_index_{param_str}_*.png")
    print(f"  • phase_map_analysis_{param_str}_*.png")
    print(f"  • {field_xy_filename}")
    print(f"  • meep_output/phase_map_{param_str}_*.npy")
    print(f"  • meep_output/amplitude_map_{param_str}_*.npy")
    print(f"\n  Parameters encoded in filename:")
    print(f"    - wl: wavelength (nm)")
    print(f"    - h: pillar/film height (nm)")
    print(f"    - dn: delta_n (refractive index modulation)")
    print(f"    - nb: n_base (base refractive index)")
    print(f"    - res: resolution (pixels/nm)")
    print(f"    - inc: incident angle (degrees)")
    print(f"    - size: cell size YxZ (nm)")
    
    return {
        'all_monitors': monitor_data,
        'front_monitors': front_monitors,
        'back_monitors': back_monitors,
        'intensity_analysis': intensity_analysis,  # PRIMARY OUTPUT
        'mask_info': mask_info,
        'simulation_params': {
            'wavelength_nm': wavelength_nm,
            'pillar_height_nm': pillar_height_nm,
            'n_base': n_base,
            'delta_n': delta_n,
            'incident_deg': incident_deg
        }
    }


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("Random Pillar Phase Map Simulation (nm units)")
    print("=" * 80)
    
    print(f"\nSimulation settings (all in nm):")
    print(f"  • Mask file: {MASK_FILE}")
    print(f"  • Wavelength: {WAVELENGTH_NM} nm")
    print(f"  • Pillar height: {PILLAR_HEIGHT_NM} nm")
    print(f"  • Pillar index: {N_BASE + DELTA_N} (n_base + Δn)")
    print(f"  • Background index: {N_BASE}")
    print(f"  • Δn: {DELTA_N}")
    print(f"  • Resolution: {RESOLUTION_NM} pixels/nm")
    print(f"  • Incident angle: {INCIDENT_DEG}°")
    
    # Check if mask file exists
    if not os.path.exists(MASK_FILE):
        print(f"\n❌ Error: Mask file not found: {MASK_FILE}")
        print(f"Please run random_pillar_generator.py first to generate the mask.")
        return
    
    # Run simulation
    results = run_random_pillar_simulation()
    
    print(f"\n✅ All done!")
    print(f"Results stored in 'results' variable")


def generate_single_training_sample(sample_idx, output_dir, 
                                   pillar_params, simulation_params,
                                   visualize=False):
    """Generate one training sample (input mask + output EM near-field intensity map)
    
    Parameters:
    -----------
    sample_idx : int
        Sample index for naming
    output_dir : Path
        Output directory path
    pillar_params : dict
        Random pillar generation parameters
    simulation_params : dict
        MEEP simulation parameters
    visualize : bool
        Whether to save visualization images
        
    Returns:
    --------
    success : bool
        Whether sample generation succeeded
    sample_info : dict
        Sample information
    """
    from pathlib import Path
    from random_pillar_generator import RandomPillarGenerator
    
    print(f"\n{'='*80}")
    print(f"📦 Generating Training Sample {sample_idx}")
    print(f"{'='*80}")
    
    try:
        # 1. Generate random pillar pattern
        print(f"\n1️⃣ Generating random pillar pattern...")
        generator = RandomPillarGenerator(
            pillar_radius=pillar_params['pillar_radius'],
            min_edge_distance=pillar_params['min_edge_distance'],
            domain_size=pillar_params['domain_size'],
            initial_density=pillar_params['initial_density'],
            max_attempts=pillar_params.get('max_attempts', 10000)
        )
        
        pillars = generator.generate_pillars()
        
        # Generate binary mask
        width, height = pillar_params['domain_size']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for cx, cy in pillars:
            cx_px = int(cx)
            cy_px = int(cy)
            radius_px = int(pillar_params['pillar_radius'])
            
            y_indices, x_indices = np.ogrid[:height, :width]
            distances = np.sqrt((x_indices - cx_px)**2 + ((height - 1 - y_indices) - cy_px)**2)
            mask[distances <= radius_px] = 1
        
        fill_ratio = np.sum(mask) / mask.size * 100
        print(f"  ✓ Mask generated: {mask.shape}, fill ratio: {fill_ratio:.1f}%")
        
        # 2. Run MEEP simulation
        print(f"\n2️⃣ Running MEEP simulation...")
        
        # Save temporary mask file
        temp_mask_file = f"temp_mask_{sample_idx}.npy"
        np.save(temp_mask_file, mask)
        
        # Run simulation
        results = run_random_pillar_simulation(
            mask_file=temp_mask_file,
            resolution_nm=simulation_params['resolution_nm'],
            pml_nm=simulation_params['pml_nm'],
            size_x_nm=simulation_params['size_x_nm'],
            pillar_height_nm=simulation_params['pillar_height_nm'],
            pillar_x_center=simulation_params['pillar_x_center'],
            incident_deg=simulation_params['incident_deg'],
            wavelength_nm=simulation_params['wavelength_nm'],
            n_base=simulation_params['n_base'],
            delta_n=simulation_params['delta_n'],
            cell_size_scale=simulation_params.get('cell_size_scale', 1.0),
            auto_terminate=simulation_params.get('auto_terminate', True),
            decay_threshold=simulation_params.get('decay_threshold', 1e-4),
            source_width_factor=simulation_params.get('source_width_factor', 10)
        )
        
        # Clean up temp file
        if os.path.exists(temp_mask_file):
            os.remove(temp_mask_file)
        
        # 3. Extract EM near-field intensity map
        intensity_analysis = results.get('intensity_analysis', {})
        if not intensity_analysis:
            print(f"  ❌ Failed to get intensity map")
            return False, {}
        
        intensity_map = intensity_analysis['intensity_map']
        phase_map = intensity_analysis['phase_map']  # For reference
        print(f"  ✓ Intensity map extracted: {intensity_map.shape}")
        
        # 4. Save input and output
        print(f"\n3️⃣ Saving data...")
        
        # Create directories
        input_dir = output_dir / 'inputs'
        output_intensity_dir = output_dir / 'outputs'
        input_dir.mkdir(parents=True, exist_ok=True)
        output_intensity_dir.mkdir(parents=True, exist_ok=True)
        
        import cv2
        
        # Resize to 4096×4096 for consistent dataset size
        TARGET_SIZE = (4096, 4096)
        
        # Save input mask as PNG (0-255 grayscale)
        sample_name = f"sample_{sample_idx:04d}"
        input_path = input_dir / f"{sample_name}.png"
        mask_img = (mask * 255).astype(np.uint8)
        
        # Resize mask to 4096×4096
        mask_img_resized = cv2.resize(mask_img, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(input_path), mask_img_resized)
        print(f"  ✓ Input mask saved: {input_path} ({mask.shape} → {TARGET_SIZE})")
        
        # Save output intensity map as .npy (preserve precision) - PRIMARY OUTPUT
        output_path = output_intensity_dir / f"{sample_name}.npy"
        
        # Resize intensity map to 4096×4096 (cubic interpolation for smooth intensity)
        intensity_map_resized = cv2.resize(intensity_map.astype(np.float32), TARGET_SIZE, 
                                          interpolation=cv2.INTER_CUBIC)
        np.save(output_path, intensity_map_resized.astype(np.float32))
        print(f"  ✓ Output intensity map saved (PRIMARY): {output_path} ({intensity_map.shape} → {TARGET_SIZE})")
        
        # 5. Optional: Save visualization
        if visualize:
            vis_dir = output_dir / 'visualizations'
            vis_dir.mkdir(exist_ok=True)
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Input mask (resized 4096×4096)
            axes[0].imshow(mask_img_resized, cmap='gray')
            axes[0].set_title(f'Input: Random Pillar Mask (Resized)\n{mask_img_resized.shape}, Fill: {fill_ratio:.1f}%')
            axes[0].axis('off')
            
            # Output intensity map (resized 4096×4096 - PRIMARY)
            im1 = axes[1].imshow(intensity_map_resized, cmap='hot')
            axes[1].set_title(f'Output: EM Intensity Map (Resized - PRIMARY)\n{intensity_map_resized.shape}')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], label='Intensity')
            
            # Intensity histogram (resized - PRIMARY)
            axes[2].hist(intensity_map_resized.flatten(), bins=50, alpha=0.7, color='red', edgecolor='black')
            axes[2].set_xlabel('Intensity')
            axes[2].set_ylabel('Count')
            axes[2].set_title('Intensity Distribution (Resized - PRIMARY)')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            vis_path = vis_dir / f"{sample_name}_vis.png"
            plt.savefig(vis_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Visualization saved: {vis_path}")
        
        # Sample info
        sample_info = {
            'sample_idx': sample_idx,
            'input_shape': mask_img_resized.shape,  # Final resized shape (4096×4096)
            'output_shape': intensity_map_resized.shape,  # Final resized shape (4096×4096)
            'original_input_shape': mask.shape,  # Original simulation shape
            'original_output_shape': intensity_map.shape,  # Original simulation shape
            'fill_ratio': fill_ratio,
            'num_pillars': len(pillars),
            'pillar_params': pillar_params,
            'simulation_params': simulation_params,
            'intensity_mean': float(np.mean(intensity_map_resized)),
            'intensity_std': float(np.std(intensity_map_resized)),
            'intensity_min': float(np.min(intensity_map_resized)),
            'intensity_max': float(np.max(intensity_map_resized)),
            # Phase info for reference (original resolution)
            'phase_mean': float(np.mean(phase_map)),
            'phase_std': float(np.std(phase_map)),
            'phase_min': float(np.min(phase_map)),
            'phase_max': float(np.max(phase_map))
        }
        
        print(f"\n✅ Sample {sample_idx} generated successfully!")
        return True, sample_info
        
    except Exception as e:
        print(f"\n❌ Error generating sample {sample_idx}: {e}")
        import traceback
        traceback.print_exc()
        return False, {}


def generate_training_dataset(num_samples=100, 
                              output_dir='data/forward_phase',
                              pillar_params=None,
                              simulation_params=None,
                              visualize_samples=True,
                              start_idx=0):
    """Generate training dataset for forward phase prediction
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    output_dir : str or Path
        Output directory path
    pillar_params : dict
        Random pillar generation parameters
        Default: {
            'pillar_radius': varies randomly 8-12 nm,
            'min_edge_distance': 5.0 nm,
            'domain_size': (4096, 4096) nm,
            'initial_density': 100.0 pillars/μm²
        }
    simulation_params : dict
        MEEP simulation parameters (uses module defaults if None)
    visualize_samples : bool
        Whether to save visualization for each sample
    start_idx : int
        Starting sample index (useful for continuing generation)
        
    Returns:
    --------
    dataset_info : dict
        Dataset generation summary
    """
    from pathlib import Path
    import json
    
    print(f"\n{'='*80}")
    print(f"🚀 Training Dataset Generation")
    print(f"{'='*80}")
    print(f"Number of samples: {num_samples}")
    print(f"Output directory: {output_dir}")
    print(f"Starting index: {start_idx}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default pillar parameters (will vary per sample)
    if pillar_params is None:
        pillar_params = {
            'pillar_radius': 10.0,  # Will vary per sample
            'min_edge_distance': 5.0,
            'domain_size': (4096, 4096),
            'initial_density': 100.0,
            'max_attempts': 10000
        }
    
    # Default simulation parameters
    if simulation_params is None:
        simulation_params = {
            'resolution_nm': RESOLUTION_NM,
            'pml_nm': PML_NM,
            'size_x_nm': SIZE_X_NM,
            'pillar_height_nm': PILLAR_HEIGHT_NM,
            'pillar_x_center': PILLAR_X_CENTER,
            'incident_deg': INCIDENT_DEG,
            'wavelength_nm': WAVELENGTH_NM,
            'n_base': N_BASE,
            'delta_n': DELTA_N,
            'cell_size_scale': CELL_SIZE_SCALE,
            'auto_terminate': AUTO_TERMINATE,
            'decay_threshold': DECAY_THRESHOLD,
            'source_width_factor': SOURCE_WIDTH_FACTOR
        }
    
    print(f"\n📋 Pillar parameters (base):")
    print(f"  • Domain size: {pillar_params['domain_size']} nm")
    print(f"  • Pillar radius: {pillar_params['pillar_radius']} nm (will vary per sample)")
    print(f"  • Min edge distance: {pillar_params['min_edge_distance']} nm")
    print(f"  • Initial density: {pillar_params['initial_density']} pillars/μm²")
    
    print(f"\n📋 Simulation parameters:")
    for key, value in simulation_params.items():
        print(f"  • {key}: {value}")
    
    # Generate samples
    print(f"\n{'='*80}")
    print(f"Starting sample generation...")
    print(f"{'='*80}")
    
    successful_samples = []
    failed_samples = []
    all_sample_info = []
    
    for i in range(num_samples):
        sample_idx = start_idx + i
        
        # Use fixed pillar parameters (no randomization)
        # 고정된 파라미터 사용 (샘플마다 동일한 평균 특성 유지)
        current_pillar_params = pillar_params.copy()
        # 다양성을 원하면 아래 주석을 해제하세요:
        # current_pillar_params['pillar_radius'] = np.random.uniform(40.0, 50.0)
        # current_pillar_params['initial_density'] = np.random.uniform(35.0, 45.0)
        
        # Generate sample
        success, sample_info = generate_single_training_sample(
            sample_idx=sample_idx,
            output_dir=output_dir,
            pillar_params=current_pillar_params,
            simulation_params=simulation_params,
            visualize=visualize_samples
        )
        
        if success:
            successful_samples.append(sample_idx)
            all_sample_info.append(sample_info)
        else:
            failed_samples.append(sample_idx)
        
        # Progress
        print(f"\n{'─'*80}")
        print(f"Progress: {i+1}/{num_samples} samples processed")
        print(f"  • Successful: {len(successful_samples)}")
        print(f"  • Failed: {len(failed_samples)}")
        print(f"{'─'*80}")
    
    # Save dataset metadata
    print(f"\n{'='*80}")
    print(f"💾 Saving dataset metadata...")
    
    metadata = {
        'num_samples': num_samples,
        'successful_samples': len(successful_samples),
        'failed_samples': len(failed_samples),
        'start_idx': start_idx,
        'pillar_params_base': pillar_params,
        'simulation_params': simulation_params,
        'sample_info': all_sample_info,
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = output_dir / 'dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Metadata saved: {metadata_path}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"🎉 Dataset Generation Complete!")
    print(f"{'='*80}")
    print(f"\n📊 Summary:")
    print(f"  • Total samples requested: {num_samples}")
    print(f"  • Successful: {len(successful_samples)} ({len(successful_samples)/num_samples*100:.1f}%)")
    print(f"  • Failed: {len(failed_samples)} ({len(failed_samples)/num_samples*100:.1f}%)")
    
    print(f"\n📁 Output structure:")
    print(f"  {output_dir}/")
    print(f"    ├── inputs/")
    print(f"    │   ├── sample_0000.png")
    print(f"    │   └── ...")
    print(f"    ├── outputs/")
    print(f"    │   ├── sample_0000.npy")
    print(f"    │   └── ...")
    if visualize_samples:
        print(f"    ├── visualizations/")
        print(f"    │   ├── sample_0000_vis.png")
        print(f"    │   └── ...")
    print(f"    └── dataset_metadata.json")
    
    print(f"\n💡 Usage with PyTorch:")
    print(f"   from pytorch_codes.datasets.hoe_dataset import ForwardIntensityDataset")
    print(f"   dataset = ForwardIntensityDataset('{output_dir}', output_extension='npy')")
    print(f"   sample = dataset[0]")
    print(f"   input_mask = sample['image']  # Shape: (1, H, W)")
    print(f"   intensity_map = sample['target']  # Shape: (1, H, W) - EM Near-Field Intensity")
    
    if failed_samples:
        print(f"\n⚠️  Failed sample indices: {failed_samples[:10]}{'...' if len(failed_samples) > 10 else ''}")
    
    return metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Random Pillar Phase Map Simulation')
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'dataset'],
                       help='Run mode: single simulation or generate dataset')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate (dataset mode)')
    parser.add_argument('--output_dir', type=str, default='data/forward_phase',
                       help='Output directory for dataset')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualizations for each sample')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='Starting sample index')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Single simulation mode (original behavior)
        main()
    elif args.mode == 'dataset':
        # Dataset generation mode
        print("\n" + "="*80)
        print("🔬 Dataset Generation Mode")
        print("="*80)
        
        metadata = generate_training_dataset(
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            visualize_samples=args.visualize,
            start_idx=args.start_idx
        )
        
        print(f"\n✅ Dataset generation complete!")
        print(f"   Dataset ready for PyTorch training!")