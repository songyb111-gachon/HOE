#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔬 Random Pillar Pattern + Plane Wave + Phase Map Simulation
================================================================

3D MEEP simulation with random pillar pattern from random_pillar_generator.py
- Random pillar pattern (not periodic grating)
- 3D plane wave source (using amp_func)
- Phase map calculation and analysis
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
        header = f"{'='*80}\nRandom Pillar Phase Map Simulation Log\nStart: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{'='*80}\n\n"
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
_auto_logger = AutoLogger("random_pillar_phase_simulation.txt")
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
RESOLUTION_NM = 0.03        # 해상도 (pixels/nm) = 30 pixels/μm
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

# Multi-parameter sweep (nm 단위)
PARAMETER_SWEEP = {
    'pillar_height_nm': [600.0],  # 기둥(필름) 두께 (nm)
    'wavelength_nm': [405.0, 532.0, 633.0],  # RGB 파장 (nm)
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
    
    Parameters:
    -----------
    mask : 2D numpy array
        Binary pattern (0 or 1), shape (nz, ny) = (z, y) in MEEP coordinates
    cell_size_x, cell_size_y, cell_size_z : float
        Cell size (nm)
    n_base : float
        Base refractive index (background)
    delta_n : float
        Refractive index modulation (pillar_index = n_base + delta_n)
    thickness_um : float
        Pillar thickness in x direction (nm)
    pillar_x_center : float
        Pillar center position in x direction (nm)
        
    Returns:
    --------
    geometry : list
        MEEP geometry objects
    background_material : mp.Medium
        Background material
    """
    print(f"\n=== Generating Random Pillar Geometry (HOE-style, nm units) ===")
    print(f"Mask size: {mask.shape} (nz × ny)")
    print(f"Base refractive index: {n_base}")
    print(f"Refractive index modulation: Δn = {delta_n}")
    print(f"Pillar refractive index: {n_base + delta_n}")
    print(f"Pillar thickness: {thickness_um:.0f} nm")
    print(f"Pillar x center: {pillar_x_center:.0f} nm")
    
    # Materials (HOE standard)
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
    pillar_count = 0
    for j in range(nz):  # z direction (vertical in pattern)
        z_pos = z_coords[j]
        
        # Check pattern at this z position
        for i in range(ny):  # y direction (horizontal in pattern)
            if mask[j, i] > 0.5:  # Pillar pixel
                y_pos = y_coords[i]
                
                # Create block
                block = mp.Block(
                    size=mp.Vector3(thickness_um, pixel_size_y, pixel_size_z),  # (x, y, z)
                    center=mp.Vector3(pillar_x_center, y_pos, z_pos),  # (x, y, z)
                    material=pillar_material
                )
                geometry.append(block)
                pillar_count += 1
    
    print(f"Generated Block count: {len(geometry):,}")
    print(f"  • Pillar pixels: {pillar_count:,}")
    print(f"  • Block size: {thickness_um:.0f} × {pixel_size_y:.2f} × {pixel_size_z:.2f} nm")
    
    return geometry, background_material


def visualize_actual_meep_pattern(sim, size_x_nm, size_y_nm, size_z_nm, 
                                  pillar_x_center, pillar_thickness_nm,
                                  wavelength_nm=535, delta_n=0.04, n_base=1.5, 
                                  resolution_nm=0.03, incident_deg=0):
    """Visualize actual refractive index distribution used in MEEP simulation (nm units)"""
    
    print(f"\n  📐 Extracting actual MEEP refractive index distribution...")
    
    # YZ plane (pillar pattern at x=pillar_x_center)
    yz_plane_center = mp.Vector3(pillar_x_center, 0, 0)
    yz_plane_size = mp.Vector3(0, size_y_nm * 0.9, size_z_nm * 0.9)
    
    eps_data_yz = sim.get_array(center=yz_plane_center, size=yz_plane_size, 
                                component=mp.Dielectric)
    n_data_yz = np.sqrt(np.real(eps_data_yz))
    
    # XZ plane (side view at y=0)
    xz_plane_center = mp.Vector3(0, 0, 0)
    xz_plane_size = mp.Vector3(size_x_nm * 0.9, 0, size_z_nm * 0.9)
    
    eps_data_xz = sim.get_array(center=xz_plane_center, size=xz_plane_size, 
                                component=mp.Dielectric)
    n_data_xz = np.sqrt(np.real(eps_data_xz))
    
    # XY plane (top view at z=0)
    xy_plane_center = mp.Vector3(0, 0, 0)
    xy_plane_size = mp.Vector3(size_x_nm * 0.9, size_y_nm * 0.9, 0)
    
    eps_data_xy = sim.get_array(center=xy_plane_center, size=xy_plane_size, 
                                component=mp.Dielectric)
    n_data_xy = np.sqrt(np.real(eps_data_xy))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # YZ plane (actual pillar pattern)
    ax1 = axes[0, 0]
    extent_yz = [-size_z_nm*0.45, size_z_nm*0.45, -size_y_nm*0.45, size_y_nm*0.45]
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
    
    # XZ plane (side view)
    ax2 = axes[0, 1]
    extent_xz = [-size_x_nm*0.45, size_x_nm*0.45, -size_z_nm*0.45, size_z_nm*0.45]
    im2 = ax2.imshow(n_data_xz.T, extent=extent_xz, cmap='viridis', origin='lower')
    ax2.set_title(f'XZ Plane: Side View (y = 0 nm)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x (nm)')
    ax2.set_ylabel('z (nm)')
    plt.colorbar(im2, ax=ax2, label='Refractive Index n')
    ax2.grid(True, alpha=0.3)
    
    # Mark pillar region
    pillar_x_min = pillar_x_center - pillar_thickness_nm/2
    pillar_x_max = pillar_x_center + pillar_thickness_nm/2
    ax2.axvline(x=pillar_x_min, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axvline(x=pillar_x_max, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.text(pillar_x_center, size_z_nm*0.35, 'Pillar\\nRegion', ha='center', va='center',
             color='red', fontweight='bold', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # XY plane (top view)
    ax3 = axes[1, 0]
    extent_xy = [-size_x_nm*0.45, size_x_nm*0.45, -size_y_nm*0.45, size_y_nm*0.45]
    im3 = ax3.imshow(n_data_xy.T, extent=extent_xy, cmap='viridis', origin='lower')
    ax3.set_title(f'XY Plane: Top View (z = 0 nm)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3, label='Refractive Index n')
    ax3.grid(True, alpha=0.3)
    
    # Mark pillar region
    ax3.axvline(x=pillar_x_min, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax3.axvline(x=pillar_x_max, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # YZ plane histogram
    ax4 = axes[1, 1]
    n_flat = n_data_yz.flatten()
    ax4.hist(n_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Refractive Index n')
    ax4.set_ylabel('Pixel Count')
    ax4.set_title('YZ Plane Refractive Index Distribution', fontsize=12, fontweight='bold')
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


def calculate_phase_map_from_monitors(back_monitors, size_y_nm, size_z_nm, wavelength_nm):
    """Calculate phase map from back monitors (nm units)
    
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
    phase_analysis : dict
        Phase map analysis results
    """
    print(f"\n📊 Calculating phase map from transmitted field...")
    
    try:
        # Use BackNear monitor
        back_near_data = list(back_monitors.values())[0]
        ez_field = back_near_data['ez']
        ex_field = back_near_data['ex']
        ey_field = back_near_data['ey']
        
        print(f"  • Field size: {ez_field.shape}")
        print(f"  • Monitor size: {size_y_nm:.0f} × {size_z_nm:.0f} nm")
        
        # Calculate phase from Ez field (dominant component)
        phase_map = np.angle(ez_field)  # Phase in radians (-π to π)
        
        # Calculate amplitude
        amplitude_map = np.abs(ez_field)
        
        # Total intensity (all components)
        intensity_map = np.abs(ez_field)**2 + np.abs(ex_field)**2 + np.abs(ey_field)**2
        
        # Statistics
        phase_mean = np.mean(phase_map)
        phase_std = np.std(phase_map)
        phase_min = np.min(phase_map)
        phase_max = np.max(phase_map)
        phase_range = phase_max - phase_min
        
        amplitude_mean = np.mean(amplitude_map)
        amplitude_std = np.std(amplitude_map)
        
        phase_analysis = {
            'phase_map': phase_map,
            'amplitude_map': amplitude_map,
            'intensity_map': intensity_map,
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
        
        print(f"  📐 Phase map statistics:")
        print(f"    • Mean phase: {phase_mean:.4f} rad ({phase_mean/np.pi:.2f}π)")
        print(f"    • Std phase: {phase_std:.4f} rad ({phase_std/np.pi:.2f}π)")
        print(f"    • Phase range: {phase_range:.4f} rad ({phase_range/np.pi:.2f}π)")
        print(f"    • Min phase: {phase_min:.4f} rad")
        print(f"    • Max phase: {phase_max:.4f} rad")
        print(f"  📐 Amplitude statistics:")
        print(f"    • Mean amplitude: {amplitude_mean:.4e}")
        print(f"    • Std amplitude: {amplitude_std:.4e}")
        
        return phase_analysis
        
    except Exception as e:
        print(f"    ⚠️ Phase map calculation failed: {e}")
        return {}


def visualize_phase_map(phase_analysis, mask_info, wavelength_nm=535, pillar_height_nm=600, 
                        delta_n=0.04, n_base=1.5, resolution_nm=0.03, incident_deg=0):
    """Visualize phase map results with parameters in filename"""
    
    if not phase_analysis:
        print(f"⚠️ No phase map data to visualize.")
        return
    
    print(f"\n🎨 Generating phase map visualization...")
    
    phase_map = phase_analysis['phase_map']
    amplitude_map = phase_analysis['amplitude_map']
    intensity_map = phase_analysis['intensity_map']
    size_y_nm = phase_analysis['size_y_nm']
    size_z_nm = phase_analysis['size_z_nm']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    extent = [-size_z_nm*0.4, size_z_nm*0.4, -size_y_nm*0.4, size_y_nm*0.4]
    
    # 1. Phase map
    ax1 = axes[0, 0]
    im1 = ax1.imshow(phase_map, extent=extent, cmap='hsv', origin='lower',
                     vmin=-np.pi, vmax=np.pi)
    ax1.set_title('Phase Map (YZ plane)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('z (nm)')
    ax1.set_ylabel('y (nm)')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Phase (rad)')
    cbar1.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar1.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Amplitude map
    ax2 = axes[0, 1]
    im2 = ax2.imshow(amplitude_map, extent=extent, cmap='viridis', origin='lower')
    ax2.set_title('Amplitude Map |Ez|', fontsize=14, fontweight='bold')
    ax2.set_xlabel('z (nm)')
    ax2.set_ylabel('y (nm)')
    plt.colorbar(im2, ax=ax2, label='Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # 3. Intensity map
    ax3 = axes[0, 2]
    im3 = ax3.imshow(intensity_map, extent=extent, cmap='hot', origin='lower')
    ax3.set_title('Total Intensity', fontsize=14, fontweight='bold')
    ax3.set_xlabel('z (nm)')
    ax3.set_ylabel('y (nm)')
    plt.colorbar(im3, ax=ax3, label='Intensity')
    ax3.grid(True, alpha=0.3)
    
    # 4. Phase histogram
    ax4 = axes[1, 0]
    ax4.hist(phase_map.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax4.set_xlabel('Phase (rad)')
    ax4.set_ylabel('Count')
    ax4.set_title('Phase Distribution', fontsize=14, fontweight='bold')
    ax4.axvline(x=phase_analysis['phase_mean'], color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {phase_analysis['phase_mean']:.2f}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Amplitude histogram
    ax5 = axes[1, 1]
    ax5.hist(amplitude_map.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
    ax5.set_xlabel('Amplitude')
    ax5.set_ylabel('Count')
    ax5.set_title('Amplitude Distribution', fontsize=14, fontweight='bold')
    ax5.axvline(x=phase_analysis['amplitude_mean'], color='red', linestyle='--',
                linewidth=2, label=f"Mean: {phase_analysis['amplitude_mean']:.2e}")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Phase unwrapping (center line profile)
    ax6 = axes[1, 2]
    center_idx = phase_map.shape[0] // 2
    phase_profile = phase_map[center_idx, :]
    z_coords = np.linspace(-size_z_nm*0.4, size_z_nm*0.4, len(phase_profile))
    
    ax6.plot(z_coords, phase_profile, 'b-', linewidth=2, label='Phase profile (y=0)')
    ax6.set_xlabel('z (nm)')
    ax6.set_ylabel('Phase (rad)')
    ax6.set_title('Phase Profile at y=0', fontsize=14, fontweight='bold')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.axhline(y=np.pi, color='r', linestyle='--', alpha=0.5, label='±π')
    ax6.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.5)
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
    
    filename = f"phase_map_analysis_{param_str}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: {filename}")


def run_random_pillar_simulation(mask_file=MASK_FILE, resolution_nm=RESOLUTION_NM, 
                                 pml_nm=PML_NM, size_x_nm=SIZE_X_NM,
                                 pillar_height_nm=PILLAR_HEIGHT_NM, 
                                 pillar_x_center=PILLAR_X_CENTER,
                                 incident_deg=INCIDENT_DEG, wavelength_nm=WAVELENGTH_NM,
                                 n_base=N_BASE, delta_n=DELTA_N,
                                 cell_size_scale=CELL_SIZE_SCALE):
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
    size_z_nm = mask_height_nm * cell_size_scale  # nm (already in nm)
    size_y_nm = mask_width_nm * cell_size_scale   # nm (already in nm)
    
    print(f"\n📐 Cell size from mask:")
    print(f"  • Mask size: {mask.shape} pixels (height × width)")
    print(f"  • 1 pixel = 1 nm")
    print(f"  • Cell size y: {size_y_nm:.0f} nm")
    print(f"  • Cell size z: {size_z_nm:.0f} nm")
    print(f"  • Scale factor: {cell_size_scale}")
    
    # MEEP grid resolution (pixels/nm)
    target_ny = int(size_y_nm * resolution_nm)  # y direction grid points
    target_nz = int(size_z_nm * resolution_nm)  # z direction grid points
    
    print(f"\n📐 MEEP grid size:")
    print(f"  • ny (y direction): {target_ny} points ({size_y_nm:.0f} nm × {resolution_nm} pixels/nm)")
    print(f"  • nz (z direction): {target_nz} points ({size_z_nm:.0f} nm × {resolution_nm} pixels/nm)")
    
    # Resample mask to MEEP grid resolution
    resampled_mask = resample_mask_to_cell_size(mask, target_ny, target_nz)
    
    print(f"\n📋 Simulation parameters (all in nm):")
    print(f"  • Cell size: {size_x_nm:.0f} × {size_y_nm:.0f} × {size_z_nm:.0f} nm")
    print(f"  • Pillar size: {pillar_height_nm:.0f} × {size_y_nm:.0f} × {size_z_nm:.0f} nm")
    print(f"  • Resolution: {resolution_nm} pixels/nm")
    print(f"  • Wavelength: {wavelength_nm} nm")
    print(f"  • Incident angle: {incident_deg}° (normal incidence)")
    print(f"  • Base index: {n_base}")
    print(f"  • Pillar index: {n_base + delta_n}")
    print(f"  • Δn: {delta_n} (HOE standard - realistic)")
    print(f"  • Pattern: Random pillar (non-periodic)")
    
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
    cell_size = mp.Vector3(size_x_nm + 2*pml_nm, size_y_nm, size_z_nm)
    pml_layers = [mp.PML(thickness=pml_nm, direction=mp.X)]
    
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
    
    sources = [
        mp.Source(
            src=mp.ContinuousSource(frequency=frequency),
            component=mp.Ez,
            center=src_center,
            size=src_size,
            amp_func=pw_amp(k_point, src_center)
        )
    ]
    
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
            size=mp.Vector3(0, size_y_nm * 0.8, size_z_nm * 0.8)
        )
        monitor_volumes.append((monitor_vol, name, x_pos, "front"))
        print(f"    • {name}: x = {x_pos:.0f} nm")
    
    print(f"  📤 Back monitors:")
    for x_pos, name in zip(back_monitor_positions, back_monitor_names):
        monitor_vol = mp.Volume(
            center=mp.Vector3(x_pos, 0, 0),
            size=mp.Vector3(0, size_y_nm * 0.8, size_z_nm * 0.8)
        )
        monitor_volumes.append((monitor_vol, name, x_pos, "back"))
        print(f"    • {name}: x = {x_pos:.0f} nm")
    
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
    print(f"\n🚀 Running simulation...")
    print(f"  • Geometry count: {len(geometry)}")
    print(f"  • Monitor count: {len(monitor_volumes)}")
    
    # Calculate required simulation time (nm/c units)
    # Distance from source to farthest monitor
    max_distance = abs(x_src) + max(abs(pos) for pos in all_monitor_positions) + pillar_height_nm/2
    # Time for light to travel through medium (with index n_base)
    travel_time = max_distance * n_base
    # Add extra time for stabilization (a few wavelengths)
    extra_time = wavelength_nm * 5
    total_time = travel_time + extra_time
    
    print(f"  • Max distance: {max_distance:.0f} nm")
    print(f"  • Travel time: {travel_time:.0f} nm/c")
    print(f"  • Total simulation time: {total_time:.0f} nm/c")
    
    sim.run(until=total_time)
    
    print(f"✅ Simulation complete!")
    
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
            'extent': [-size_z_nm*0.4, size_z_nm*0.4, -size_y_nm*0.4, size_y_nm*0.4]
        }
        
        monitor_data[name] = monitor_info
        
        if position_type == "front":
            front_monitors[name] = monitor_info
        elif position_type == "back":
            back_monitors[name] = monitor_info
    
    # Calculate phase map
    phase_analysis = calculate_phase_map_from_monitors(back_monitors, size_y_nm, 
                                                       size_z_nm, wavelength_nm)
    
    # Visualize phase map
    if phase_analysis:
        visualize_phase_map(phase_analysis, mask_info, 
                           wavelength_nm=wavelength_nm,
                           pillar_height_nm=pillar_height_nm,
                           delta_n=delta_n,
                           n_base=n_base,
                           resolution_nm=resolution_nm,
                           incident_deg=incident_deg)
        
        # Save phase map with all parameters in filename
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
        
        filename_base = f"phase_map_{param_str}_{timestamp}"
        
        np.save(f"meep_output/{filename_base}.npy", phase_analysis['phase_map'])
        np.save(f"meep_output/amplitude_map_{param_str}_{timestamp}.npy", phase_analysis['amplitude_map'])
        print(f"\n💾 Phase map saved: meep_output/{filename_base}.npy")
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
        'phase_analysis': phase_analysis,
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


if __name__ == "__main__":
    main()