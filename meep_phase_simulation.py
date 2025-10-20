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
# Adjust all major parameters in this section

# Resolution and PML
RESOLUTION_UM = 20          # Resolution (pixels/μm) - lower for faster simulation
PML_UM = 0.5               # PML thickness (μm)

# Simulation cell size (μm)
SIZE_X_UM = 3.0            # x direction (propagation direction)
SIZE_Y_UM_SCALE = 1.0      # y direction scale (will be set from mask)
SIZE_Z_UM_SCALE = 1.0      # z direction scale (will be set from mask)

# Random pillar structure parameters
PILLAR_HEIGHT_UM = 0.2     # Pillar height (μm) = 200nm
PILLAR_X_CENTER = 0.0      # Pillar center position in x direction (μm)

# Optical parameters
WAVELENGTH_UM = 0.633      # Wavelength (μm) - 633nm red laser
INCIDENT_DEG = 0.0         # Incident angle (degrees)

# Material properties
N_BASE = 1.5               # Base refractive index
DELTA_N = 0.5              # Refractive index modulation (pillar vs background)

# Input file
MASK_FILE = 'random_pillar_slice_mask.npy'  # Binary mask from random_pillar_generator.py

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


def create_random_pillar_geometry(mask, cell_size_x, cell_size_y, cell_size_z,
                                  n_base=1.5, delta_n=0.5, thickness_um=0.2, 
                                  pillar_x_center=0.0):
    """Convert random pillar mask to MEEP geometry
    
    Parameters:
    -----------
    mask : 2D numpy array
        Binary pattern (0 or 1), shape (height, width) = (z, y) in MEEP coordinates
    cell_size_x, cell_size_y, cell_size_z : float
        Cell size (μm)
    n_base : float
        Base refractive index (background)
    delta_n : float
        Refractive index modulation (pillar_index = n_base + delta_n)
    thickness_um : float
        Pillar thickness in x direction (μm)
    pillar_x_center : float
        Pillar center position in x direction (μm)
        
    Returns:
    --------
    geometry : list
        MEEP geometry objects
    background_material : mp.Medium
        Background material
    """
    print(f"\n=== Generating Random Pillar Geometry ===")
    print(f"Mask size: {mask.shape} (height × width)")
    print(f"Base refractive index: {n_base}")
    print(f"Pillar refractive index: {n_base + delta_n} (n_base + Δn)")
    print(f"Background refractive index: {n_base}")
    print(f"Pillar thickness (x): {thickness_um} μm")
    print(f"Pillar x center: {pillar_x_center} μm")
    
    # Materials
    background_material = mp.Medium(index=n_base)
    pillar_material = mp.Medium(index=n_base + delta_n)
    
    # Generate geometry from mask
    geometry = []
    
    ny, nx = mask.shape  # (height, width) = (z, y) in MEEP
    z_coords = np.linspace(-cell_size_z/2, cell_size_z/2, ny)  # z coordinates
    y_coords = np.linspace(-cell_size_y/2, cell_size_y/2, nx)  # y coordinates
    
    print(f"Coordinates:")
    print(f"  • y range: {y_coords[0]:.2f} to {y_coords[-1]:.2f} μm ({nx} points)")
    print(f"  • z range: {z_coords[0]:.2f} to {z_coords[-1]:.2f} μm ({ny} points)")
    
    # Create blocks for each pillar pixel
    pixel_size_y = cell_size_y / nx
    pixel_size_z = cell_size_z / ny
    
    pillar_count = 0
    for i in range(ny):
        for j in range(nx):
            if mask[i, j] == 1:  # Pillar pixel
                y_pos = y_coords[j]
                z_pos = z_coords[i]
                
                block = mp.Block(
                    size=mp.Vector3(thickness_um, pixel_size_y, pixel_size_z),
                    center=mp.Vector3(pillar_x_center, y_pos, z_pos),
                    material=pillar_material
                )
                geometry.append(block)
                pillar_count += 1
    
    print(f"Generated geometry:")
    print(f"  • Total blocks: {len(geometry):,}")
    print(f"  • Pillar pixels: {pillar_count:,}")
    print(f"  • Block size: {thickness_um} × {pixel_size_y:.4f} × {pixel_size_z:.4f} μm")
    
    return geometry, background_material


def visualize_actual_meep_pattern(sim, size_x_um, size_y_um, size_z_um, 
                                  pillar_x_center, pillar_thickness_um):
    """Visualize actual refractive index distribution used in MEEP simulation"""
    
    print(f"\n  📐 Extracting actual MEEP refractive index distribution...")
    
    # YZ plane (pillar pattern at x=pillar_x_center)
    yz_plane_center = mp.Vector3(pillar_x_center, 0, 0)
    yz_plane_size = mp.Vector3(0, size_y_um * 0.9, size_z_um * 0.9)
    
    eps_data_yz = sim.get_array(center=yz_plane_center, size=yz_plane_size, 
                                component=mp.Dielectric)
    n_data_yz = np.sqrt(np.real(eps_data_yz))
    
    # XZ plane (side view at y=0)
    xz_plane_center = mp.Vector3(0, 0, 0)
    xz_plane_size = mp.Vector3(size_x_um * 0.9, 0, size_z_um * 0.9)
    
    eps_data_xz = sim.get_array(center=xz_plane_center, size=xz_plane_size, 
                                component=mp.Dielectric)
    n_data_xz = np.sqrt(np.real(eps_data_xz))
    
    # XY plane (top view at z=0)
    xy_plane_center = mp.Vector3(0, 0, 0)
    xy_plane_size = mp.Vector3(size_x_um * 0.9, size_y_um * 0.9, 0)
    
    eps_data_xy = sim.get_array(center=xy_plane_center, size=xy_plane_size, 
                                component=mp.Dielectric)
    n_data_xy = np.sqrt(np.real(eps_data_xy))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # YZ plane (actual pillar pattern)
    ax1 = axes[0, 0]
    extent_yz = [-size_z_um*0.45, size_z_um*0.45, -size_y_um*0.45, size_y_um*0.45]
    im1 = ax1.imshow(n_data_yz, extent=extent_yz, cmap='viridis', origin='lower')
    ax1.set_title(f'YZ Plane: Actual MEEP Refractive Index\\n(x = {pillar_x_center} μm, Random Pillar Pattern)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('z (μm)')
    ax1.set_ylabel('y (μm)')
    plt.colorbar(im1, ax=ax1, label='Refractive Index n')
    ax1.grid(True, alpha=0.3)
    
    # Statistics
    n_min, n_max = np.min(n_data_yz), np.max(n_data_yz)
    ax1.text(0.05, 0.95, f'n_min: {n_min:.3f}\\nn_max: {n_max:.3f}\\nΔn: {n_max-n_min:.3f}', 
             transform=ax1.transAxes, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # XZ plane (side view)
    ax2 = axes[0, 1]
    extent_xz = [-size_x_um*0.45, size_x_um*0.45, -size_z_um*0.45, size_z_um*0.45]
    im2 = ax2.imshow(n_data_xz.T, extent=extent_xz, cmap='viridis', origin='lower')
    ax2.set_title(f'XZ Plane: Side View (y = 0 μm)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x (μm)')
    ax2.set_ylabel('z (μm)')
    plt.colorbar(im2, ax=ax2, label='Refractive Index n')
    ax2.grid(True, alpha=0.3)
    
    # Mark pillar region
    pillar_x_min = pillar_x_center - pillar_thickness_um/2
    pillar_x_max = pillar_x_center + pillar_thickness_um/2
    ax2.axvline(x=pillar_x_min, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axvline(x=pillar_x_max, color='red', linestyle='--', alpha=0.8, linewidth=2)
    ax2.text(pillar_x_center, size_z_um*0.35, 'Pillar\\nRegion', ha='center', va='center',
             color='red', fontweight='bold', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # XY plane (top view)
    ax3 = axes[1, 0]
    extent_xy = [-size_x_um*0.45, size_x_um*0.45, -size_y_um*0.45, size_y_um*0.45]
    im3 = ax3.imshow(n_data_xy.T, extent=extent_xy, cmap='viridis', origin='lower')
    ax3.set_title(f'XY Plane: Top View (z = 0 μm)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x (μm)')
    ax3.set_ylabel('y (μm)')
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
    plt.savefig("meep_random_pillar_refractive_index.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 MEEP Refractive Index Analysis:")
    print(f"    • Min: {np.min(n_data_yz):.4f}")
    print(f"    • Max: {np.max(n_data_yz):.4f}")
    print(f"    • Mean: {np.mean(n_data_yz):.4f} ± {np.std(n_data_yz):.4f}")
    print(f"    • Δn: {np.max(n_data_yz) - np.min(n_data_yz):.4f}")


def calculate_phase_map_from_monitors(back_monitors, size_y_um, size_z_um, wavelength_um):
    """Calculate phase map from back monitors
    
    Parameters:
    -----------
    back_monitors : dict
        Back monitor data with Ez, Ex, Ey fields
    size_y_um, size_z_um : float
        Monitor size
    wavelength_um : float
        Wavelength
        
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
        print(f"  • Monitor size: {size_y_um:.2f} × {size_z_um:.2f} μm")
        
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
            'wavelength_um': wavelength_um,
            'size_y_um': size_y_um,
            'size_z_um': size_z_um
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


def visualize_phase_map(phase_analysis, mask_info):
    """Visualize phase map results"""
    
    if not phase_analysis:
        print(f"⚠️ No phase map data to visualize.")
        return
    
    print(f"\n🎨 Generating phase map visualization...")
    
    phase_map = phase_analysis['phase_map']
    amplitude_map = phase_analysis['amplitude_map']
    intensity_map = phase_analysis['intensity_map']
    size_y_um = phase_analysis['size_y_um']
    size_z_um = phase_analysis['size_z_um']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    extent = [-size_z_um*0.4, size_z_um*0.4, -size_y_um*0.4, size_y_um*0.4]
    
    # 1. Phase map
    ax1 = axes[0, 0]
    im1 = ax1.imshow(phase_map, extent=extent, cmap='hsv', origin='lower',
                     vmin=-np.pi, vmax=np.pi)
    ax1.set_title('Phase Map (YZ plane)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('z (μm)')
    ax1.set_ylabel('y (μm)')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Phase (rad)')
    cbar1.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar1.set_ticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
    ax1.grid(True, alpha=0.3)
    
    # 2. Amplitude map
    ax2 = axes[0, 1]
    im2 = ax2.imshow(amplitude_map, extent=extent, cmap='viridis', origin='lower')
    ax2.set_title('Amplitude Map |Ez|', fontsize=14, fontweight='bold')
    ax2.set_xlabel('z (μm)')
    ax2.set_ylabel('y (μm)')
    plt.colorbar(im2, ax=ax2, label='Amplitude')
    ax2.grid(True, alpha=0.3)
    
    # 3. Intensity map
    ax3 = axes[0, 2]
    im3 = ax3.imshow(intensity_map, extent=extent, cmap='hot', origin='lower')
    ax3.set_title('Total Intensity', fontsize=14, fontweight='bold')
    ax3.set_xlabel('z (μm)')
    ax3.set_ylabel('y (μm)')
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
    z_coords = np.linspace(-size_z_um*0.4, size_z_um*0.4, len(phase_profile))
    
    ax6.plot(z_coords, phase_profile, 'b-', linewidth=2, label='Phase profile (y=0)')
    ax6.set_xlabel('z (μm)')
    ax6.set_ylabel('Phase (rad)')
    ax6.set_title('Phase Profile at y=0', fontsize=14, fontweight='bold')
    ax6.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax6.axhline(y=np.pi, color='r', linestyle='--', alpha=0.5, label='±π')
    ax6.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.5)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("random_pillar_phase_map_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  💾 Saved: random_pillar_phase_map_analysis.png")


def run_random_pillar_simulation(mask_file=MASK_FILE, resolution_um=RESOLUTION_UM, 
                                 pml_um=PML_UM, size_x_um=SIZE_X_UM,
                                 pillar_height_um=PILLAR_HEIGHT_UM, 
                                 pillar_x_center=PILLAR_X_CENTER,
                                 incident_deg=INCIDENT_DEG, wavelength_um=WAVELENGTH_UM,
                                 n_base=N_BASE, delta_n=DELTA_N):
    """Run random pillar + plane wave simulation and calculate phase map"""
    
    print("=" * 80)
    print("🔬 Random Pillar + Plane Wave + Phase Map Simulation")
    print("=" * 80)
    
    # Load mask
    mask, mask_info = load_random_pillar_mask(mask_file)
    
    # Determine cell size from mask
    # Assume 1 pixel = 1 nm, convert to μm
    mask_height_nm, mask_width_nm = mask.shape
    size_z_um = mask_height_nm / 1000.0 * SIZE_Z_UM_SCALE  # Convert nm to μm
    size_y_um = mask_width_nm / 1000.0 * SIZE_Y_UM_SCALE
    
    print(f"\n📋 Simulation parameters:")
    print(f"  • Cell size: {size_x_um} × {size_y_um:.2f} × {size_z_um:.2f} μm")
    print(f"  • Pillar size: {pillar_height_um} × {size_y_um:.2f} × {size_z_um:.2f} μm")
    print(f"  • Resolution: {resolution_um} pixels/μm")
    print(f"  • Wavelength: {wavelength_um} μm ({wavelength_um*1000:.0f} nm)")
    print(f"  • Incident angle: {incident_deg}° (normal incidence)")
    print(f"  • Base index: {n_base}")
    print(f"  • Pillar index: {n_base + delta_n}")
    print(f"  • Δn: {delta_n}")
    
    # Physical parameters
    incident_angle = math.radians(incident_deg)
    frequency = 1.0 / wavelength_um
    
    # Create geometry
    geometry, default_material = create_random_pillar_geometry(
        mask, size_x_um, size_y_um, size_z_um,
        n_base=n_base, delta_n=delta_n, 
        thickness_um=pillar_height_um,
        pillar_x_center=pillar_x_center
    )
    
    # Cell and boundary
    cell_size = mp.Vector3(size_x_um + 2*pml_um, size_y_um, size_z_um)
    pml_layers = [mp.PML(thickness=pml_um, direction=mp.X)]
    
    # k-vector for plane wave
    k_vec = mp.Vector3(n_base*frequency, 0, 0)
    k_point = k_vec
    
    print(f"\n🌊 Plane wave setup:")
    print(f"  • k-vector: ({k_vec.x:.3f}, {k_vec.y:.3f}, {k_vec.z:.3f})")
    print(f"  • Frequency: {frequency:.3f}")
    
    # Plane wave source
    x_src = -0.4*size_x_um
    src_center = mp.Vector3(x_src, 0, 0)
    src_size = mp.Vector3(0, size_y_um, size_z_um)
    
    sources = [
        mp.Source(
            src=mp.ContinuousSource(frequency=frequency),
            component=mp.Ez,
            center=src_center,
            size=src_size,
            amp_func=pw_amp(k_point, src_center)
        )
    ]
    
    print(f"  • Source position: x = {x_src} μm")
    print(f"  • Source size: {src_size.y} × {src_size.z} μm")
    
    # Monitors (front and back)
    print(f"\n📡 Setting up monitors...")
    
    front_monitor_positions = [
        pillar_x_center - pillar_height_um/2 - 0.3,
        pillar_x_center - pillar_height_um/2 - 0.1
    ]
    front_monitor_names = ["FrontFar", "FrontNear"]
    
    back_monitor_positions = [
        pillar_x_center + pillar_height_um/2 + 0.1,
        pillar_x_center + pillar_height_um/2 + 0.3
    ]
    back_monitor_names = ["BackNear", "BackFar"]
    
    all_monitor_positions = front_monitor_positions + back_monitor_positions
    all_monitor_names = front_monitor_names + back_monitor_names
    monitor_volumes = []
    
    print(f"  📥 Front monitors:")
    for x_pos, name in zip(front_monitor_positions, front_monitor_names):
        monitor_vol = mp.Volume(
            center=mp.Vector3(x_pos, 0, 0),
            size=mp.Vector3(0, size_y_um * 0.8, size_z_um * 0.8)
        )
        monitor_volumes.append((monitor_vol, name, x_pos, "front"))
        print(f"    • {name}: x = {x_pos:.2f} μm")
    
    print(f"  📤 Back monitors:")
    for x_pos, name in zip(back_monitor_positions, back_monitor_names):
        monitor_vol = mp.Volume(
            center=mp.Vector3(x_pos, 0, 0),
            size=mp.Vector3(0, size_y_um * 0.8, size_z_um * 0.8)
        )
        monitor_volumes.append((monitor_vol, name, x_pos, "back"))
        print(f"    • {name}: x = {x_pos:.2f} μm")
    
    # Create simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution_um,
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
    
    sim.run(until=30.0)
    
    print(f"✅ Simulation complete!")
    
    # Visualize refractive index
    visualize_actual_meep_pattern(sim, size_x_um, size_y_um, size_z_um, 
                                  pillar_x_center, pillar_height_um)
    
    # Collect monitor data
    print(f"\n📊 Collecting monitor data...")
    
    monitor_data = {}
    front_monitors = {}
    back_monitors = {}
    
    for monitor_vol, name, x_pos, position_type in monitor_volumes:
        print(f"  • {name} monitor (x = {x_pos:.2f} μm)...")
        
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
            'extent': [-size_z_um*0.4, size_z_um*0.4, -size_y_um*0.4, size_y_um*0.4]
        }
        
        monitor_data[name] = monitor_info
        
        if position_type == "front":
            front_monitors[name] = monitor_info
        elif position_type == "back":
            back_monitors[name] = monitor_info
    
    # Calculate phase map
    phase_analysis = calculate_phase_map_from_monitors(back_monitors, size_y_um, 
                                                       size_z_um, wavelength_um)
    
    # Visualize phase map
    if phase_analysis:
        visualize_phase_map(phase_analysis, mask_info)
        
        # Save phase map
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("meep_output", exist_ok=True)
        np.save(f"meep_output/phase_map_{timestamp}.npy", phase_analysis['phase_map'])
        np.save(f"meep_output/amplitude_map_{timestamp}.npy", phase_analysis['amplitude_map'])
        print(f"\n💾 Phase map saved: meep_output/phase_map_{timestamp}.npy")
    
    # Basic visualization
    print(f"\n🎨 Generating field visualizations...")
    
    # XY plane
    fig, ax = plt.subplots(figsize=(14, 6))
    output_plane_xy = mp.Volume(center=mp.Vector3(0, 0, 0),
                                size=mp.Vector3(size_x_um, size_y_um, 0))
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
    
    plt.savefig("random_pillar_field_xy.png", bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"\n🎉 Random pillar phase map simulation complete!")
    print(f"📁 Output files:")
    print(f"  • meep_random_pillar_refractive_index.png")
    print(f"  • random_pillar_phase_map_analysis.png")
    print(f"  • random_pillar_field_xy.png")
    print(f"  • meep_output/phase_map_*.npy")
    print(f"  • meep_output/amplitude_map_*.npy")
    
    return {
        'all_monitors': monitor_data,
        'front_monitors': front_monitors,
        'back_monitors': back_monitors,
        'phase_analysis': phase_analysis,
        'mask_info': mask_info,
        'simulation_params': {
            'wavelength_um': wavelength_um,
            'pillar_height_um': pillar_height_um,
            'n_base': n_base,
            'delta_n': delta_n,
            'incident_deg': incident_deg
        }
    }


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("Random Pillar Phase Map Simulation")
    print("=" * 80)
    
    print(f"\nSimulation settings:")
    print(f"  • Mask file: {MASK_FILE}")
    print(f"  • Wavelength: {WAVELENGTH_UM} μm ({WAVELENGTH_UM*1000:.0f} nm)")
    print(f"  • Pillar height: {PILLAR_HEIGHT_UM} μm ({PILLAR_HEIGHT_UM*1000:.0f} nm)")
    print(f"  • Pillar index: {N_BASE + DELTA_N} (n_base + Δn)")
    print(f"  • Background index: {N_BASE}")
    print(f"  • Δn: {DELTA_N}")
    print(f"  • Resolution: {RESOLUTION_UM} pixels/μm")
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