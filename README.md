# 랜덤 필러 생성 및 MEEP 시뮬레이션

랜덤 필러(random pillar) 구성을 생성하고, 평면파 입사에 따른 위상맵(phase map)을 계산하는 프로그램입니다.

## 알고리즘 설명

이 프로그램은 다음과 같은 랜덤 필러 생성 알고리즘을 구현합니다:

1. 임의의 개수의 기둥(pillars)에 대해 무작위로 위치를 생성합니다.
2. 각 기둥을 하나씩 검사하면서, 다른 기둥과 너무 가까운 거리(기둥 간 edge-to-edge 거리가 5 nm 미만)에 있는지 확인합니다.
3. 조건을 만족하지 않으면 해당 기둥을 제거하고, 새로운 위치를 무작위로 다시 생성합니다.
4. 모든 기둥이 조건을 만족할 때까지 반복합니다.

## 설치 방법

```bash
pip install -r requirements.txt
```

## 사용 방법

### 기본 실행

```bash
python random_pillar_generator.py
```

### 파라미터 조정

`random_pillar_generator.py` 파일을 열고 `main()` 함수 맨 위의 파라미터 섹션을 수정하세요:

```python
# ========================================
# 파라미터 설정 (여기서 수정하세요!)
# ========================================

# 기본 구조 파라미터
PILLAR_RADIUS = 45.0          # 기둥 반지름 (nm)
MIN_EDGE_DISTANCE = 5.0       # 기둥 간 최소 edge-to-edge 거리 (nm)

# 영역 크기
DOMAIN_WIDTH = 4096           # 시뮬레이션 영역 너비 (nm)
DOMAIN_HEIGHT = 4096          # 시뮬레이션 영역 높이 (nm)

# 기둥 밀도 제어
INITIAL_DENSITY = 40.0        # 초기 기둥 밀도 (/μm²)
                              # 높을수록 기둥이 많아짐
                              # 권장값: 10-50 (낮은 밀도), 50-100 (중간 밀도)

# 알고리즘 파라미터
MAX_ITERATIONS = 10000        # 최대 반복 횟수
RANDOM_SEED = 42              # 랜덤 시드 (재현성을 위해)
                              # None으로 설정하면 매번 다른 결과

# 출력 파일 이름 (타임스탬프 자동 추가)
OUTPUT_FILE = f'random_pillar_{timestamp}.png'
```

### 프로그래밍 방식 사용

```python
from random_pillar_generator import RandomPillarGenerator
import numpy as np

# 랜덤 시드 설정 (선택사항)
np.random.seed(42)

# 랜덤 필러 생성기 초기화
generator = RandomPillarGenerator(
    pillar_radius=45.0,        # 기둥 반지름 (nm)
    min_edge_distance=5.0,     # 최소 edge-to-edge 거리 (nm)
    domain_size=(4096, 4096),  # 시뮬레이션 영역 크기 (nm)
    initial_density=30.0,      # 초기 기둥 밀도 (/μm²)
    max_attempts=10000         # 최대 반복 횟수
)

# 랜덤 필러 생성
pillars = generator.generate_pillars()

# 통계 정보 출력
generator.print_statistics()

# 2D 슬라이스 시각화 (이진 마스크)
mask = generator.visualize_slice(save_path='my_pillar_slice.png')

# numpy 배열로 접근
# mask는 0(빈 공간) 또는 1(기둥) 값을 가진 2D 배열
print(f"마스크 크기: {mask.shape}")
print(f"충진율: {np.sum(mask) / mask.size * 100:.2f}%")
```

## 출력 결과

프로그램을 실행하면 다음과 같은 결과를 얻습니다:

1. **콘솔 출력**: 파라미터 설정, 생성 진행 상황 및 통계 정보
2. **이미지 파일**: 2D 슬라이스 이진 마스크 (PNG 형식)
   - 기둥 픽셀: **흰색 (값 1)**
   - 빈 공간 픽셀: **검은색 (값 0)**
   - 예: `random_pillar_slice.png`
3. **numpy 배열 파일**: 이진 마스크 원본 데이터 (.npy 형식)
   - 예: `random_pillar_slice_mask.npy`

## 주요 기능

- ✅ 랜덤 필러 위치 생성 (기둥 개수 제약 없음)
- ✅ 기둥 간 최소 거리 제약 조건 적용
- ✅ 이진 마스크 생성 (흰색: 기둥, 검은색: 빈 공간)
- ✅ 통계 정보 제공 (기둥 밀도, 충진율, 최근접 이웃 거리 등)
- ✅ 모든 기둥 크기 동일
- ✅ 쉬운 파라미터 조정

## 파라미터 설명

| 파라미터 | 설명 | 기본값 | 권장 범위 |
|---------|------|--------|----------|
| `PILLAR_RADIUS` | 기둥의 반지름 (nm) | 45.0 | 10-100 |
| `MIN_EDGE_DISTANCE` | 기둥 간 최소 edge-to-edge 거리 (nm) | 5.0 | 0-20 |
| `DOMAIN_WIDTH` | 시뮬레이션 영역 너비 (nm) | 4096 | 512-8192 |
| `DOMAIN_HEIGHT` | 시뮬레이션 영역 높이 (nm) | 4096 | 512-8192 |
| `INITIAL_DENSITY` | 초기 기둥 밀도 (/μm²) | 40.0 | 10-100 |
| `MAX_ITERATIONS` | 최대 반복 횟수 | 10000 | 1000-50000 |
| `RANDOM_SEED` | 랜덤 시드 (재현성) | 42 | 정수 or None |
| `OUTPUT_FILE` | 출력 파일 이름 | 타임스탬프 포함 | 임의의 파일명 |

### 밀도 조절 가이드

- **낮은 밀도 (10-30 /μm²)**: 충진율 ~10-20%, 기둥 간 간격이 넓음
- **중간 밀도 (30-60 /μm²)**: 충진율 ~20-35%, 적당한 간격
- **높은 밀도 (60-100 /μm²)**: 충진율 ~35-50%, 조밀한 배치

## 예제 출력

```
============================================================
랜덤 필러 생성기 - 파라미터 설정
============================================================
  기둥 반지름: 45.0 nm
  최소 간격: 5.0 nm
  영역 크기: 4096 × 4096 nm²
  초기 밀도: 40.0 /μm²
  랜덤 시드: 42
  출력 파일: random_pillar_slice.png
============================================================

랜덤 필러 생성 시작...
기둥 반지름: 45.0 nm
최소 edge-to-edge 거리: 5.0 nm
시뮬레이션 영역: 4096 x 4096 nm²
기둥 개수 제약: 없음 (임의의 개수)
--------------------------------------------------
Step 1: 임의의 671개 기둥 위치를 무작위로 생성 중...
        초기 위치 생성 완료
Step 2: 기둥 간 거리 검사 및 조정 중...
        반복 횟수: 100, 조정 중...
        반복 횟수: 200, 조정 중...
        반복 횟수: 300, 조정 중...
        반복 횟수: 400, 조정 중...
        반복 횟수: 500, 조정 중...
        모든 기둥이 조건을 만족함!
--------------------------------------------------
[완료] 랜덤 필러 생성 완료!
최종 기둥 개수: 503
총 반복 횟수: 559

============================================================
랜덤 필러 구성 통계
============================================================
기둥 개수                        :        503
기둥 반지름 (nm)                 :      45.00
최소 edge-to-edge 거리 (nm)      :       5.49
평균 최근접 이웃 거리 (nm)       :      38.00
최대 최근접 이웃 거리 (nm)       :     158.38
영역 크기 (nm²)                  : 16777216
기둥 밀도 (/μm²)                 :      29.98
충진율 (%)                       :      19.07
============================================================

이진 마스크 생성 중...
  50/503 기둥 처리 완료
  100/503 기둥 처리 완료
  ...
  전체 503개 기둥 처리 완료

이진 마스크 저장 완료: random_pillar_slice.png
numpy 배열 저장 완료: random_pillar_slice_mask.npy
```

## MEEP 시뮬레이션

생성된 랜덤 필러 패턴에 평면파를 입사시켜 위상맵(phase map)을 계산합니다.

### 특징

- **HOE 시뮬레이션 구조 기반**: 물리적으로 정확한 평면파 입사 구현
- **3D FDTD 시뮬레이션**: MEEP을 사용한 완전한 전자기파 시뮬레이션
- **다중 모니터**: Front/Back 위치에 여러 모니터 배치하여 투과/반사 분석
- **위상맵 분석**: 투과된 전자기장의 위상 분포 계산
- **자동 로그 저장**: 모든 콘솔 출력을 타임스탬프가 포함된 로그 파일로 저장

### 실행 방법

```bash
python meep_phase_simulation.py
```

### 파라미터 조정

`meep_phase_simulation.py` 파일 상단의 파라미터 섹션에서 수정하세요:

```python
# ================== Simulation Parameters ==================
# HOE 시뮬레이션 코드의 물리적 파라미터 + 랜덤 필러 패턴 크기

# Resolution and PML (HOE 코드 표준)
RESOLUTION_UM = 30          # 해상도 (pixels/μm) - HOE 표준값
PML_UM = 1.5               # PML 두께 (μm) - HOE 표준값

# Simulation cell size (μm)
SIZE_X_UM = 20.0           # x 방향 (전파 방향) - HOE 표준값
# SIZE_Y_UM, SIZE_Z_UM은 마스크 크기에서 자동 계산 (1 픽셀 = 1 nm 가정)

# Random pillar structure parameters (nm)
PILLAR_HEIGHT_NM = 600.0   # 기둥(필름) 두께 (nm) = 0.6 μm
PILLAR_X_CENTER = 0.0      # 기둥 중심 x 위치 (nm) - 셀 중앙

# Optical parameters (nm)
WAVELENGTH_NM = 535.0      # 파장 (nm) - 535nm 녹색 레이저
INCIDENT_DEG = 0.0         # 입사각 (도) - 수직 입사

# Material properties (HOE 코드 표준)
N_BASE = 1.5               # 기본 굴절률 (HOE 표준)
DELTA_N = 0.04             # 굴절률 변조 (HOE 표준값 - 현실적)

# Multi-parameter sweep (nm 단위)
PARAMETER_SWEEP = {
    'pillar_height_nm': [600.0],  # 기둥(필름) 두께 (nm)
    'wavelength_nm': [405.0, 532.0, 633.0],  # RGB 파장 (nm)
    'delta_n': [0.04],  # 굴절률 변조
    'incident_deg': [0.0]  # 입사각
}

# Input file
MASK_FILE = 'random_pillar_slice_mask.npy'  # 랜덤 필러 마스크

# Cell size scaling factor (optional, 1.0 = use mask size as-is)
CELL_SIZE_SCALE = 1.0      # 패턴 크기 스케일 조정 (필요시)
```

**주요 특징:**
- ✅ **모든 단위 nm로 통일**: random_pillar_generator와 동일한 단위 사용
- ✅ **물리적 파라미터는 HOE 표준**: 해상도(0.03 pixels/nm), PML(1500 nm), 파장(535 nm), 굴절률 변조(Δn=0.04)
- ✅ **셀 크기는 패턴에 맞춤**: y, z 방향은 실제 마스크 크기에서 자동 계산 (왜곡 방지)
- ✅ **1 픽셀 = 1 nm**: 4096×4096 픽셀 마스크 → 4100×4100 nm 셀 (정수 픽셀로 자동 조정)
- ✅ **필름 두께**: 600 nm (0.6 μm) - 파장 정도의 두께
- ✅ **시뮬레이션 시간 자동 계산**: 광원-모니터 거리 및 굴절률을 고려하여 충분한 시간 자동 설정
- ✅ **정수 픽셀 자동 조정**: MEEP 경고를 방지하기 위해 셀 크기를 자동으로 가장 가까운 정수 픽셀로 조정 (조정량 < 0.1%)

### 출력 결과

#### 자동 생성 파일

1. **로그 파일** (`logs/` 디렉토리):
   - `random_pillar_phase_simulation_YYYYMMDD_HHMMSS.txt`
   - 모든 콘솔 출력 자동 저장

2. **굴절률 분포** (시뮬레이션 검증):
   - `meep_refractive_index_wl535nm_h600nm_dn0.040_nb1.50_res0.030_inc0deg_size4100x4100nm_YYYYMMDD_HHMMSS.png`
   - YZ plane: 실제 MEEP 굴절률 분포 (랜덤 필러 패턴)
   - XZ plane: 측면 뷰 (기둥 영역 표시)
   - XY plane: 상단 뷰
   - 히스토그램: 굴절률 분포 통계

3. **위상맵 분석**:
   - `phase_map_analysis_wl535nm_h600nm_dn0.040_nb1.50_res0.030_inc0deg_size4100x4100nm_YYYYMMDD_HHMMSS.png`
     - Phase map (YZ plane): 투과 전자기장 위상 (-π ~ π)
     - Amplitude map: 전기장 크기 |Ez|
     - Intensity map: 총 강도 (|Ex|² + |Ey|² + |Ez|²)
     - Phase histogram: 위상 분포
     - Amplitude histogram: 진폭 분포
     - Phase profile: y=0에서의 위상 프로파일

4. **전자기장 시각화**:
   - `field_xy_wl535nm_h600nm_dn0.040_nb1.50_res0.030_inc0deg_size4100x4100nm_YYYYMMDD_HHMMSS.png`
   - Ez 필드 분포 (XY plane, z=0)
   - 모니터 위치 및 기둥 영역 표시

5. **numpy 배열** (`meep_output/` 디렉토리):
   - `phase_map_wl535nm_h600nm_dn0.040_nb1.50_res0.030_inc0deg_size4100x4100nm_YYYYMMDD_HHMMSS.npy`
   - `amplitude_map_wl535nm_h600nm_dn0.040_nb1.50_res0.030_inc0deg_size4100x4100nm_YYYYMMDD_HHMMSS.npy`

**파일명 형식 설명:**
- `wl535nm`: 파장 535 nm
- `h600nm`: 필름 두께 600 nm
- `dn0.040`: 굴절률 변조 Δn = 0.04
- `nb1.50`: 기본 굴절률 n_base = 1.5
- `res0.030`: 해상도 0.03 pixels/nm
- `inc0deg`: 입사각 0도
- `size4100x4100nm`: 셀 크기 4100×4100 nm (정수 픽셀로 조정된 값)
- `YYYYMMDD_HHMMSS`: 타임스탬프

**참고**: 원본 마스크는 4096×4096이지만, MEEP 시뮬레이션에서는 해상도(0.03 pixels/nm)와의 호환성을 위해 4100×4100 nm로 자동 조정됩니다 (약 0.1% 차이).

#### 콘솔/로그 출력

```
============================================================
🔬 Random Pillar + Plane Wave + Phase Map Simulation (HOE-based)
============================================================

=== Loading Random Pillar Mask ===
Mask file: random_pillar_slice_mask.npy
Mask size: (4096, 4096) (height × width)
  • Total pixels: 16,777,216
  • Pillar pixels (1): 3,199,854
  • Fill ratio: 19.1%
  • Pattern type: Random pillar (non-periodic)

📐 Cell size from mask:
  • Mask size: (4096, 4096) pixels (height × width)
  • 1 pixel = 1 nm
  • Raw cell size: 4096 × 4096 nm (Y × Z)
  • Adjusted cell size: 4100.00 × 4100.00 nm (Y × Z)
  • Adjustment: 4.00 nm (0.098%)
  • Scale factor: 1.0

📐 MEEP grid size (integer pixels):
  • ny (y direction): 123 points (4100.00 nm × 0.03 pixels/nm = 123)
  • nz (z direction): 123 points (4100.00 nm × 0.03 pixels/nm = 123)

📐 Total cell size (with PML, adjusted for integer pixels):
  • X: 23000.00 nm (690 pixels)
  • Y: 4100.00 nm (123 pixels)
  • Z: 4100.00 nm (123 pixels)

📐 Resampling mask to MEEP grid:
    • Original mask: (4096, 4096) pixels (height × width)
    • Target MEEP grid: (123 × 123) points (z × y)
    • Zoom factors: (z=0.0300, y=0.0300)
    • Fill ratio: 19.1% → 19.0%
    • Resampled shape: (123, 123) (nz × ny)

📋 Simulation parameters (all in nm):
  • Cell size: 20000 × 4100 × 4100 nm (X × Y × Z, adjusted for integer pixels)
  • Pillar size: 600 × 4100 × 4100 nm
  • Resolution: 0.03 pixels/nm
  • Wavelength: 535 nm
  • Incident angle: 0° (normal incidence)
  • Base index: 1.5
  • Pillar index: 1.54 (n_base + Δn)
  • Δn: 0.04 (HOE standard - realistic)
  • Pattern: Random pillar (non-periodic)

=== Generating Random Pillar Geometry (HOE-style, nm units) ===
Mask size: (123, 123) (nz × ny)
Base refractive index: 1.5
Refractive index modulation: Δn = 0.04
Pillar refractive index: 1.54
Pillar thickness: 600 nm
  • Total blocks: 2,874
  • Pillar pixels: 2,874
  • Block size: 600 × 33.3 × 33.3 nm

🚀 Running simulation...
  • Geometry count: 3,674
  • Monitor count: 4
  • Max distance: 8900 nm
  • Travel time: 13350 nm/c
  • Total simulation time: 16025 nm/c

✅ Simulation complete!

📊 Calculating phase map from transmitted field...
  📐 Phase map statistics:
    • Mean phase: -0.0234 rad (-0.01π)
    • Std phase: 1.2345 rad (0.39π)
    • Phase range: 6.2831 rad (2.00π)

🎉 Random pillar phase map simulation complete!
📁 Output files:
  • meep_random_pillar_refractive_index.png
  • random_pillar_phase_map_analysis.png
  • random_pillar_field_xy.png
  • meep_output/phase_map_*.npy
  • meep_output/amplitude_map_*.npy
```

### 워크플로우

1. **랜덤 필러 생성**:
   ```bash
   python random_pillar_generator.py
   ```
   → `random_pillar_YYYYMMDD_HHMMSS.png` 및 `random_pillar_YYYYMMDD_HHMMSS_mask.npy` 생성

2. **MEEP 시뮬레이션 실행**:
   - `meep_phase_simulation.py`의 `MASK_FILE`을 생성된 `.npy` 파일로 설정
   - ```bash
     python meep_phase_simulation.py
     ```
   - 시뮬레이션 진행 중 자동으로 로그 파일 생성 (`logs/` 디렉토리)
   - 결과 파일들이 자동 생성 (현재 디렉토리 및 `meep_output/`)

3. **결과 분석**:
   - `random_pillar_phase_map_analysis.png`: 위상맵 시각화
   - `meep_output/phase_map_*.npy`: 추가 분석용 원본 데이터
   - `logs/*.txt`: 전체 시뮬레이션 로그

### 물리적 특성

#### 굴절률 참고값 (@ 633nm)

| 재료 | 굴절률 |
|------|--------|
| 공기 | 1.0 |
| PMMA | 1.49 |
| SiO2 (석영) | 1.46 |
| 포토레지스트 | 1.5-1.7 |
| TiO2 | 2.5 |
| GaN | 2.3 |
| Si (실리콘) | 3.5 |

#### 시뮬레이션 구조

- **평면파 소스**: Bloch k-vector를 사용한 물리적으로 정확한 평면파
- **경계 조건**: PML (Perfectly Matched Layer) - 무반사 경계
- **모니터 배치**:
  - Front monitors: 기둥 앞쪽 (입사파 측)
  - Back monitors: 기둥 뒤쪽 (투과파 측, 위상맵 계산)
- **전자기장 성분**: Ex, Ey, Ez 모두 기록

### 주의사항

- **메모리**: 고해상도 시뮬레이션은 많은 메모리를 사용합니다
  - `RESOLUTION_UM`을 낮추면 (예: 10) 메모리 사용량과 시간이 줄어듭니다
- **시뮬레이션 시간**: 마스크 크기와 해상도에 따라 수 분~수 시간 소요
- **로그 파일**: `logs/` 디렉토리에 자동 저장되므로 별도 관리 필요

## 라이센스

MIT License

#   H O E 
 
 
