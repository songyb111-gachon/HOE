"""
MEEP 시뮬레이션: 랜덤 필러 패턴에 평면파 입사 및 위상맵 계산
"""

import numpy as np
import matplotlib.pyplot as plt
import meep as mp
from datetime import datetime
import os

class PillarPhaseSimulation:
    """랜덤 필러 구조에서의 위상맵 시뮬레이션"""
    
    def __init__(self, mask_file):
        """
        Parameters:
        -----------
        mask_file : str
            이진 마스크 numpy 파일 경로 (.npy)
        """
        self.mask_file = mask_file
        self.mask = None
        self.phase_map = None
        
    def load_mask(self):
        """이진 마스크 로드"""
        self.mask = np.load(self.mask_file)
        print(f"마스크 로드 완료: {self.mask.shape}")
        print(f"기둥 픽셀 수: {np.sum(self.mask)}")
        print(f"충진율: {np.sum(self.mask) / self.mask.size * 100:.2f}%")
        return self.mask
    
    def setup_simulation(self, 
                        wavelength=633,      # 입사파 파장 (nm)
                        pillar_height=200,   # 기둥 높이 (nm)
                        pillar_index=2.0,    # 기둥 굴절률 (예: SiO2 ~1.46, Si ~3.5)
                        background_index=1.0, # 배경 굴절률 (공기=1.0)
                        resolution=10,       # 해상도 (픽셀/nm)
                        pml_thickness=100):  # PML 두께 (nm)
        """
        MEEP 시뮬레이션 설정
        
        Parameters:
        -----------
        wavelength : float
            입사 평면파의 파장 (nm)
        pillar_height : float
            기둥의 높이 (nm)
        pillar_index : float
            기둥의 굴절률
        background_index : float
            배경(공기)의 굴절률
        resolution : int
            공간 해상도 (픽셀당 nm)
        pml_thickness : float
            PML(완전정합층) 두께 (nm)
        """
        if self.mask is None:
            raise ValueError("먼저 load_mask()를 실행하세요.")
        
        self.wavelength = wavelength
        self.frequency = 1 / wavelength  # MEEP에서는 주파수 단위 사용
        self.pillar_height = pillar_height
        self.pillar_index = pillar_index
        self.background_index = background_index
        self.resolution = resolution
        self.pml_thickness = pml_thickness
        
        # 시뮬레이션 영역 크기
        sx, sy = self.mask.shape[1], self.mask.shape[0]  # x, y 크기 (nm)
        sz = pillar_height + 2 * pml_thickness + wavelength * 2  # z 방향 (높이)
        
        self.cell_size = mp.Vector3(sx, sy, sz)
        
        print("\n" + "=" * 60)
        print("MEEP 시뮬레이션 설정")
        print("=" * 60)
        print(f"  파장: {wavelength} nm")
        print(f"  주파수: {self.frequency:.6f}")
        print(f"  기둥 높이: {pillar_height} nm")
        print(f"  기둥 굴절률: {pillar_index}")
        print(f"  배경 굴절률: {background_index}")
        print(f"  해상도: {resolution} 픽셀/nm")
        print(f"  셀 크기: {sx} × {sy} × {sz} nm³")
        print(f"  PML 두께: {pml_thickness} nm")
        print("=" * 60 + "\n")
        
        return self
    
    def create_geometry(self):
        """기둥 geometry 생성"""
        geometry = []
        
        # 마스크를 기반으로 기둥 생성
        sy, sx = self.mask.shape
        
        # 각 픽셀을 검사하여 기둥이 있는 위치에 원기둥 추가
        # (간단한 구현 - 모든 픽셀을 개별 블록으로)
        print("Geometry 생성 중...")
        
        # 효율적인 방법: 기둥 영역을 단일 커스텀 material로
        pillar_material = mp.Medium(index=self.pillar_index)
        background_material = mp.Medium(index=self.background_index)
        
        # Custom material function
        def material_function(pos):
            x, y, z = pos.x, pos.y, pos.z
            
            # 픽셀 인덱스로 변환 (중심 기준)
            ix = int(x + sx/2)
            iy = int(y + sy/2)
            
            # 범위 체크
            if 0 <= ix < sx and 0 <= iy < sy:
                # z 범위 체크 (기둥 높이 내)
                if abs(z) <= self.pillar_height / 2:
                    if self.mask[iy, ix] == 1:
                        return pillar_material
            
            return background_material
        
        # 전체 영역을 커버하는 큰 블록 생성
        geometry.append(
            mp.Block(
                size=mp.Vector3(sx, sy, self.pillar_height),
                center=mp.Vector3(0, 0, 0),
                material=material_function
            )
        )
        
        print(f"Geometry 생성 완료")
        return geometry
    
    def run_simulation(self, output_dir='output'):
        """시뮬레이션 실행"""
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Geometry 생성
        geometry = self.create_geometry()
        
        # PML 경계 조건
        pml_layers = [mp.PML(thickness=self.pml_thickness)]
        
        # 평면파 소스 (z 방향에서 입사)
        sources = [
            mp.Source(
                mp.ContinuousSource(frequency=self.frequency),
                component=mp.Ex,  # x-편광
                center=mp.Vector3(0, 0, -self.cell_size.z/2 + self.pml_thickness),
                size=mp.Vector3(self.cell_size.x, self.cell_size.y, 0)
            )
        ]
        
        # 시뮬레이션 생성
        sim = mp.Simulation(
            cell_size=self.cell_size,
            geometry=geometry,
            sources=sources,
            boundary_layers=pml_layers,
            resolution=self.resolution
        )
        
        # 위상 모니터 (기둥 통과 후)
        monitor_z = self.pillar_height / 2 + 10  # 기둥 위 10nm
        
        print("시뮬레이션 실행 중...")
        print(f"  모니터 위치: z = {monitor_z} nm")
        
        # 시뮬레이션 실행
        sim.run(until=100)  # 100 시간 단위 실행
        
        # 전기장 추출
        ex = sim.get_array(
            center=mp.Vector3(0, 0, monitor_z),
            size=mp.Vector3(self.cell_size.x, self.cell_size.y, 0),
            component=mp.Ex
        )
        
        ey = sim.get_array(
            center=mp.Vector3(0, 0, monitor_z),
            size=mp.Vector3(self.cell_size.x, self.cell_size.y, 0),
            component=mp.Ey
        )
        
        # 위상 계산
        phase_x = np.angle(ex)
        phase_y = np.angle(ey)
        
        # 전체 위상 (진폭 가중 평균)
        amplitude = np.abs(ex) + np.abs(ey)
        self.phase_map = np.where(amplitude > 0, 
                                  (phase_x * np.abs(ex) + phase_y * np.abs(ey)) / amplitude,
                                  0)
        
        print("시뮬레이션 완료!")
        
        # 결과 저장
        self.save_results(output_dir, timestamp)
        
        return self.phase_map
    
    def save_results(self, output_dir, timestamp):
        """결과 저장"""
        
        # 위상맵 numpy 저장
        phase_file = os.path.join(output_dir, f'phase_map_{timestamp}.npy')
        np.save(phase_file, self.phase_map)
        print(f"\n위상맵 저장: {phase_file}")
        
        # 위상맵 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 원본 마스크
        axes[0].imshow(self.mask, cmap='gray', origin='lower')
        axes[0].set_title('Input: Pillar Mask', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('X (nm)')
        axes[0].set_ylabel('Y (nm)')
        axes[0].axis('equal')
        
        # 2. 위상맵
        im1 = axes[1].imshow(self.phase_map, cmap='hsv', origin='lower', 
                            vmin=-np.pi, vmax=np.pi)
        axes[1].set_title('Output: Phase Map', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('X (nm)')
        axes[1].set_ylabel('Y (nm)')
        axes[1].axis('equal')
        plt.colorbar(im1, ax=axes[1], label='Phase (rad)')
        
        # 3. 위상 히스토그램
        axes[2].hist(self.phase_map.flatten(), bins=50, edgecolor='black')
        axes[2].set_title('Phase Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Phase (rad)')
        axes[2].set_ylabel('Count')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 이미지 저장
        image_file = os.path.join(output_dir, f'phase_map_{timestamp}.png')
        plt.savefig(image_file, dpi=300, bbox_inches='tight')
        print(f"위상맵 이미지 저장: {image_file}")
        plt.close()
        
        # 통계 정보
        print("\n" + "=" * 60)
        print("위상맵 통계")
        print("=" * 60)
        print(f"  최소 위상: {np.min(self.phase_map):.4f} rad")
        print(f"  최대 위상: {np.max(self.phase_map):.4f} rad")
        print(f"  평균 위상: {np.mean(self.phase_map):.4f} rad")
        print(f"  표준편차: {np.std(self.phase_map):.4f} rad")
        print(f"  위상 범위: {np.ptp(self.phase_map):.4f} rad ({np.ptp(self.phase_map)/np.pi:.2f}π)")
        print("=" * 60)


def main():
    """메인 실행 함수"""
    
    # ========================================
    # 파라미터 설정 (여기서 수정하세요!)
    # ========================================
    
    # 입력 파일
    MASK_FILE = 'random_pillar_slice_mask.npy'
    # 랜덤 필러 생성기에서 생성된 .npy 파일
    
    # 광학 파라미터
    WAVELENGTH = 633         # 입사파 파장 (nm), 예: 633 (HeNe 레이저)
    PILLAR_HEIGHT = 200      # 기둥 높이 (nm)
    PILLAR_INDEX = 2.0       # 기둥 굴절률 (예: SiO2=1.46, Si=3.5, TiO2=2.5)
    BACKGROUND_INDEX = 1.0   # 배경 굴절률 (공기=1.0)
    
    # 시뮬레이션 파라미터
    RESOLUTION = 10          # 공간 해상도 (픽셀/nm) - 높을수록 정확하지만 느림
    PML_THICKNESS = 100      # PML 두께 (nm)
    
    # 출력 디렉토리
    OUTPUT_DIR = 'meep_output'
    
    # ========================================
    
    print("\n" + "=" * 60)
    print("MEEP 위상맵 시뮬레이션")
    print("=" * 60)
    print(f"  입력 파일: {MASK_FILE}")
    print(f"  파장: {WAVELENGTH} nm")
    print(f"  기둥 높이: {PILLAR_HEIGHT} nm")
    print(f"  기둥 굴절률: {PILLAR_INDEX}")
    print(f"  출력 디렉토리: {OUTPUT_DIR}")
    print("=" * 60 + "\n")
    
    # 시뮬레이션 생성
    sim = PillarPhaseSimulation(MASK_FILE)
    
    # 마스크 로드
    sim.load_mask()
    
    # 시뮬레이션 설정
    sim.setup_simulation(
        wavelength=WAVELENGTH,
        pillar_height=PILLAR_HEIGHT,
        pillar_index=PILLAR_INDEX,
        background_index=BACKGROUND_INDEX,
        resolution=RESOLUTION,
        pml_thickness=PML_THICKNESS
    )
    
    # 시뮬레이션 실행
    phase_map = sim.run_simulation(output_dir=OUTPUT_DIR)
    
    print("\n모든 작업 완료!")
    print(f"결과는 '{OUTPUT_DIR}' 디렉토리에 저장되었습니다.")


if __name__ == "__main__":
    main()

