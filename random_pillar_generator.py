import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches
import warnings
import logging
from datetime import datetime

# matplotlib 폰트 경고 메시지 억제
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

class RandomPillarGenerator:
    """랜덤 필러 생성 알고리즘"""
    
    def __init__(self, 
                 pillar_radius=45.0,  # nm
                 min_edge_distance=5.0,  # nm
                 domain_size=(10000, 10000),  # nm (10 μm × 10 μm)
                 initial_density=29.5,  # pillars per μm²
                 max_attempts=10000):
        """
        Parameters:
        -----------
        pillar_radius : float
            기둥의 반지름 (nm)
        min_edge_distance : float
            기둥 간 최소 edge-to-edge 거리 (nm)
        domain_size : tuple
            시뮬레이션 영역 크기 (width, height) in nm
        initial_density : float
            초기 기둥 밀도 (/μm²) - 이 값을 기준으로 초기 기둥 개수 결정
        max_attempts : int
            새로운 기둥 추가 시도 최대 횟수
        """
        self.pillar_radius = pillar_radius
        self.min_edge_distance = min_edge_distance
        self.domain_size = domain_size
        self.initial_density = initial_density
        self.max_attempts = max_attempts
        self.pillars = []
        
    def calculate_edge_to_edge_distance(self, pos1, pos2):
        """두 기둥 간의 edge-to-edge 거리 계산"""
        center_distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        edge_to_edge = center_distance - 2 * self.pillar_radius
        return edge_to_edge
    
    def is_valid_position(self, new_pos):
        """새로운 위치가 유효한지 확인 (다른 기둥과의 거리 체크)"""
        for existing_pos in self.pillars:
            edge_distance = self.calculate_edge_to_edge_distance(new_pos, existing_pos)
            if edge_distance < self.min_edge_distance:
                return False
        return True
    
    def generate_random_position(self):
        """영역 내에서 무작위 위치 생성 (어떠한 제약도 없음)"""
        # 기둥의 중심이 영역 전체에 걸쳐 완전히 무작위로 배치
        # 기둥이 부분적으로 영역 밖으로 나갈 수 있음
        x = np.random.uniform(0, self.domain_size[0])
        y = np.random.uniform(0, self.domain_size[1])
        return (x, y)
    
    def generate_pillars(self):
        """랜덤 필러 생성 알고리즘 실행
        
        알고리즘:
        1. 임의의 개수의 기둥에 대해 무작위로 위치를 생성
        2. 각 기둥을 하나씩 검사하면서 거리 조건 위반 시 제거하고 새 위치 재생성
        3. 모든 기둥이 조건을 만족할 때까지 반복
        """
        print(f"랜덤 필러 생성 시작...")
        print(f"기둥 반지름: {self.pillar_radius} nm")
        print(f"최소 edge-to-edge 거리: {self.min_edge_distance} nm")
        print(f"시뮬레이션 영역: {self.domain_size[0]} x {self.domain_size[1]} nm²")
        print(f"기둥 개수 제약: 없음 (임의의 개수)")
        print("-" * 50)
        
        # Step 1: 임의의 개수의 기둥 위치를 무작위로 생성
        area_um2 = (self.domain_size[0] * self.domain_size[1]) / 1e6  # nm² to μm²
        initial_num = int(self.initial_density * area_um2)
        
        print(f"Step 1: 임의의 {initial_num}개 기둥 위치를 무작위로 생성 중...")
        self.pillars = [self.generate_random_position() for _ in range(initial_num)]
        print(f"        초기 위치 생성 완료")
        
        # Step 2-3: 각 기둥을 검사하며 거리 조건 위반 시 제거 및 재생성 반복 (벡터화 최적화)
        print(f"Step 2: 기둥 간 거리 검사 및 조정 중 (최적화)...")
        iteration_count = 0
        
        # NumPy 배열로 변환
        pillars_array = np.array(self.pillars)
        
        while iteration_count < self.max_attempts:
            # 모든 pillar 쌍의 거리를 한번에 계산 (벡터화)
            # pillars_array shape: (N, 2)
            # 각 pillar와 다른 모든 pillar 사이의 center distance 계산
            diff = pillars_array[:, np.newaxis, :] - pillars_array[np.newaxis, :, :]  # (N, N, 2)
            center_distances = np.sqrt(np.sum(diff**2, axis=2))  # (N, N)ㅃㅃ
            edge_distances = center_distances - 2 * self.pillar_radius
            
            # 자기 자신과의 거리는 무시 (대각선 = inf)
            np.fill_diagonal(edge_distances, np.inf)
            
            # 각 pillar의 최소 거리 찾기
            min_distances = np.min(edge_distances, axis=1)  # (N,)
            
            # 조건 위반하는 pillar 찾기
            violating_indices = np.where(min_distances < self.min_edge_distance)[0]
            
            if len(violating_indices) == 0:
                # 모든 기둥이 조건을 만족
                print(f"        모든 기둥이 조건을 만족함!")
                break
            
            # 위반하는 pillar 중 하나만 교체 (첫 번째)
            idx_to_replace = violating_indices[0]
            new_pos = self.generate_random_position()
            pillars_array[idx_to_replace] = new_pos
            
            if iteration_count % 100 == 0 and iteration_count > 0:
                print(f"        반복 횟수: {iteration_count}, 남은 충돌: {len(violating_indices)}개...")
            
            iteration_count += 1
        
        # 다시 리스트로 변환
        self.pillars = [tuple(p) for p in pillars_array]
        
        if iteration_count >= self.max_attempts:
            print(f"\n경고: 최대 반복 횟수({self.max_attempts})에 도달했습니다.")
        
        print("-" * 50)
        print(f"[완료] 랜덤 필러 생성 완료!")
        print(f"최종 기둥 개수: {len(self.pillars)}")
        print(f"총 반복 횟수: {iteration_count}")
        
        return np.array(self.pillars)
    
    def visualize_slice(self, save_path='pillar_slice.png'):
        """2D 슬라이스 이진 마스크 시각화
        기둥이 있는 픽셀: 1 (white)
        비어 있는 공간: 0 (black)
        """
        if len(self.pillars) == 0:
            print("먼저 generate_pillars()를 실행해주세요.")
            return
        
        # 이진 마스크 생성 (영역 크기와 동일한 픽셀 크기)
        width, height = int(self.domain_size[0]), int(self.domain_size[1])
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # 각 기둥을 마스크에 그리기
        print(f"\n이진 마스크 생성 중...")
        for i, (cx, cy) in enumerate(self.pillars):
            # 기둥 중심 좌표 (픽셀 좌표계로 변환)
            cx_px = int(cx)
            cy_px = int(cy)
            radius_px = int(self.pillar_radius)
            
            # 원형 영역을 1로 채우기
            y_indices, x_indices = np.ogrid[:height, :width]
            # 좌표계: 이미지는 top-left가 원점, 우리는 bottom-left가 원점
            # 따라서 y 좌표를 반전
            distances = np.sqrt((x_indices - cx_px)**2 + ((height - 1 - y_indices) - cy_px)**2)
            mask[distances <= radius_px] = 1
            
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(self.pillars)} 기둥 처리 완료")
        
        print(f"  전체 {len(self.pillars)}개 기둥 처리 완료")
        
        # 이진 마스크 저장 (흑백 이미지)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mask, cmap='gray', interpolation='nearest', origin='upper')
        ax.set_xlabel('X (nm)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Y (nm)', fontsize=12, fontweight='bold')
        ax.set_title(f'Random Pillar Binary Mask - {width}×{height} pixels\n'
                    f'Pillar Count: {len(self.pillars)} | '
                    f'Radius: {self.pillar_radius} nm | '
                    f'White(1): Pillar, Black(0): Empty',
                    fontsize=12, fontweight='bold', pad=20)
        
        # 통계 정보
        filled_pixels = np.sum(mask)
        total_pixels = mask.size
        fill_ratio = (filled_pixels / total_pixels) * 100
        
        stats_text = (f'Domain: {width} × {height} nm²\n'
                     f'Pillar pixels: {filled_pixels:,}\n'
                     f'Total pixels: {total_pixels:,}\n'
                     f'Fill ratio: {fill_ratio:.2f}%')
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n이진 마스크 저장 완료: {save_path}")
        plt.close()
        
        # 추가: 원본 마스크 배열을 numpy 파일로도 저장
        mask_npy_path = save_path.replace('.png', '_mask.npy')
        np.save(mask_npy_path, mask)
        print(f"numpy 배열 저장 완료: {mask_npy_path}")
        
        return mask
        
    def get_statistics(self):
        """기둥 배치 통계 정보"""
        if len(self.pillars) == 0:
            return None
        
        # 최근접 이웃 거리 계산
        min_distances = []
        for i, pos1 in enumerate(self.pillars):
            distances = []
            for j, pos2 in enumerate(self.pillars):
                if i != j:
                    edge_dist = self.calculate_edge_to_edge_distance(pos1, pos2)
                    distances.append(edge_dist)
            if distances:
                min_distances.append(min(distances))
        
        stats = {
            '기둥 개수': len(self.pillars),
            '기둥 반지름 (nm)': self.pillar_radius,
            '최소 edge-to-edge 거리 (nm)': np.min(min_distances) if min_distances else None,
            '평균 최근접 이웃 거리 (nm)': np.mean(min_distances) if min_distances else None,
            '최대 최근접 이웃 거리 (nm)': np.max(min_distances) if min_distances else None,
            '영역 크기 (nm²)': self.domain_size[0] * self.domain_size[1],
            '기둥 밀도 (/μm²)': len(self.pillars) / (self.domain_size[0] * self.domain_size[1]) * 1e6,
            '충진율 (%)': len(self.pillars) * np.pi * self.pillar_radius**2 / (self.domain_size[0] * self.domain_size[1]) * 100
        }
        
        return stats
    
    def print_statistics(self):
        """통계 정보 출력"""
        stats = self.get_statistics()
        if stats is None:
            print("통계 정보를 계산할 수 없습니다.")
            return
        
        print("\n" + "=" * 60)
        print("랜덤 필러 구성 통계")
        print("=" * 60)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key:30s}: {value:10.2f}")
            else:
                print(f"{key:30s}: {value}")
        print("=" * 60)


def main():
    """메인 실행 함수"""
    
    # ========================================
    # 파라미터 설정 (여기서 원하는 값으로 수정하세요!)
    # ========================================
    
    # -------------------- 기본 구조 파라미터 --------------------
    
    PILLAR_RADIUS = 45.0
    # 기둥의 반지름 (단위: nm, 나노미터)
    # - 모든 기둥이 동일한 크기의 원형으로 생성됩니다
    # - 값이 클수록 기둥이 커지고 충진율이 높아집니다
    # - 예: 10.0, 25.0, 45.0, 100.0
    
    MIN_EDGE_DISTANCE = 5.0
    # 기둥 간 최소 가장자리 거리 (단위: nm)
    # - 두 기둥의 가장자리(edge) 사이의 최소 간격
    # - 이 값보다 가까운 기둥은 재배치됩니다
    # - 값이 클수록 기둥 사이 간격이 넓어집니다
    # - 예: 0.0 (접촉 가능), 5.0 (기본), 10.0 (여유 있음)
    
    # -------------------- 시뮬레이션 영역 크기 --------------------
    
    DOMAIN_WIDTH = 10000
    # 시뮬레이션 영역의 가로 크기 (단위: nm)
    # - 10000 nm = 10 μm
    # - MEEP 시뮬레이션에서 2048×2048 픽셀로 리사이즈됨
    # - 예: 512, 1024, 2048, 4096, 8192, 10000
    
    DOMAIN_HEIGHT = 10000
    # 시뮬레이션 영역의 세로 크기 (단위: nm)
    # - 10000 nm = 10 μm (정사각형 도메인)
    # - MEEP 시뮬레이션에서 2048×2048 픽셀로 리사이즈됨
    # - 예: 512, 1024, 2048, 4096, 8192, 10000
    
    # -------------------- 기둥 밀도 제어 --------------------
    
    INITIAL_DENSITY = 29.5
    # 초기 기둥 배치 밀도 (단위: 개/μm², 마이크로미터 제곱당 개수)
    # - 이 밀도로 초기 기둥을 무작위 배치한 후, 거리 조건 위반 기둥을 조정합니다
    # - 목표: 10000×10000 nm² (100 μm²) 영역에 평균 2951 ± 14개 기둥
    #   → 29.5 /μm² × 100 μm² = 2950개
    # - 권장 범위:
    #   * 10-30: 낮은 밀도, 충진율 ~10-20%
    #   * 30-60: 중간 밀도, 충진율 ~20-35%
    #   * 60-100: 높은 밀도, 충진율 ~35-50%
    # - 예: 10.0 (희소), 29.5 (목표), 50.0 (조밀), 100.0 (매우 조밀)
    
    # -------------------- 알고리즘 설정 --------------------
    
    MAX_ITERATIONS = 10000
    # 기둥 위치 조정 최대 반복 횟수
    # - 모든 기둥이 거리 조건을 만족할 때까지 반복하는 최대 횟수
    # - 이 값을 초과하면 경고와 함께 종료됩니다
    # - 일반적으로 수백~수천 회 이내에 완료됩니다
    # - 예: 5000 (빠름), 10000 (기본), 20000 (안전)
    
    RANDOM_SEED = 42
    # 난수 생성 시드 (재현성 제어)
    # - 정수 값: 같은 시드를 사용하면 항상 같은 결과가 생성됩니다
    # - None: 실행할 때마다 다른 무작위 결과가 생성됩니다
    # - 재현 가능한 결과가 필요하면 정수(예: 42)를 사용하세요
    # - 다양한 샘플이 필요하면 None으로 설정하세요
    # - 예: 42, 123, 2024, None
    
    # -------------------- 출력 설정 --------------------
    
    # 현재 날짜와 시간을 파일명에 추가
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE = f'random_pillar_{timestamp}.png'
    # 생성될 이미지 파일의 이름
    # - PNG 형식의 이진 마스크 이미지로 저장됩니다
    # - 같은 이름의 .npy 파일(numpy 배열)도 함께 생성됩니다
    # - 파일명 형식: random_pillar_YYYYMMDD_HHMMSS.png
    # - 예: random_pillar_20241020_143052.png
    
    # ========================================
    
    # 랜덤 시드 설정
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED)
    
    print("\n" + "=" * 60)
    print("랜덤 필러 생성기 - 파라미터 설정")
    print("=" * 60)
    print(f"  기둥 반지름: {PILLAR_RADIUS} nm")
    print(f"  최소 간격: {MIN_EDGE_DISTANCE} nm")
    print(f"  영역 크기: {DOMAIN_WIDTH} × {DOMAIN_HEIGHT} nm²")
    print(f"  초기 밀도: {INITIAL_DENSITY} /μm²")
    print(f"  랜덤 시드: {RANDOM_SEED if RANDOM_SEED is not None else '랜덤'}")
    print(f"  출력 파일: {OUTPUT_FILE}")
    print("=" * 60 + "\n")
    
    # 랜덤 필러 생성기 초기화
    generator = RandomPillarGenerator(
        pillar_radius=PILLAR_RADIUS,
        min_edge_distance=MIN_EDGE_DISTANCE,
        domain_size=(DOMAIN_WIDTH, DOMAIN_HEIGHT),
        initial_density=INITIAL_DENSITY,
        max_attempts=MAX_ITERATIONS
    )
    
    # 랜덤 필러 생성
    pillars = generator.generate_pillars()
    
    # 통계 정보 출력
    generator.print_statistics()
    
    # 2D 슬라이스 시각화
    generator.visualize_slice(save_path=OUTPUT_FILE)


if __name__ == "__main__":
    main()

