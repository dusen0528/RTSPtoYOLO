"""
서버 설정
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """서버 전역 설정"""
    
    # 서버 설정
    host: str = "0.0.0.0"
    port: int = 8000
    
    # YOLO 모델 설정
    model_path: str = "yolov8n.pt"  # person 감지용 (얼굴 대신 머리 전체 블러)
    use_person_detection: bool = True  # person 감지 모드 (True: 머리 전체, False: 얼굴만)
    head_region_ratio: float = 0.45  # person 박스에서 머리 영역 비율 (상단 45%, 머리+목 포함)
    
    # 기본 블러 설정
    default_confidence: float = 0.09  # 낮은 신뢰도 = 작은 얼굴/측면 얼굴도 감지 (멀리 있는 작은 사람 감지 향상)
    default_anonymize_method: str = "pixelate"  # pixelate=빠름, blur=느림
    default_blur_strength: int = 31
    default_pixelate_size: int = 10  # 픽셀화 크기 (5~15 권장)
    default_imgsz: int = 320  # 경량화 모드
    default_max_age: int = 50  # 블러 유지 프레임 수 (25 → 50, 배경 얼굴 안정화)
    default_smoothing: float = 0.5
    default_iou_threshold: float = 0.35  # 더 많은 감지 (0.45 → 0.35, 작은 얼굴 감지 개선)
    default_box_expand_ratio: float = 0.15  # 측면 얼굴 커버 강화 (0.1 → 0.15)
    
    # FFmpeg 설정
    ffmpeg_path: str = "ffmpeg"
    output_fps: int = 25  # 입력 FPS와 맞추기 (15 → 25/30 권장)
    output_bitrate: str = "4000k"  # 화질 개선 (1500k → 4000k 이상 권장)
    output_crf: Optional[int] = None  # CRF 값 (None이면 비트레이트 사용, 18-23 권장)
    output_preset: str = "veryfast"  # 인코딩 프리셋 (ultrafast → veryfast, 화질과 속도 균형)
    output_codec: str = "libx264"  # 인코딩 코덱 (libx264, h264_nvenc, h264_qsv 등)
    
    # 부하 제어 설정
    max_fps: int = 15  # 최대 처리 FPS (부하 제어) - 30에서 15로 낮춤 (성능 향상)
    enable_frame_skip: bool = True  # 프레임 스킵 활성화
    output_scale: float = 1.0  # 출력 해상도 배율 (0.5 = 절반)
    inference_timeout: float = 5.0  # 추론 타임아웃 (초)
    max_concurrent_streams: int = 10  # 최대 동시 스트림 수
    frame_skip_ratio: int = 2  # 레거시(사용 안 함): 과거 스킵 정책 호환용
    skip_interval: int = 3  # N프레임마다 1번만 추론 (3 = 3프레임마다 1번)
    
    class Config:
        env_file = ".env"
        env_prefix = "BLUR_"


settings = Settings()
