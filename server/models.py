"""
Pydantic 모델 정의
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


class StreamStatus(str, Enum):
    """스트림 상태"""
    PENDING = "pending"      # 대기 중
    STARTING = "starting"    # 시작 중 (연결 시도)
    RUNNING = "running"      # 실행 중
    STOPPED = "stopped"      # 중지됨
    ERROR = "error"          # 오류


class AnonymizeMethod(str, Enum):
    """익명화 방법"""
    BLUR = "blur"           # 가우시안 블러 (느림, 부드러움)
    PIXELATE = "pixelate"   # 픽셀화 (빠름, 효율적) ⚡
    MOSAIC = "mosaic"       # 모자이크 (빠름, 효율적) ⚡
    BLACK = "black"         # 검은 박스 (가장 빠름)
    SOLID = "solid"         # 단색 채우기 (빠름)


class BlurSettings(BaseModel):
    """블러 설정"""
    confidence_threshold: float = Field(default=0.09, ge=0.01, le=1.0, description="감지 신뢰도 (낮을수록 민감, 멀리 있는 작은 사람 감지 향상)")
    anonymize_method: AnonymizeMethod = Field(default=AnonymizeMethod.PIXELATE, description="익명화 방법 (pixelate=빠름)")
    blur_strength: int = Field(default=31, ge=11, le=101, description="블러 강도 (홀수, blur 방법만 사용)")
    pixelate_size: int = Field(default=10, ge=5, le=50, description="픽셀화 크기 (pixelate/mosaic 방법)")
    imgsz: int = Field(default=320, description="감지 이미지 크기 (320/480/640)")
    max_age: int = Field(default=25, ge=1, le=100, description="블러 유지 프레임 수")
    smoothing: float = Field(default=0.5, ge=0.0, le=1.0, description="박스 스무딩 강도")
    iou_threshold: float = Field(default=0.45, ge=0.1, le=1.0, description="NMS IoU 임계값 (낮을수록 더 많은 감지)")
    box_expand_ratio: float = Field(default=0.1, ge=0.0, le=0.5, description="박스 확장 비율 (측면 얼굴 커버)")


class StreamCreate(BaseModel):
    """스트림 생성 요청"""
    name: str = Field(..., min_length=1, max_length=100, description="스트림 이름")
    input_url: str = Field(..., description="입력 RTSP URL")
    output_url: str = Field(..., description="출력 RTSP URL (Flashphoner)")
    blur_settings: Optional[BlurSettings] = Field(default=None, description="블러 설정 (없으면 기본값)")


class StreamUpdate(BaseModel):
    """스트림 업데이트 요청"""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    blur_settings: Optional[BlurSettings] = None


class StreamInfo(BaseModel):
    """스트림 정보 응답"""
    id: str
    name: str
    input_url: str
    output_url: str
    status: StreamStatus
    blur_settings: BlurSettings
    created_at: datetime
    started_at: Optional[datetime] = None
    
    # 실시간 통계
    fps: float = 0.0
    frame_count: int = 0
    faces_detected: int = 0
    cpu_usage: float = 0.0
    inference_time_ms: float = 0.0
    error_message: Optional[str] = None


class StreamStats(BaseModel):
    """스트림 실시간 통계"""
    id: str
    name: str
    status: StreamStatus
    fps: float
    frame_count: int
    faces_detected: int
    cpu_usage: float
    inference_time_ms: float
    uptime_seconds: float


class ServerStats(BaseModel):
    """서버 전체 통계"""
    total_streams: int
    running_streams: int
    total_cpu_usage: float
    total_memory_mb: float
    system_cpu_percent: float
    system_memory_percent: float
    streams: List[StreamStats]


class ApiResponse(BaseModel):
    """API 응답 공통 형식"""
    success: bool
    message: str
    data: Optional[dict] = None

