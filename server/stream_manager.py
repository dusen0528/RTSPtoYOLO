"""
스트림 매니저 - 동적 스트림 관리
"""
import uuid
import psutil
import threading
import sys
from typing import Dict, Optional, List
from datetime import datetime
from ultralytics import YOLO

from .models import (
    StreamCreate, StreamUpdate, StreamInfo, StreamStats,
    ServerStats, BlurSettings, StreamStatus
)
from .stream_processor import StreamProcessor
from .config import settings


class StreamManager:
    """스트림 관리자 - 동적 스트림 추가/삭제/조회"""
    
    _instance = None
    
    def __new__(cls):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._streams: Dict[str, dict] = {}  # {id: {info, processor}}
        self._model = None
        self._model_lock = threading.Lock()  # 추론 락 (스레드 안전성)
        self._initialized = True
    
    @property
    def model_lock(self) -> threading.Lock:
        """추론 락 반환 (스레드 안전한 추론용)"""
        return self._model_lock
    
    @property
    def is_model_ready(self) -> bool:
        """모델 로드 상태 확인"""
        return self._model is not None
    
    def initialize_model(self):
        """YOLO 모델 초기화 (서버 시작 시 한 번만)"""
        if self._model is None:
            print(f"YOLO 모델 로드 중: {settings.model_path}")
            self._model = YOLO(settings.model_path)
            print("YOLO 모델 로드 완료!")
    
    def create_stream(self, data: StreamCreate) -> StreamInfo:
        """새 스트림 생성"""
        # 모델 로드 확인
        if not self.is_model_ready:
            raise RuntimeError("YOLO 모델이 로드되지 않았습니다. 서버를 재시작하세요.")

        # 입력/출력 URL 중복 방지 (옵션)
        if not data.allow_duplicate:
            normalized_input = data.input_url.strip()
            normalized_output = data.output_url.strip()
            for sid, stream in self._streams.items():
                info: StreamInfo = stream["info"]
                if info.input_url.strip() == normalized_input:
                    raise RuntimeError(f"동일 입력 URL 스트림이 이미 존재합니다 (id={sid})")
                if info.output_url.strip() == normalized_output:
                    raise RuntimeError(f"동일 출력 URL 스트림이 이미 존재합니다 (id={sid})")
        
        stream_id = str(uuid.uuid4())[:8]
        
        # 블러 설정 (기본값 또는 사용자 지정)
        blur_settings = data.blur_settings or BlurSettings()
        
        # 스트림 정보
        now = datetime.now()
        info = StreamInfo(
            id=stream_id,
            name=data.name,
            input_url=data.input_url,
            output_url=data.output_url,
            status=StreamStatus.PENDING,
            blur_settings=blur_settings,
            created_at=now,
        )
        
        # 프로세서 생성 (모델 + 락 전달)
        processor = StreamProcessor(
            stream_id=stream_id,
            input_url=data.input_url,
            output_url=data.output_url,
            model=self._model,
            model_lock=self._model_lock,  # 추론 락 전달
            blur_settings=blur_settings
        )
        
        self._streams[stream_id] = {
            'info': info,
            'processor': processor,
        }
        
        return info
    
    def start_stream(self, stream_id: str) -> Optional[StreamInfo]:
        """스트림 시작"""
        if stream_id not in self._streams:
            return None
        
        stream = self._streams[stream_id]
        processor: StreamProcessor = stream['processor']
        processor.start()
        
        # 상태 업데이트
        stream['info'].status = processor.status
        stream['info'].started_at = processor.started_at
        
        return stream['info']
    
    def stop_stream(self, stream_id: str) -> Optional[StreamInfo]:
        """스트림 중지"""
        if stream_id not in self._streams:
            return None
        
        stream = self._streams[stream_id]
        processor: StreamProcessor = stream['processor']
        processor.stop()
        
        # 상태 업데이트
        stream['info'].status = processor.status
        
        return stream['info']
    
    def delete_stream(self, stream_id: str) -> bool:
        """스트림 삭제"""
        if stream_id not in self._streams:
            return False
        
        # 실행 중이면 먼저 중지
        stream = self._streams[stream_id]
        processor: StreamProcessor = stream['processor']
        if processor.status == StreamStatus.RUNNING:
            processor.stop()
        
        del self._streams[stream_id]
        return True
    
    def get_stream(self, stream_id: str) -> Optional[StreamInfo]:
        """스트림 정보 조회"""
        if stream_id not in self._streams:
            return None
        
        stream = self._streams[stream_id]
        processor: StreamProcessor = stream['processor']
        info: StreamInfo = stream['info']
        
        # 실시간 통계 업데이트
        stats = processor.get_stats()
        info.status = processor.status
        info.fps = stats['fps']
        info.frame_count = stats['frame_count']
        info.faces_detected = stats['faces_detected']
        info.frames_skipped = stats['frames_skipped']
        info.cpu_usage = stats['cpu_usage']
        info.inference_time_ms = stats['inference_time_ms']
        info.error_message = processor.error_message
        info.ffmpeg_alive = stats['ffmpeg_alive']
        
        return info
    
    def get_all_streams(self) -> List[StreamInfo]:
        """모든 스트림 목록"""
        return [self.get_stream(sid) for sid in self._streams.keys()]
    
    def get_stream_by_input_url(self, input_url: str) -> Optional[StreamInfo]:
        """입력 RTSP URL로 스트림 조회"""
        # URL 정규화 (공백 제거, 슬래시 정규화)
        normalized_input = input_url.strip()
        
        # 디버깅: 모든 스트림의 input_url 출력
        print(f"[DEBUG] 조회 요청 input_url: '{normalized_input}' (길이: {len(normalized_input)})", flush=True)
        print(f"[DEBUG] 현재 등록된 스트림 수: {len(self._streams)}", flush=True)
        sys.stdout.flush()
        
        for stream_id, stream in self._streams.items():
            info: StreamInfo = stream['info']
            stored_url = info.input_url.strip()
            print(f"[DEBUG] 스트림 {stream_id}: 저장된 URL = '{stored_url}' (길이: {len(stored_url)})", flush=True)
            
            # 정확한 매칭
            if stored_url == normalized_input:
                print(f"[DEBUG] ✅ 매칭 성공! 스트림 ID: {stream_id}", flush=True)
                sys.stdout.flush()
                return self.get_stream(stream_id)
            else:
                # 차이점 출력
                if len(stored_url) != len(normalized_input):
                    print(f"[DEBUG] ❌ 길이 불일치: 저장={len(stored_url)}, 요청={len(normalized_input)}", flush=True)
                else:
                    # 첫 번째 다른 문자 찾기
                    for i, (s, n) in enumerate(zip(stored_url, normalized_input)):
                        if s != n:
                            print(f"[DEBUG] ❌ 위치 {i}에서 불일치: 저장='{s}'({ord(s)}), 요청='{n}'({ord(n)})", flush=True)
                            break
            sys.stdout.flush()
        
        print(f"[DEBUG] ❌ 매칭 실패: 해당 input_url을 가진 스트림이 없습니다", flush=True)
        sys.stdout.flush()
        return None
    
    def update_stream(self, stream_id: str, data: StreamUpdate) -> Optional[StreamInfo]:
        """스트림 설정 업데이트"""
        if stream_id not in self._streams:
            return None
        
        stream = self._streams[stream_id]
        info: StreamInfo = stream['info']
        processor: StreamProcessor = stream['processor']
        
        # 이름 업데이트
        if data.name:
            info.name = data.name
        
        # 블러 설정 업데이트
        if data.blur_settings:
            info.blur_settings = data.blur_settings
            processor.update_settings(data.blur_settings)
        
        return info
    
    def get_server_stats(self) -> ServerStats:
        """서버 전체 통계 (논블로킹)"""
        total = len(self._streams)
        running = sum(
            1 for s in self._streams.values()
            if s['processor'].status == StreamStatus.RUNNING
        )
        
        # 프로세스 리소스 (interval=0으로 논블로킹)
        process = psutil.Process()
        cpu_usage = process.cpu_percent(interval=0) / psutil.cpu_count()
        mem_mb = process.memory_info().rss / (1024 * 1024)
        
        # 시스템 전체 (interval=0으로 논블로킹)
        sys_cpu = psutil.cpu_percent(interval=0)
        sys_mem = psutil.virtual_memory().percent
        
        # 스트림별 통계
        streams = []
        total_fps = 0.0
        total_skip_ratio = 0.0
        counted = 0
        for sid, s in self._streams.items():
            processor: StreamProcessor = s['processor']
            stats = processor.get_stats()
            streams.append(StreamStats(
                id=sid,
                name=s['info'].name,
                status=processor.status,
                fps=stats['fps'],
                frame_count=stats['frame_count'],
                faces_detected=stats['faces_detected'],
                frames_skipped=stats['frames_skipped'],
                cpu_usage=stats['cpu_usage'],
                inference_time_ms=stats['inference_time_ms'],
                uptime_seconds=stats['uptime_seconds'],
                ffmpeg_alive=stats['ffmpeg_alive'],
            ))

            skip_ratio = stats['frames_skipped'] / max(1, stats['frame_count'])
            total_skip_ratio += skip_ratio
            total_fps += stats['fps']
            counted += 1

        average_fps = total_fps / counted if counted else 0.0
        average_skip_ratio = total_skip_ratio / counted if counted else 0.0
        
        return ServerStats(
            total_streams=total,
            running_streams=running,
            total_cpu_usage=cpu_usage,
            total_memory_mb=mem_mb,
            system_cpu_percent=sys_cpu,
            system_memory_percent=sys_mem,
            model_ready=self.is_model_ready,
            average_fps=average_fps,
            average_skip_ratio=average_skip_ratio,
            streams=streams,
        )
    
    def shutdown(self):
        """서버 종료 시 모든 스트림 중지"""
        for stream_id in list(self._streams.keys()):
            self.delete_stream(stream_id)


# 전역 매니저 인스턴스
manager = StreamManager()

