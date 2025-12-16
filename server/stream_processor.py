"""
스트림 프로세서 - YOLO 얼굴 감지 + 블러 처리 + FFmpeg RTSP 출력
"""
import cv2
import numpy as np
import subprocess
import threading
import time
from collections import deque
from typing import Optional, Tuple
from datetime import datetime
import psutil
import os
import select

from .models import BlurSettings, StreamStatus
from .config import settings


class FaceTracker:
    """얼굴 위치 추적 (블러 안정화)"""
    
    def __init__(self, max_age: int = 25, smoothing: float = 0.5):
        self.tracks = {}
        self.next_id = 0
        self.max_age = max_age
        self.smoothing = smoothing
    
    def update(self, detections: list) -> list:
        """새 감지 결과로 트랙 업데이트"""
        # 모든 트랙의 age 증가
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['age'] += 1
            if self.tracks[track_id]['age'] > self.max_age:
                del self.tracks[track_id]
        
        # 새 감지와 기존 트랙 매칭
        used_tracks = set()
        for det in detections:
            best_iou = 0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track_id in used_tracks:
                    continue
                iou = self._calc_iou(det, track['box'])
                if iou > best_iou and iou > 0.3:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                old_box = self.tracks[best_track_id]['box']
                new_box = [
                    int(old_box[i] * self.smoothing + det[i] * (1 - self.smoothing))
                    for i in range(4)
                ]
                self.tracks[best_track_id] = {'box': new_box, 'age': 0}
                used_tracks.add(best_track_id)
            else:
                self.tracks[self.next_id] = {'box': list(det), 'age': 0}
                self.next_id += 1
        
        return [track['box'] for track in self.tracks.values()]
    
    def _calc_iou(self, box1, box2) -> float:
        """IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0
    
    def reset(self):
        """트래커 초기화"""
        self.tracks = {}
        self.next_id = 0


class StreamProcessor:
    """RTSP 스트림 처리기"""
    
    def __init__(
        self,
        stream_id: str,
        input_url: str,
        output_url: str,
        model,  # YOLO 모델 (공유)
        model_lock: threading.Lock,  # 추론 락 (스레드 안전성)
        blur_settings: BlurSettings
    ):
        self.stream_id = stream_id
        self.input_url = input_url
        self.output_url = output_url
        self.model = model
        self.model_lock = model_lock
        self.blur_settings = blur_settings
        
        # 상태
        self.status = StreamStatus.PENDING
        self.error_message: Optional[str] = None
        self.started_at: Optional[datetime] = None
        
        # 통계
        self.frame_count = 0
        self.frames_skipped = 0  # 스킵된 프레임 수
        self.faces_detected = 0
        self.fps = 0.0
        self.cpu_usage = 0.0
        self.inference_time_ms = 0.0
        
        # 내부
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._ffmpeg_process: Optional[subprocess.Popen] = None
        self._tracker = FaceTracker(
            max_age=blur_settings.max_age,
            smoothing=blur_settings.smoothing
        )
        
        # FFmpeg 에러 메시지 저장용
        self._ffmpeg_stderr = None
        
        # FPS 계산용
        self._fps_times = deque(maxlen=30)
        self._inference_times = deque(maxlen=30)
        self._last_frame_received_time = None  # 마지막 프레임 수신 시간 (중단 감지용)
        
        # 부하 제어용
        self._min_frame_interval = 1.0 / settings.max_fps  # FPS 제한
        self._last_frame_time = 0.0
        self._last_boxes = []  # 마지막 감지 박스 (프레임 스킵 시 재사용)
        self._last_frame = None  # 마지막 전송 프레임
        
        # 프레임 스킵 제어
        self._frame_index = 0  # 프레임 인덱스 (frame_skip_ratio 제어용)
        self.frame_count = 0  # 최적화: 프레임 카운터 (skip_interval 제어용)
        self.skip_interval = getattr(settings, 'skip_interval', 3)  # 최적화: 3프레임마다 1번만 추론
        
        # 입력 FPS 저장 (동적 FPS 설정용)
        self._input_fps = 25  # 기본값
    
    def start(self):
        """스트림 처리 시작 (STARTING 상태로 전환, 성공 후 RUNNING)"""
        # RUNNING이나 STARTING 상태면 재시작 불필요
        if self.status in (StreamStatus.RUNNING, StreamStatus.STARTING):
            return
        
        # ERROR 상태에서 재시작: 기존 리소스 정리 후 재시작
        if self.status == StreamStatus.ERROR:
            print(f"[{self.stream_id}] ERROR 상태에서 재시작 시도...")
            # 기존 스레드가 있으면 정리
            if self._thread and self._thread.is_alive():
                self._stop_event.set()
                # 리소스 강제 정리
                self._cleanup_fast()
                # 스레드 종료 대기 (짧은 타임아웃)
                self._thread.join(timeout=1.0)
                if self._thread.is_alive():
                    print(f"[{self.stream_id}] 경고: 기존 스레드가 종료되지 않았지만 재시작 진행")
            self._thread = None
        
        # 리소스 초기화
        self._stop_event.clear()
        self.error_message = None
        self.status = StreamStatus.STARTING  # 시작 중 상태
        self.started_at = datetime.now()
        
        # 트래커 초기화 (재시작 시)
        self._tracker.reset()
        self.frame_count = 0
        self._frame_index = 0
        self._last_boxes = []
        
        # 새 스레드 시작
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print(f"[{self.stream_id}] 재시작 요청 완료 (상태: {self.status.value})")
    
    def stop(self):
        """스트림 처리 중지 (빠른 종료 + 강제 정리)"""
        if self.status == StreamStatus.STOPPED:
            return
        
        print(f"[{self.stream_id}] 스트림 중지 요청...")
        self._stop_event.set()
        
        # 1. FFmpeg stdin 즉시 닫기 (블로킹 방지)
        if self._ffmpeg_process and self._ffmpeg_process.stdin:
            try:
                self._ffmpeg_process.stdin.close()
            except:
                pass
        
        # 2. VideoCapture 강제 해제 (read() 블로킹 탈출) - 즉시 해제
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception as e:
                print(f"[{self.stream_id}] VideoCapture 해제 오류: {e}")
            finally:
                self._cap = None  # 즉시 None으로 설정
        
        # 3. FFmpeg 프로세스 즉시 강제 종료 (terminate 대신 kill 우선)
        if self._ffmpeg_process:
            try:
                # stdin이 이미 닫혔으므로 kill로 즉시 종료
                self._ffmpeg_process.kill()
            except:
                pass
        
        # 4. 스레드 종료 대기 (타임아웃 0.5초로 단축)
        if self._thread and self._thread.is_alive():
            # 백오프 재시도 3회 (총 최대 1.5초)
            for attempt in range(3):
                self._thread.join(timeout=0.5)
                if not self._thread.is_alive():
                    break
                print(f"[{self.stream_id}] 경고: 스레드 종료 대기 중 (재시도 {attempt + 1}/3)...")
            
            # 스레드가 여전히 살아있으면 상태를 ERROR로 표시하여 재시작 방지
            if self._thread.is_alive():
                print(f"[{self.stream_id}] 경고: 스레드가 종료되지 않아 강제 정리 후 ERROR 상태로 설정")
                self.status = StreamStatus.ERROR
        
        # 5. 최종 정리 (비동기로 실행하여 빠른 반환)
        self._cleanup_fast()
        if self.status != StreamStatus.ERROR:
            self.status = StreamStatus.STOPPED
        self._thread = None
        print(f"[{self.stream_id}] 스트림 중지 완료")
    
    def update_settings(self, blur_settings: BlurSettings):
        """블러 설정 업데이트"""
        self.blur_settings = blur_settings
        self._tracker = FaceTracker(
            max_age=blur_settings.max_age,
            smoothing=blur_settings.smoothing
        )
    
    def get_stats(self) -> dict:
        """현재 통계 반환"""
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.now() - self.started_at).total_seconds()
        
        return {
            'fps': self.fps,
            'frame_count': self.frame_count,
            'faces_detected': self.faces_detected,
            'frames_skipped': self.frames_skipped,
            'cpu_usage': self.cpu_usage,
            'inference_time_ms': self.inference_time_ms,
            'uptime_seconds': uptime,
            'ffmpeg_alive': self._ffmpeg_is_alive(),
        }
    
    def _process_loop(self):
        """메인 처리 루프 (부하 제어 + 안정성 강화)"""
        first_frame_success = False
        
        try:
            # 모델 체크
            if self.model is None:
                raise Exception("YOLO 모델이 로드되지 않았습니다.")
            
            # 입력 스트림 연결 (지연 최소화)
            print(f"[{self.stream_id}] RTSP 연결 중: {self.input_url[:50]}...")
            
            # RTSP 저지연 옵션을 os.environ으로 설정 (OpenCV FFmpeg이 올바르게 인식하도록)
            # rw_timeout 단축: read()가 0.5초 내에 반환하여 stop 이벤트 빠르게 확인 가능
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|"
                "fflags;nobuffer|"
                "flags;low_delay|"
                "max_delay;0|"
                "reorder_queue_size;0|"
                "stimeout;500000|"  # 0.5초로 단축
                "rw_timeout;500000"  # 0.5초로 단축 (stop 시 빠른 반환)
            )
            
            self._cap = cv2.VideoCapture(self.input_url, cv2.CAP_FFMPEG)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self._cap.isOpened():
                # 구체적인 에러 메시지 생성
                error_msg = f"입력 RTSP 스트림 연결 실패: {self.input_url}"
                error_msg += "\n가능한 원인:"
                error_msg += "\n  - RTSP 서버가 다운되었거나 접근 불가"
                error_msg += "\n  - URL이 잘못되었거나 스트림이 존재하지 않음"
                error_msg += "\n  - 네트워크 연결 문제"
                error_msg += "\n  - 인증 정보가 필요하거나 권한 없음"
                raise Exception(error_msg)
            
            # 프레임 정보 가져오기
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 입력 FPS 동적 추출
            input_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if input_fps <= 0 or input_fps > 60:
                input_fps = 25  # 기본값
            fps = round(input_fps)
            self._input_fps = fps  # 인스턴스 변수에 저장
            
            if width == 0 or height == 0:
                width, height = 1920, 1080  # 기본값
            
            # 출력 해상도 계산 (다운스케일)
            out_width = int(width * settings.output_scale)
            out_height = int(height * settings.output_scale)
            
            # FFmpeg 출력 프로세스 시작 (FPS 전달)
            self._start_ffmpeg(out_width, out_height, fps)
            
            # FFmpeg 시작 확인 (조금 더 기다려서 연결 시도 확인)
            time.sleep(1.0)  # 0.5초 → 1초로 증가 (연결 시도 시간 확보)
            if not self._check_ffmpeg_alive():
                error_msg = self._get_ffmpeg_error()
                # 에러 메시지가 있으면 상세 출력
                if error_msg:
                    print(f"[{self.stream_id}] FFmpeg 연결 실패 상세:")
                    for line in error_msg.split('\n')[:10]:  # 최대 10줄만
                        if line.strip():
                            print(f"[{self.stream_id}]    {line}")
                raise Exception(f"FFmpeg 시작 실패: {self.output_url}. {error_msg if error_msg else '에러 메시지 없음'}")
            else:
                print(f"[{self.stream_id}] FFmpeg 연결 성공: {self.output_url}")
            
            process = psutil.Process()
            reconnect_count = 0
            max_reconnect = 5
            
            while not self._stop_event.is_set():
                current_time = time.time()
                
                # 중지 체크 (read() 전)
                if self._stop_event.is_set():
                    break
                
                # VideoCapture가 None이면 즉시 종료
                if self._cap is None:
                    break
                
                ret, frame = self._cap.read()
                
                # 중지 체크 (read() 후 즉시)
                if self._stop_event.is_set() or self._cap is None:
                    break
                
                if not ret or frame is None:
                    # 중지 체크 (read() 실패 시 즉시 확인)
                    if self._stop_event.is_set():
                        break
                    
                    reconnect_count += 1
                    
                    # 재연결 시도 전에 에러 원인 파악
                    error_details = []
                    if self._cap:
                        # VideoCapture 상태 확인
                        if not self._cap.isOpened():
                            error_details.append("VideoCapture 연결 끊김")
                        # 프레임 정보 확인
                        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0
                        if width == 0:
                            error_details.append("스트림 해상도 정보 없음")
                    
                    if reconnect_count > max_reconnect:
                        # 구체적인 에러 메시지 생성
                        error_parts = [f"RTSP 재연결 실패 (시도 {max_reconnect}회)"]
                        if error_details:
                            error_parts.append(f"원인: {', '.join(error_details)}")
                        error_parts.append(f"입력 URL: {self.input_url}")
                        error_msg = ". ".join(error_parts)
                        raise Exception(error_msg)
                    
                    # 지수 백오프: 재시도 간격 점진적 증가 (1초, 2초, 3초, 4초, 5초)
                    retry_delay = min(reconnect_count, 5)
                    print(f"[{self.stream_id}] 프레임 수신 실패, 재연결 시도 {reconnect_count}/{max_reconnect} (대기 {retry_delay}초)...")
                    if error_details:
                        print(f"[{self.stream_id}]    상세: {', '.join(error_details)}")
                    
                    # 중지 체크 (재연결 대기 중)
                    if self._stop_event.wait(timeout=retry_delay):
                        break
                    
                    # 중지 체크 (재연결 전)
                    if self._stop_event.is_set():
                        break
                    
                    # 재연결 시도
                    try:
                        if self._cap:
                            self._cap.release()
                    except:
                        pass
                    
                    # 재연결 시에도 RTSP 저지연 옵션 적용
                    try:
                        self._cap = cv2.VideoCapture(self.input_url, cv2.CAP_FFMPEG)
                        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # 재연결 확인
                        if not self._cap.isOpened():
                            print(f"[{self.stream_id}] 재연결 실패: VideoCapture 열기 실패")
                            continue
                        
                        # 재연결 성공 확인 (짧은 대기 후 프레임 읽기 시도)
                        time.sleep(0.5)
                        test_ret, _ = self._cap.read()
                        if test_ret:
                            print(f"[{self.stream_id}] 재연결 성공!")
                            reconnect_count = 0  # 성공 시 카운터 리셋
                    except Exception as e:
                        print(f"[{self.stream_id}] 재연결 중 오류: {e}")
                    
                    continue
                
                reconnect_count = 0  # 성공 시 카운터 리셋
                self._last_frame_received_time = current_time  # 프레임 수신 시간 기록
                
                # 첫 프레임 성공 → RUNNING 상태로 전환
                if not first_frame_success:
                    first_frame_success = True
                    self.status = StreamStatus.RUNNING
                    print(f"[{self.stream_id}] 스트림 시작 성공! ({width}x{height})")
                
                # 중지 체크 (처리 전)
                if self._stop_event.is_set():
                    break
                
                # FPS 제한: 너무 빠르면 프레임 스킵
                time_since_last = current_time - self._last_frame_time
                should_skip_fps = settings.enable_frame_skip and time_since_last < self._min_frame_interval
                
                # frame_skip_ratio를 사용하여 프레임 드롭 (레거시)
                self._frame_index += 1
                should_process_frame = (self._frame_index % settings.frame_skip_ratio == 0)
                
                # 최적화: 프레임 스킵 (3프레임마다 1번만 추론)
                self.frame_count += 1
                should_run_inference = (self.frame_count % self.skip_interval == 0)
                
                # 기본값: 이전 프레임의 박스 사용
                stable_boxes = self._last_boxes
                
                # FPS 제한 또는 레거시 프레임 스킵 조건
                if should_skip_fps or not should_process_frame:
                    self.frames_skipped += 1
                # 최적화: 설정한 간격마다만 YOLO 추론 실행
                elif should_run_inference:
                    # 최적화: 설정한 간격마다만 YOLO 추론 실행
                    # YOLO 추론 (락으로 스레드 안전 + 타임아웃)
                    inference_start = time.time()
                    
                    acquired = self.model_lock.acquire(timeout=settings.inference_timeout)
                    if not acquired:
                        # 타임아웃: 이전 박스 사용
                        print(f"[{self.stream_id}] 추론 타임아웃, 프레임 드롭")
                    else:
                        try:
                            # 중지 체크 (추론 전)
                            if not self._stop_event.is_set():
                                # person 감지 모드인 경우 classes=[0]으로 person만 감지
                                # imgsz 방어 코드: 속성이 없으면 기본값 사용
                                imgsz = getattr(self.blur_settings, 'imgsz', 320)
                                model_kwargs = {
                                    'verbose': False,
                                    'imgsz': imgsz,
                                    'conf': self.blur_settings.confidence_threshold,
                                    'iou': self.blur_settings.iou_threshold
                                }
                                
                                if settings.use_person_detection:
                                    model_kwargs['classes'] = [0]  # person 클래스만 (COCO 데이터셋 기준)
                                
                                results = self.model(frame, **model_kwargs)
                                
                                # 중지 체크 (추론 후)
                                if not self._stop_event.is_set():
                                    inference_time = time.time() - inference_start
                                    self._inference_times.append(inference_time * 1000)
                                    
                                    # 감지된 객체 추출 (person 또는 face)
                                    detections = []
                                    h, w = frame.shape[:2]
                                    
                                    for box in results[0].boxes:
                                        # 클래스 확인 (person = 0, face는 별도 클래스)
                                        cls_id = int(box.cls[0].cpu().numpy())
                                        cls_name = results[0].names[cls_id] if hasattr(results[0], 'names') else None
                                        
                                        # person 감지 모드인 경우 person 클래스만 필터링
                                        if settings.use_person_detection:
                                            # person 클래스는 보통 0번 (COCO 데이터셋 기준)
                                            if cls_id != 0 and cls_name != 'person':
                                                continue
                                        
                                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                        
                                        # 박스 크기 계산 (확장 전 원본 크기)
                                        box_width = x2 - x1
                                        box_height = y2 - y1
                                        
                                        # 박스 확장 비율을 가로/세로 분리 (가로는 작게, 세로는 적당히)
                                        expand_x_ratio = 0.05  # 가로 확장은 5% 정도만
                                        expand_y_ratio = 0.08  # 세로는 8% 정도
                                        
                                        expand_x = int(box_width * expand_x_ratio)
                                        expand_y = int(box_height * expand_y_ratio)
                                        
                                        # 박스 확장 적용
                                        x1 = max(0, x1 - expand_x)
                                        y1 = max(0, y1 - expand_y)
                                        x2 = min(w, x2 + expand_x)
                                        y2 = min(h, y2 + expand_y)
                                        
                                        # 머리 부분만 자르기: 상단 60% 영역만 사용
                                        head_ratio = 0.6
                                        new_height = int((y2 - y1) * head_ratio)
                                        y2 = y1 + new_height
                                        
                                        # 세로 길이에 맞춰서 정사각형으로 만들기
                                        height = y2 - y1
                                        center_x = (x1 + x2) // 2
                                        half_size = height // 2
                                        
                                        x1 = max(0, center_x - half_size)
                                        x2 = min(w, center_x + half_size)
                                        
                                        # 정사각형이 프레임을 벗어나면 조정
                                        if x2 - x1 < height:
                                            # 가로가 세로보다 작으면 세로를 가로에 맞춤
                                            if x2 - x1 > 0:
                                                y2 = y1 + (x2 - x1)
                                        
                                        detections.append((x1, y1, x2, y2))
                                    
                                    # 트래커로 안정화
                                    stable_boxes = self._tracker.update(detections)
                                    self._last_boxes = stable_boxes  # 최적화: 다음 스킵 프레임에서 재사용
                                    self._last_frame_time = current_time
                        finally:
                            self.model_lock.release()
                        
                        # 중지 체크 (락 해제 후)
                        if self._stop_event.is_set():
                            break
                else:
                    # 최적화: 추론 안 하는 프레임은 이전 좌표(last_boxes)를 그대로 사용
                    self.frames_skipped += 1
                
                self.faces_detected = len(stable_boxes)
                
                # 블러 처리
                for box in stable_boxes:
                    frame = self._apply_blur(frame, box)

                # 추론/인코딩을 수행하지 않은 스킵 프레임도 처리 시점 기록
                if should_skip_fps or not should_process_frame:
                    self._last_frame_time = current_time
                
                # 해상도 다운스케일 (설정된 경우)
                if settings.output_scale != 1.0:
                    frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                
                # 프레임 크기 검증 (FFmpeg 명령어와 일치해야 함)
                h, w = frame.shape[:2]
                if h != out_height or w != out_width:
                    # 크기가 맞지 않으면 리사이즈
                    frame = cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_LINEAR)
                
                # 중지 체크
                if self._stop_event.is_set():
                    break
                
                # FFmpeg으로 출력 (exit 감시 포함)
                if not self._check_ffmpeg_alive():
                    if not self._stop_event.is_set():
                        error_msg = self._get_ffmpeg_error()
                        if error_msg:
                            # FFmpeg 에러 분석
                            error_lower = error_msg.lower()
                            if "connection refused" in error_lower or "404" in error_msg:
                                print(f"[{self.stream_id}] FFmpeg 연결 실패: 출력 RTSP 서버에 연결할 수 없음")
                                print(f"[{self.stream_id}]    URL: {self.output_url}")
                            elif "timeout" in error_lower:
                                print(f"[{self.stream_id}] FFmpeg 타임아웃: 출력 서버 응답 없음")
                            elif "permission denied" in error_lower or "forbidden" in error_lower:
                                print(f"[{self.stream_id}] FFmpeg 권한 오류: 출력 서버 접근 거부")
                            else:
                                print(f"[{self.stream_id}] FFmpeg 에러: {error_msg[:300]}")
                        print(f"[{self.stream_id}] FFmpeg 재시작 시도...")
                        try:
                            self._restart_ffmpeg(out_width, out_height, self._input_fps)
                            print(f"[{self.stream_id}] FFmpeg 재시작 완료")
                        except Exception as e:
                            print(f"[{self.stream_id}] FFmpeg 재시작 실패: {e}")
                
                # 중지 체크
                if self._stop_event.is_set():
                    break
                
                if self._ffmpeg_process and self._ffmpeg_process.poll() is None:
                    try:
                        if not self._stop_event.is_set():
                            # 프레임 검증
                            if frame is None or frame.size == 0:
                                continue
                            
                            frame_bytes = frame.tobytes()
                            expected_size = out_width * out_height * 3  # BGR24 = 3 bytes per pixel
                            if len(frame_bytes) != expected_size:
                                continue
                            
                            self._ffmpeg_process.stdin.write(frame_bytes)
                            self._ffmpeg_process.stdin.flush()  # 버퍼 플러시 추가
                    except (BrokenPipeError, OSError):
                        if not self._stop_event.is_set():
                            self._restart_ffmpeg(out_width, out_height, self._input_fps)
                
                # 통계 업데이트
                self.frame_count += 1
                self._fps_times.append(time.time())
                
                if len(self._fps_times) >= 2:
                    elapsed = self._fps_times[-1] - self._fps_times[0]
                    if elapsed > 0:
                        self.fps = len(self._fps_times) / elapsed
                
                if self._inference_times:
                    self.inference_time_ms = sum(self._inference_times) / len(self._inference_times)
                
                # 프레임 수신 중단 감지 (5초 이상 프레임 없으면 경고)
                if self._last_frame_received_time and self.status == StreamStatus.RUNNING:
                    time_since_last_frame = current_time - self._last_frame_received_time
                    if time_since_last_frame > 5.0:
                        print(f"[{self.stream_id}] 프레임 수신 중단 감지 ({time_since_last_frame:.1f}초), 재연결 시도...")
                        # VideoCapture 재연결 시도
                        try:
                            if self._cap:
                                self._cap.release()
                            self._cap = cv2.VideoCapture(self.input_url, cv2.CAP_FFMPEG)
                            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            if not self._cap.isOpened():
                                print(f"[{self.stream_id}] 재연결 실패")
                            else:
                                self._last_frame_received_time = current_time  # 재연결 성공 시 시간 갱신
                        except Exception as e:
                            print(f"[{self.stream_id}] 재연결 중 오류: {e}")
                
                # CPU 사용률 (30프레임마다, 논블로킹)
                if self.frame_count % 30 == 0:
                    try:
                        self.cpu_usage = process.cpu_percent(interval=0) / psutil.cpu_count()
                    except:
                        pass
        
        except Exception as e:
            self.status = StreamStatus.ERROR
            error_str = str(e)
            self.error_message = error_str
            
            # 에러 타입별 상세 로깅
            error_lower = error_str.lower()
            if "rtsp" in error_lower or "재연결" in error_str:
                print(f"[{self.stream_id}] RTSP 연결 에러: {error_str}")
            elif "ffmpeg" in error_lower:
                print(f"[{self.stream_id}] FFmpeg 에러: {error_str}")
            elif "모델" in error_str or "yolo" in error_lower:
                print(f"[{self.stream_id}] 모델 에러: {error_str}")
            else:
                print(f"[{self.stream_id}] 에러: {error_str}")
        finally:
            self._cleanup()
            # 성공 못하고 종료되면 ERROR 상태
            if not first_frame_success and self.status == StreamStatus.STARTING:
                self.status = StreamStatus.ERROR
                if not self.error_message:
                    self.error_message = "첫 프레임 수신 전 종료됨"
    
    def _apply_blur(self, frame: np.ndarray, box: tuple) -> np.ndarray:
        """익명화 처리 (블러/픽셀화/모자이크 등)"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        method = self.blur_settings.anonymize_method
        face_region = frame[y1:y2, x1:x2]
        face_h, face_w = face_region.shape[:2]
        
        if method == "blur":
            # 가우시안 블러 (느림, 부드러움)
            blur_strength = self.blur_settings.blur_strength
            if blur_strength % 2 == 0:
                blur_strength += 1
            processed = cv2.GaussianBlur(face_region, (blur_strength, blur_strength), 0)
            
        elif method == "pixelate":
            # 픽셀화 (빠름, 효율적)
            pixel_size = self.blur_settings.pixelate_size
            small_w = max(1, face_w // pixel_size)
            small_h = max(1, face_h // pixel_size)
            # 작게 축소
            small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            # 원래 크기로 확대
            processed = cv2.resize(small, (face_w, face_h), interpolation=cv2.INTER_NEAREST)
            
        elif method == "mosaic":
            # 최적화: 모자이크 (ratio=0.05로 축소 후 확대, 매우 빠름)
            ratio = 0.05  # 1/20로 축소 (정보 날리기)
            small_w = max(1, int(face_w * ratio))
            small_h = max(1, int(face_h * ratio))
            # 아주 작게 축소 (예: 100x100 -> 5x5)
            small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
            # 원래 크기로 다시 확대 (모자이크 효과, 픽셀이 깨져 보임)
            processed = cv2.resize(small, (face_w, face_h), interpolation=cv2.INTER_NEAREST)
            
        elif method == "black":
            # 검은 박스 (가장 빠름)
            processed = np.zeros_like(face_region)
            
        elif method == "solid":
            # 단색 채우기 (빠름)
            # 평균 색상으로 채우기
            avg_color = np.mean(face_region, axis=(0, 1)).astype(np.uint8)
            processed = np.full_like(face_region, avg_color)
            
        else:
            # 기본값: 픽셀화
            pixel_size = self.blur_settings.pixelate_size
            small_w = max(1, face_w // pixel_size)
            small_h = max(1, face_h // pixel_size)
            small = cv2.resize(face_region, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
            processed = cv2.resize(small, (face_w, face_h), interpolation=cv2.INTER_NEAREST)
        
        frame[y1:y2, x1:x2] = processed
        return frame
    
    def _start_ffmpeg(self, width: int, height: int, fps: int):
        """FFmpeg 출력 시작 (지연 누적 방지 + 동적 FPS 최적화)"""
        # 인코딩 설정: CRF 기반 가변 비트레이트 (화질 우선) 또는 고정 비트레이트
        use_crf = settings.output_crf is not None and settings.output_crf > 0
        
        # GOP 크기 = FPS (1초 단위)
        gop_size = fps
        
        cmd = [
            settings.ffmpeg_path,
            '-y',
            # --- 입력 옵션 (-i 앞) ---
            '-fflags', 'nobuffer',  # 입력 버퍼 최소화
            '-use_wallclock_as_timestamps', '1',  # ⭐ 지연 누적 방지 핵심
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),  # 입력 FPS (참고용, 벽시계와 함께 사용)
            '-i', '-',  # 파이프 입력
            # --- 출력 옵션 (-i 뒤) ---
            '-c:v', settings.output_codec,  # libx264 또는 하드웨어 인코더
            '-preset', settings.output_preset,  # veryfast (화질과 속도 균형)
            '-tune', 'zerolatency',  # 지연 최소화
            '-r', str(fps),  # 출력 FPS 고정 (RTSP 플레이어를 위한 일정한 속도)
        ]
        
        # 비트레이트 또는 CRF 설정
        if use_crf:
            cmd.extend(['-crf', str(settings.output_crf)])  # 가변 비트레이트 (화질 우선)
        else:
            cmd.extend([
                '-b:v', settings.output_bitrate,
                '-maxrate', settings.output_bitrate,
                '-bufsize', '2000k',  # 0.5초 버퍼 (8000k → 2000k로 고정)
            ])
        
        cmd.extend([
            '-pix_fmt', 'yuv420p',
            '-g', str(gop_size),  # 1초 GOP (빠른 복구)
            '-x264-params', f'keyint={gop_size}:min-keyint={gop_size}:scenecut=0',  # 키프레임 최적화
            '-flags', 'low_delay',  # 낮은 지연 플래그
            '-strict', 'experimental',
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',  # RTSP 전송 방식: TCP
            '-rtsp_flags', 'prefer_tcp',  # TCP 우선 사용
            '-muxdelay', '0',  # 멀티플렉서 지연 제거
            self.output_url
        ])
        
        # 연결 시도 로그
        print(f"[{self.stream_id}] Flashphoner 연결 시도: {self.output_url} (FPS: {fps}, GOP: {gop_size})")
        
        # FFmpeg 실행 (에러 메시지 확인을 위해 stderr를 PIPE로 설정)
        self._ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,  # 에러 메시지 확인용
            bufsize=0,  # 버퍼링 없음
            start_new_session=True  # 별도 프로세스 그룹에서 실행 (서버 종료 방지)
        )
        self._ffmpeg_stderr = ""
        
        # stderr 읽기 스레드 시작 (에러 메시지 수집)
        def read_stderr():
            try:
                if self._ffmpeg_process and self._ffmpeg_process.stderr:
                    # H.264 디코딩 경고 필터링 (무시할 패턴)
                    ignore_patterns = [
                        'non-existing pps',
                        'decode_slice_header error',
                        'no frame!',
                        'out of range intra chroma pred mode',
                        'error while decoding mb',
                        'p sub_mb_type',
                        'top block unavailable',
                        '[h264 @',  # H.264 디코더 경고 전체
                    ]
                    
                    while True:
                        line = self._ffmpeg_process.stderr.readline()
                        if not line:
                            break
                        line_str = line.decode('utf-8', errors='ignore').strip()
                        if line_str:
                            self._ffmpeg_stderr += line_str + "\n"
                            line_lower = line_str.lower()
                            
                            # H.264 디코딩 경고는 무시 (스트림은 정상 작동)
                            if any(pattern in line_lower for pattern in ignore_patterns):
                                continue
                            
                            # RTSP 관련 메시지 출력 (디버깅용, 중요한 것만)
                            if 'rtsp' in line_lower and ('output' in line_lower or 'connected' in line_lower):
                                print(f"[{self.stream_id}] FFmpeg RTSP: {line_str}")
                            
                            # 실제 에러 키워드가 있으면 출력
                            if any(keyword in line_lower for keyword in ['error', 'failed', 'connection refused', 'timeout', 'unable', 'denied', 'forbidden', 'connection reset', 'cannot', 'unable to']):
                                # H.264 경고가 아닌 실제 에러만
                                if not any(pattern in line_lower for pattern in ignore_patterns):
                                    print(f"[{self.stream_id}] FFmpeg 에러: {line_str}")
            except Exception:
                pass
        
        stderr_thread = threading.Thread(target=read_stderr, daemon=True)
        stderr_thread.start()
    
    def _check_ffmpeg_alive(self) -> bool:
        """FFmpeg 프로세스 생존 확인"""
        if self._ffmpeg_process is None:
            return False
        
        ret = self._ffmpeg_process.poll()
        if ret is not None:
            # 프로세스 종료됨 - stderr 읽기 시도
            error_msg = self._get_ffmpeg_error()
            print(f"[{self.stream_id}] FFmpeg 종료 (code={ret})")
            if error_msg:
                print(f"[{self.stream_id}] FFmpeg 에러: {error_msg[:500]}")
            return False
        return True

    def _ffmpeg_is_alive(self) -> bool:
        """FFmpeg 생존 여부 (로그/재시작 없이 확인용)"""
        if self._ffmpeg_process is None:
            return False
        return self._ffmpeg_process.poll() is None
    
    def _get_ffmpeg_error(self) -> str:
        """FFmpeg stderr에서 에러 메시지 읽기"""
        if self._ffmpeg_process is None or self._ffmpeg_process.stderr is None:
            return self._ffmpeg_stderr or ""
        
        try:
            # Unix: select로 non-blocking 읽기
            if hasattr(select, 'select') and hasattr(select.select, '__call__'):
                try:
                    ready, _, _ = select.select([self._ffmpeg_process.stderr], [], [], 0.1)
                    if ready:
                        error = self._ffmpeg_process.stderr.read(2000)
                        if error:
                            error_str = error.decode('utf-8', errors='ignore') if isinstance(error, bytes) else error
                            if not self._ffmpeg_stderr:
                                self._ffmpeg_stderr = ""
                            self._ffmpeg_stderr += error_str
                            return self._ffmpeg_stderr
                except (OSError, ValueError):
                    # Windows에서는 select가 파이프에서 작동하지 않음
                    pass
        except:
            pass
        
        # Windows 또는 select 실패 시: 이미 읽은 내용이 있으면 반환
        return self._ffmpeg_stderr or ""
    
    def _restart_ffmpeg(self, width: int, height: int, fps: int):
        """FFmpeg 재시작 (빠른 종료)"""
        if self._ffmpeg_process:
            try:
                # stdin 닫기
                if self._ffmpeg_process.stdin:
                    try:
                        self._ffmpeg_process.stdin.close()
                    except:
                        pass
                # 즉시 kill (wait 없음)
                self._ffmpeg_process.kill()
            except:
                pass
        self._start_ffmpeg(width, height, fps)
        print(f"[{self.stream_id}] FFmpeg 재시작됨")
    
    def _cleanup_fast(self):
        """빠른 리소스 정리 (타임아웃 최소화)"""
        import os
        
        # VideoCapture 정리
        if self._cap:
            try:
                self._cap.release()
            except:
                pass
            self._cap = None
        
        # FFmpeg 정리 (즉시 kill, wait 없음)
        if self._ffmpeg_process:
            try:
                # stdin 닫기
                if self._ffmpeg_process.stdin:
                    try:
                        self._ffmpeg_process.stdin.close()
                    except:
                        pass
                
                # 프로세스 그룹 종료 (Unix) - 별도 프로세스 그룹에서 실행되므로 안전
                if hasattr(os, 'killpg'):
                    try:
                        pgid = os.getpgid(self._ffmpeg_process.pid)
                        # 서버 프로세스 그룹과 다를 때만 killpg 사용 (안전성 강화)
                        if pgid != os.getpgid(os.getpid()):
                            os.killpg(pgid, 9)  # SIGKILL
                    except (OSError, ProcessLookupError):
                        pass
                
                # 즉시 kill (wait 없음)
                try:
                    self._ffmpeg_process.kill()
                except:
                    pass
            except Exception as e:
                print(f"[{self.stream_id}] FFmpeg 정리 오류: {e}")
            finally:
                self._ffmpeg_process = None
        
        self._tracker.reset()
    
    def _cleanup(self):
        """리소스 정리 (강화) - _process_loop에서 사용"""
        import os
        
        # VideoCapture 정리
        if self._cap:
            try:
                self._cap.release()
            except:
                pass
            self._cap = None
        
        # FFmpeg 정리 (강제 종료 포함)
        if self._ffmpeg_process:
            try:
                # stdin 닫기
                if self._ffmpeg_process.stdin:
                    try:
                        self._ffmpeg_process.stdin.close()
                    except:
                        pass
                
                # 프로세스 그룹 종료 (Unix) - 별도 프로세스 그룹에서 실행되므로 안전
                if hasattr(os, 'killpg'):
                    try:
                        pgid = os.getpgid(self._ffmpeg_process.pid)
                        # 서버 프로세스 그룹과 다를 때만 killpg 사용 (안전성 강화)
                        if pgid != os.getpgid(os.getpid()):
                            os.killpg(pgid, 15)  # SIGTERM
                    except (OSError, ProcessLookupError):
                        pass
                
                # 종료 대기
                self._ffmpeg_process.terminate()
                try:
                    self._ffmpeg_process.wait(timeout=1)
                except:
                    self._ffmpeg_process.kill()
                    try:
                        self._ffmpeg_process.wait(timeout=0.5)
                    except:
                        pass
            except Exception as e:
                print(f"[{self.stream_id}] FFmpeg 정리 오류: {e}")
            self._ffmpeg_process = None
        
        self._tracker.reset()
