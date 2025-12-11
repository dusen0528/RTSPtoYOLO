# 🎭 RTSP 얼굴 블러 처리 서버

RTSP 스트림에서 얼굴을 감지하고 블러 처리하여 Flashphoner로 재전송하는 FastAPI 서버입니다.

## 📦 설치

```bash
# 프로젝트 루트에서
cd yolo

# uv로 의존성 설치
uv sync

# 또는 pip으로 설치
pip install -e ..
```

## 🚀 서버 실행

```bash
# 방법 1: Python 직접 실행
cd yolo
python run_server.py

# 방법 2: uvicorn으로 실행
cd yolo
uvicorn server.main:app --host 0.0.0.0 --port 8000

# 방법 3: 프로젝트 루트에서
python -m yolo.server.main
```

서버가 시작되면 http://localhost:8000 에서 관리 페이지에 접속할 수 있습니다.

## 📡 API 엔드포인트

### 스트림 관리

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/streams` | 새 스트림 생성 |
| GET | `/api/streams` | 모든 스트림 목록 |
| GET | `/api/streams/{id}` | 스트림 정보 조회 |
| PUT | `/api/streams/{id}` | 스트림 설정 수정 |
| DELETE | `/api/streams/{id}` | 스트림 삭제 |

### 스트림 제어

| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/streams/{id}/start` | 스트림 시작 |
| POST | `/api/streams/{id}/stop` | 스트림 중지 |

### 모니터링

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/stats` | 서버 전체 통계 |
| GET | `/api/health` | 헬스 체크 |

## 📝 사용 예시

### 스트림 추가 (curl)

```bash
curl -X POST http://localhost:8000/api/streams \
  -H "Content-Type: application/json" \
  -d '{
    "name": "카메라1",
    "input_url": "rtsp://user:pass@192.168.1.100:554/stream",
    "output_url": "rtsp://flashphoner:1935/live/camera1",
    "blur_settings": {
      "confidence_threshold": 0.15,
      "blur_strength": 31,
      "imgsz": 320,
      "max_age": 25,
      "smoothing": 0.5
    }
  }'
```

### 스트림 시작

```bash
curl -X POST http://localhost:8000/api/streams/{stream_id}/start
```

### 서버 통계 조회

```bash
curl http://localhost:8000/api/stats
```

## ⚙️ 환경 변수

`.env` 파일을 생성하여 설정을 변경할 수 있습니다:

```env
BLUR_HOST=0.0.0.0
BLUR_PORT=8000
BLUR_MODEL_PATH=yolov8n-face.pt
BLUR_DEFAULT_CONFIDENCE=0.15
BLUR_DEFAULT_BLUR_STRENGTH=31
BLUR_DEFAULT_IMGSZ=320
BLUR_OUTPUT_FPS=15
BLUR_OUTPUT_BITRATE=1500k
```

## 🎛️ 블러 설정 가이드

| 설정 | 권장값 | 설명 |
|------|--------|------|
| `confidence_threshold` | 0.15 | 낮을수록 민감 (사람 같으면 블러) |
| `blur_strength` | 31 | 홀수, 클수록 강한 블러 |
| `imgsz` | 320 | 작을수록 빠름 (320/480/640) |
| `max_age` | 25 | 감지 실패 시 블러 유지 프레임 수 |
| `smoothing` | 0.5 | 박스 떨림 방지 (0~1) |

## 📋 요구사항

- Python 3.10+
- FFmpeg (RTSP 출력용)
- YOLOv8n-face 모델 (자동 다운로드)

### FFmpeg 설치

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg

# macOS
brew install ffmpeg
```

## 🏗️ 프로젝트 구조

```
yolo/
├── server/
│   ├── __init__.py
│   ├── main.py           # FastAPI 앱 + 관리 페이지
│   ├── config.py         # 설정
│   ├── models.py         # Pydantic 모델
│   ├── stream_manager.py # 스트림 관리
│   └── stream_processor.py # YOLO + FFmpeg 처리
├── run_server.py         # 실행 스크립트
├── detect.ipynb          # 테스트 노트북
└── README.md
```

## 📊 성능 참고

| 사양 | 동시 스트림 (예상) |
|------|-------------------|
| CPU 8코어, 16GB RAM | 3~5개 |
| CPU 16코어, 32GB RAM | 6~10개 |
| GPU RTX 3060 | 15~25개 |
| GPU RTX 4090 | 40~60개 |

## 🔧 트러블슈팅

### FFmpeg 연결 실패
- Flashphoner RTSP 서버가 실행 중인지 확인
- 출력 URL이 올바른지 확인
- 방화벽 설정 확인

### 높은 CPU 사용률
- `imgsz`를 320으로 낮추기
- `confidence_threshold`를 높이기 (0.2~0.3)
- 스트림 수 줄이기

### 블러가 자주 풀림
- `max_age` 값 증가 (30~50)
- `smoothing` 값 증가 (0.7~0.8)
- `confidence_threshold` 낮추기 (0.1)

