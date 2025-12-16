"""
FastAPI 서버 - RTSP 얼굴 블러 처리
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from typing import List
from pathlib import Path
import uvicorn
import sys

from .models import (
    StreamCreate, StreamUpdate, StreamInfo, ServerStats,
    ApiResponse, BlurSettings
)
from fastapi import Query
from urllib.parse import urlparse
from .stream_manager import manager
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 처리"""
    # 시작 시
    print("서버 시작 중...")
    manager.initialize_model()
    print("서버 준비 완료!")
    yield
    # 종료 시
    print("서버 종료 중...")
    manager.shutdown()
    print("서버 종료 완료!")


app = FastAPI(
    title="RTSP 얼굴 블러 처리 서버",
    description="RTSP 스트림에서 얼굴을 감지하고 블러 처리하여 Flashphoner로 재전송합니다.",
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================
# 스트림 관리 API
# ============================================

@app.post("/api/streams", response_model=StreamInfo, tags=["Streams"])
async def create_stream(data: StreamCreate):
    """새 스트림 생성"""
    # 최대 동시 스트림 수 체크
    current_count = len(manager.get_all_streams())
    if current_count >= settings.max_concurrent_streams:
        raise HTTPException(
            status_code=400, 
            detail=f"최대 {settings.max_concurrent_streams}개 스트림까지 생성 가능합니다"
        )
    
    try:
        info = manager.create_stream(data)
        return info
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/streams", response_model=List[StreamInfo], tags=["Streams"])
async def list_streams():
    """모든 스트림 목록"""
    return manager.get_all_streams()


# 중요: /api/streams/by-input은 /api/streams/{stream_id}보다 먼저 정의해야 함!
# 그렇지 않으면 "by-input"이 stream_id로 인식됨
@app.get("/api/streams/by-input", tags=["Streams"])
async def get_output_url_by_input(
    input_url: str = Query(..., description="입력 RTSP URL")
):
    """입력 RTSP URL로 출력 RTSP URL 조회"""
    print(f"[API] /api/streams/by-input 호출됨, input_url: {input_url}", flush=True)
    sys.stdout.flush()
    
    info = manager.get_stream_by_input_url(input_url)
    
    if not info:
        print(f"[API] 스트림 없음, 자동 생성 시도: {input_url}", flush=True)
        sys.stdout.flush()
        try:
            generated = _generate_output_url(input_url)
            new_stream = StreamCreate(
                name=_derive_name_from_input(input_url),
                input_url=input_url.strip(),
                output_url=generated,
                allow_duplicate=True,
                blur_settings=BlurSettings()
            )
            info = manager.create_stream(new_stream)
            print(f"[API] 스트림 자동 생성 완료: {info.id}, output_url: {info.output_url}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"[API] 자동 생성 실패: {e}", flush=True)
            sys.stdout.flush()
            raise HTTPException(
                status_code=500,
                detail=f"입력 URL '{input_url}'로 스트림을 생성할 수 없습니다: {e}"
            )
    
    print(f"[API] 스트림 찾음: {info.id}, output_url: {info.output_url}", flush=True)
    sys.stdout.flush()
    
    # 출력 URL만 반환
    return {
        "input_url": info.input_url,
        "output_url": info.output_url,
        "stream_id": info.id,
        "name": info.name,
        "status": info.status.value
    }


def _generate_output_url(input_url: str) -> str:
    """입력 RTSP URL을 기반으로 블러 출력 URL 생성"""
    parsed = urlparse(input_url.strip())
    if not parsed.scheme.startswith("rtsp"):
        raise ValueError("RTSP URL이 아닙니다.")
    
    path = parsed.path or ""
    if not path:
        raise ValueError("RTSP URL에 경로가 없습니다.")
    
    if path.endswith("/"):
        path = path[:-1]
    # 마지막 세그먼트에 -blur 추가
    segments = path.split("/")
    last = segments[-1]
    if not last:
        raise ValueError("RTSP URL 경로가 비어 있습니다.")
    segments[-1] = f"{last}-blur"
    new_path = "/".join(segments)
    
    port_part = f":{parsed.port}" if parsed.port else ""
    return f"rtsp://{parsed.hostname}{port_part}{new_path}"


def _derive_name_from_input(input_url: str) -> str:
    """입력 URL에서 이름 생성"""
    parsed = urlparse(input_url.strip())
    path = parsed.path or ""
    if path.endswith("/"):
        path = path[:-1]
    if not path:
        return "auto-stream"
    last = path.split("/")[-1]
    return last or "auto-stream"


@app.get("/api/streams/{stream_id}", response_model=StreamInfo, tags=["Streams"])
async def get_stream(stream_id: str):
    """스트림 정보 조회"""
    info = manager.get_stream(stream_id)
    if not info:
        raise HTTPException(status_code=404, detail="스트림을 찾을 수 없습니다")
    return info


@app.put("/api/streams/{stream_id}", response_model=StreamInfo, tags=["Streams"])
async def update_stream(stream_id: str, data: StreamUpdate):
    """스트림 설정 업데이트"""
    info = manager.update_stream(stream_id, data)
    if not info:
        raise HTTPException(status_code=404, detail="스트림을 찾을 수 없습니다")
    return info


@app.delete("/api/streams/{stream_id}", response_model=ApiResponse, tags=["Streams"])
async def delete_stream(stream_id: str):
    """스트림 삭제"""
    success = manager.delete_stream(stream_id)
    if not success:
        raise HTTPException(status_code=404, detail="스트림을 찾을 수 없습니다")
    return ApiResponse(success=True, message="스트림이 삭제되었습니다")


# ============================================
# 스트림 제어 API
# ============================================

@app.post("/api/streams/{stream_id}/start", response_model=StreamInfo, tags=["Control"])
async def start_stream(stream_id: str):
    """스트림 시작"""
    info = manager.start_stream(stream_id)
    if not info:
        raise HTTPException(status_code=404, detail="스트림을 찾을 수 없습니다")
    return info


@app.post("/api/streams/{stream_id}/stop", response_model=StreamInfo, tags=["Control"])
async def stop_stream(stream_id: str):
    """스트림 중지"""
    info = manager.stop_stream(stream_id)
    if not info:
        raise HTTPException(status_code=404, detail="스트림을 찾을 수 없습니다")
    return info


# ============================================
# 모니터링 API
# ============================================

@app.get("/api/stats", response_model=ServerStats, tags=["Monitoring"])
async def get_server_stats():
    """서버 전체 통계"""
    return manager.get_server_stats()


@app.get("/api/health", tags=["Monitoring"])
async def health_check():
    """헬스 체크"""
    stats = manager.get_server_stats()
    return {
        "status": "healthy",
        "total_streams": stats.total_streams,
        "running_streams": stats.running_streams,
    }


# ============================================
# 기본 블러 설정 API
# ============================================

@app.get("/api/settings/default", response_model=BlurSettings, tags=["Settings"])
async def get_default_settings():
    """기본 블러 설정 조회"""
    return BlurSettings()


# ============================================
# 관리 페이지 (HTML)
# ============================================

@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def admin_page():
    """관리 페이지"""
    return get_admin_html()


# static 폴더 경로
STATIC_DIR = Path(__file__).parent / "static"


def get_admin_html() -> str:
    """관리 페이지 HTML (파일에서 로드)"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>index.html not found</h1>"


def run_server():
    """서버 실행 함수 (CLI용)"""
    uvicorn.run(
        "server.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


# 서버 실행
if __name__ == "__main__":
    run_server()

