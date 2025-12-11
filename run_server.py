#!/usr/bin/env python3
"""
RTSP 얼굴 블러 서버 실행 스크립트
"""
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yolo.server.main import run_server

if __name__ == "__main__":
    run_server()

