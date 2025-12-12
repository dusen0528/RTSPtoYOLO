#!/usr/bin/env python3
"""
RTSP 얼굴 블러 서버 실행 스크립트
"""
import sys
import os

# 현재 디렉토리(yolo 폴더)를 path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# server 패키지로 직접 import (yolo 폴더가 루트인 경우)
from server.main import run_server

if __name__ == "__main__":
    run_server()

