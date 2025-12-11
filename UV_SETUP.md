# UV 설치 및 설정 가이드

이 프로젝트는 `uv`를 사용하여 Python 패키지를 관리합니다.

## 📦 UV 설치

### Windows

#### 방법 1: PowerShell (권장)
```powershell
# PowerShell에서 실행
irm https://astral.sh/uv/install.ps1 | iex
```

#### 방법 2: pip로 설치
```bash
pip install uv
```

#### 방법 3: 직접 다운로드
1. [uv 릴리스 페이지](https://github.com/astral-sh/uv/releases)에서 최신 버전 다운로드
2. `uv.exe`를 PATH에 추가

### Linux / macOS

```bash
# Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# macOS
brew install uv
```

## 🐍 Python 설치

### Windows

1. **Python 공식 사이트에서 설치**
   - [python.org/downloads](https://www.python.org/downloads/)에서 Python 3.11 다운로드
   - 설치 시 "Add Python to PATH" 옵션 체크

2. **Microsoft Store에서 설치**
   ```bash
   # Microsoft Store 앱에서 "Python 3.11" 검색 후 설치
   ```

3. **Chocolatey 사용 (관리자 권한 필요)**
   ```powershell
   choco install python311
   ```

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev

# CentOS/RHEL
sudo yum install python311 python311-pip
```

### macOS

```bash
# Homebrew 사용
brew install python@3.11
```

## 🚀 프로젝트 설정

### 1. UV로 프로젝트 초기화 및 의존성 설치

```bash
# 프로젝트 루트 디렉토리에서 실행
cd yolo

# UV로 의존성 설치 (가상환경 자동 생성)
uv sync

# 또는 개발 의존성 포함
uv sync --dev
```

### 2. 가상환경 활성화

UV는 자동으로 가상환경을 생성하고 관리합니다.

#### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows (CMD)
```cmd
.\.venv\Scripts\activate.bat
```

#### Linux / macOS
```bash
source .venv/bin/activate
```

### 3. 서버 실행

```bash
# 방법 1: UV로 직접 실행
uv run python run_server.py

# 방법 2: 가상환경 활성화 후 실행
python run_server.py

# 방법 3: uvicorn으로 실행
uv run uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## 📋 주요 UV 명령어

```bash
# 의존성 설치
uv sync

# 새 패키지 추가
uv add 패키지명

# 개발 의존성 추가
uv add --dev 패키지명

# 패키지 제거
uv remove 패키지명

# 의존성 업데이트
uv sync --upgrade

# Python 버전 확인
uv python list

# 특정 Python 버전 설치
uv python install 3.11
```

## 🔍 문제 해결

### UV가 설치되지 않는 경우

1. **PowerShell 실행 정책 확인**
   ```powershell
   Get-ExecutionPolicy
   # Restricted인 경우:
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

2. **수동 설치**
   ```bash
   pip install uv
   ```

### Python이 인식되지 않는 경우

1. **Python 버전 확인**
   ```bash
   python --version
   # 또는
   python3 --version
   ```

2. **UV로 Python 설치**
   ```bash
   uv python install 3.11
   ```

3. **PATH 환경 변수 확인**
   - Windows: 시스템 속성 > 환경 변수 > Path에 Python 경로 추가
   - Linux/macOS: `~/.bashrc` 또는 `~/.zshrc`에 PATH 추가

### 의존성 설치 오류

1. **캐시 정리**
   ```bash
   uv cache clean
   ```

2. **가상환경 재생성**
   ```bash
   rm -rf .venv
   uv sync
   ```

## 📚 추가 자료

- [UV 공식 문서](https://docs.astral.sh/uv/)
- [UV GitHub](https://github.com/astral-sh/uv)
- [Python 공식 사이트](https://www.python.org/)

