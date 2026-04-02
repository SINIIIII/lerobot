# 1. CUDA devel 이미지 사용: Flash Attention 등 커스텀 CUDA 커널 빌드를 위해 필수적입니다.
# 최신 PyTorch와의 안정적인 호환성을 위해 12.4 버전을 추천합니다.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# 설치 중 사용자 입력 대기로 인한 빌드 멈춤 방지
ENV DEBIAN_FRONTEND=noninteractive

# 2. 필수 패키지 및 LeRobot/비전 처리에 필요한 시스템 라이브러리 추가
# EGL 헤더, CMake, OpenCV 관련 패키지 등을 미리 설치하여 의존성 에러를 방지합니다.
RUN apt-get update && apt-get install -y \
    curl git build-essential cmake \
    software-properties-common \
    ffmpeg libsm6 libxext6 libgl1-mesa-glx libegl1-mesa-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
#RUN curl -LsSf https://astral.sh/uv/install.sh | sh
#ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /workspace

# 4. Python 3.12 가상환경 생성 및 활성화
RUN uv venv --python 3.12 /opt/env
ENV VIRTUAL_ENV="/opt/env"
ENV PATH="/opt/env/bin:${PATH}"

# requirements 복사
COPY requirements.txt .

# 5. PyTorch 및 의존성 설치
# RUN uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install -r requirements.txt


COPY . .
RUN uv pip install -e .
RUN uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# 6. Hugging Face 데이터셋 다운로드 등을 위한 환경 변수 (실행 시 --env로 넘겨주는 것을 권장)
# ENV HF_TOKEN=""

CMD ["/bin/bash"]