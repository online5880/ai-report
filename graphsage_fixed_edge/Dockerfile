# Base image: slim 버전 사용
FROM python:3.8-slim

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 비루트 사용자 추가
RUN adduser --disabled-password --gecos '' appuser

# 작업 디렉토리 설정
WORKDIR /app

# 종속성 파일 복사 및 설치 (빌드 캐시 활용 극대화)
COPY ./requirements.txt /app/requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    pip install --no-cache-dir torch==2.4.1 \
        --extra-index-url https://download.pytorch.org/whl/cpu && \
    apt-get remove -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 애플리케이션 파일 복사
COPY . /app

# 권한 변경
RUN chown -R appuser:appuser /app

# 비루트 사용자로 전환
USER appuser

# 실행 커맨드
CMD ["uvicorn", "fastapi_app.main:app", "--host", "0.0.0.0", "--port", "8200", "--proxy-headers"]
