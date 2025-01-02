# Base image: 더 작은 이미지 사용 (slim 버전)
FROM python:3.8-slim

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /code

# 폴더가 없으면 생성하도록 명시적으로 추가
RUN mkdir -p /code/app

# 종속성 파일 복사 및 설치
COPY ./requirements.txt /code/requirements.txt

# 종속성 설치 (빌드 캐시 최소화)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /code/requirements.txt && \
    apt-get remove -y gcc && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 애플리케이션 파일 복사
COPY ./app /code/app

# 실행 커맨드
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--proxy-headers"]
