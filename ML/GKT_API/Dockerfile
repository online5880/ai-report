# Base image
FROM python:3.8

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /code

# 종속성 파일 복사 및 설치
COPY ./requirements.txt /code/requirements.txt

# pip 업그레이드 및 requirements.txt 설치
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 애플리케이션 파일 복사
COPY ./app /code/app

# CMD 명령 설정
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--proxy-headers"]