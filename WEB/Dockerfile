# Python 베이스 이미지 사용
FROM python:3.8

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 파일 복사
COPY ./app /app

# Django 애플리케이션 실행 (Gunicorn 사용)
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app.wsgi:application"]
