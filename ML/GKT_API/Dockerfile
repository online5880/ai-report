FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8-alpine3.10-2024-01-29

# 환경 변수 설정
ENV PYTHONUNBUFFERED 1

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "1000"]

# If running behind a proxy like Nginx or Traefik add --proxy-headers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--proxy-headers"]