# 필요한 라이브러리 임포트
from fastapi import FastAPI  # FastAPI 애플리케이션 생성 및 라우팅을 위해 필요
from pydantic import BaseModel  # 데이터 검증 및 타입 힌트를 제공하는 모델 생성
from typing import List  # Python 3.8에서는 typing 모듈의 List를 사용해야 함

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI()


# Health Check 엔드포인트 정의
@app.get("/health")  # HTTP GET 요청을 처리
def health_check():
    """
    Health Check 엔드포인트.
    서버가 정상적으로 작동하는지 확인하기 위해 사용.
    """
    return {"status": "ok"}  # 서버 상태를 JSON 형식으로 반환


# 요청 본문에 사용할 데이터 모델 정의
class RequestBody(BaseModel):
    """
    요청 본문에서 기대하는 데이터 구조 정의.

    - `user_data`: 문자열로 구성된 리스트. 사용자 데이터를 포함.
    """

    user_data: List[str]  # 문자열 리스트를 타입으로 지정


# POST 요청을 처리하는 API 엔드포인트 정의
@app.post("/api/gkt")  # HTTP POST 요청을 처리
def post_data(request_body: RequestBody):
    """
    POST 요청을 처리하는 엔드포인트.

    - 요청 본문에 포함된 `user_data` 리스트를 처리.
    - 요청 데이터는 RequestBody 모델에 의해 자동 검증됨.
    """
    # ! 함수 내부 구현

    return {"received_data": request_body.user_data}  # 요청에서 받은 데이터를 그대로 반환
