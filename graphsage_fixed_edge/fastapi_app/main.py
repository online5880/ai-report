from fastapi import FastAPI
from .routers import recommend
from fastapi.middleware.cors import CORSMiddleware

# FastAPI 앱 초기화
app = FastAPI(
    title="GraphSAGE API",
    docs_url="/api/graphsage/docs",  # Swagger UI 경로
    openapi_url="/api/graphsage/openapi.json",  # OpenAPI 경로
)

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (보안상 필요하면 특정 도메인만 허용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 라우터 등록
app.include_router(
    recommend.router, prefix="/api/graphsage/recommend", tags=["Recommend"]
)


# 기본 라우트
@app.get("/api/graphsage/")
def read_root():
    return {"message": "Welcome to GraphSAGE API"}


@app.get("/")
def root():
    return {"message": "Welcome to the root of the API"}
