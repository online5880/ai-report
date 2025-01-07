from fastapi import FastAPI
from .routers import recommend
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="GraphSAGE API")

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용 (보안상 필요하면 특정 도메인만 허용)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
)

# 라우터 등록
app.include_router(recommend.router, prefix="/recommend", tags=["Recommend"])


@app.get("/")
def read_root():
    return {"message": "Welcome to GraphSAGE API"}
