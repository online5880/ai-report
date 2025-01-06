from fastapi import FastAPI
from fastapi_app.routers import recommend

app = FastAPI(title="GraphSAGE API")

# 라우터 등록
app.include_router(recommend.router, prefix="/recommend", tags=["Recommend"])

@app.get("/")
def read_root():
    return {"message": "Welcome to GraphSAGE API"}
