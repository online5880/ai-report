from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi_app.routers import recommend

app = FastAPI(title="GraphSAGE API")

app.include_router(recommend.router, prefix="/recommend", tags=["Recommend"])

@app.get("/")
def read_root():
    return {"message": "Welcome to GraphSAGE API"}

# ---------- 글로벌 에러 핸들러 ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail}},
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": {"message": f"Unexpected error: {str(exc)}"}},
    )