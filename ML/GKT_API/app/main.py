from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import mlflow.pytorch
import os
import polars as pl
import sys
import boto3
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "model"))

print("Current sys.path:", sys.path)


# try:
#     import models

#     print("모듈이 성공적으로 로드되었습니다.")
# except ImportError as e:
#     print(f"모듈 로드 실패: {e}")

# import app.models.models


# MLflow Tracking URI 설정
mlflow.set_tracking_uri(
    os.getenv("MLFLOW_SERVER_URI", "http://bigdata9:bigdata9-@mane.my/mlflow/")
)


def get_csv_file(local_path, s3_bucket_name, s3_key):

    if os.path.exists(local_path):
        print(f"로컬에서 파일을 읽습니다: {local_path}")
        return pl.read_csv(local_path)

    print(f"로컬 파일이 없습니다. S3에서 파일을 다운로드합니다: {s3_bucket_name}/{s3_key}")
    try:
        # S3 클라이언트 생성
        s3 = boto3.client("s3")

        # 파일 다운로드
        s3.download_file(s3_bucket_name, s3_key, local_path)
        print(f"파일이 성공적으로 다운로드되었습니다: {local_path}")

        # CSV 파일 읽기
        return pl.read_csv(local_path)

    except Exception as e:
        raise RuntimeError(f"S3에서 파일을 다운로드하는 중 에러가 발생했습니다: {e}")


def get_parquet_file(local_path, s3_bucket_name, s3_key):
    """
    Parquet 파일을 로컬에서 읽거나 S3에서 다운로드 후 읽는 함수.

    Args:
        local_path (str): 로컬 파일 경로.
        s3_bucket_name (str): S3 버킷 이름.
        s3_key (str): S3에서 파일의 경로.

    Returns:
        pl.DataFrame: Polars DataFrame.
    """
    # 로컬 파일이 있는 경우 로드
    if os.path.exists(local_path):
        print(f"로컬에서 파일을 읽습니다: {local_path}")
        return pl.read_parquet(local_path)

    # 로컬 파일이 없는 경우 S3에서 다운로드
    print(f"로컬 파일이 없습니다. S3에서 파일을 다운로드합니다: {s3_bucket_name}/{s3_key}")
    try:
        # S3 클라이언트 생성
        s3 = boto3.client("s3")

        # S3에서 파일 다운로드
        s3.download_file(s3_bucket_name, s3_key, local_path)
        print(f"파일이 성공적으로 다운로드되었습니다: {local_path}")

        # Parquet 파일 읽기
        return pl.read_parquet(local_path)

    except Exception as e:
        raise RuntimeError(f"S3에서 파일을 다운로드하는 중 에러가 발생했습니다: {e}")


# 예제 사용
local_csv_path = "/code/app/filtered_combined_user_data.csv"
local_parquet_path = "/code/app/filtered_combined_user_data.parquet"
s3_bucket = "big9-project-01"
s3_file_key = "data/tbl_app_testhisdtl/filtered_combined_user_data.parquet"

try:
    # data = get_csv_file(local_csv_path, s3_bucket, s3_file_key)
    data = get_parquet_file(local_parquet_path, s3_bucket, s3_file_key)
    print("데이터가 성공적으로 로드되었습니다.")
except RuntimeError as e:
    print(f"에러: {e}")


# 모델 로드
def load_model():
    logged_model = "runs:/446b1a8e75ff4263a59f168a5605ba90/best_model"
    model = mlflow.pytorch.load_model(logged_model, map_location=torch.device("cpu"))
    model.eval()
    return model


model = load_model()

# FastAPI 애플리케이션 인스턴스 생성
app = FastAPI(
    docs_url="/docs",  # Swagger UI 경로
    openapi_url="/api/gkt/openapi.json",  # OpenAPI 스펙 경로
)


# Health Check 엔드포인트 정의
@app.get("/health")  # HTTP GET 요청을 처리
def health_check():
    """
    Health Check 엔드포인트.
    서버가 정상적으로 작동하는지 확인하기 위해 사용.
    """
    return {"status": "ok"}  # 서버 상태를 JSON 형식으로 반환


# 입력 데이터 모델 정의
class InputData(BaseModel):
    user_id: str
    skill_list: list
    correct_list: list


@app.post("/api/gkt")
async def predict(input_data: InputData):
    try:
        start_time = time.time()  # 전체 프로세스 시작 시간

        # Step 1 - 유저 데이터 필터링
        user_id = input_data.user_id
        step1_start = time.time()
        user_data = data.filter(pl.col("UserID") == user_id)
        step1_end = time.time()
        print(f"Step 1 (유저 데이터 필터링): {step1_end - step1_start:.4f} 초")

        # Step 2 - 데이터 정렬
        step2_start = time.time()
        user_data = user_data.sort("CreDate")  # "CreDate" 기준 정렬
        step2_end = time.time()
        print(f"Step 2 (데이터 정렬): {step2_end - step2_start:.4f} 초")

        # Step 3 - Enumerate skill id
        step3_start = time.time()
        skill_map = user_data["f_mchapter_id"].unique().sort().to_list()
        skill_map_dict = {value: idx for idx, value in enumerate(skill_map)}
        user_data = user_data.with_columns(
            pl.col("f_mchapter_id")
            .replace(skill_map_dict)
            .cast(pl.Int32)
            .alias("skill")
        )
        step3_end = time.time()
        print(f"Step 3 (스킬 매핑): {step3_end - step3_start:.4f} 초")

        # Step 4 - Correct 변환
        step4_start = time.time()
        user_data = user_data.with_columns(
            pl.col("Correct").replace({"O": 1, "X": 0}).cast(pl.Int32).alias("Correct")
        )
        step4_end = time.time()
        print(f"Step 4 (Correct 변환): {step4_end - step4_start:.4f} 초")

        # Step 5 - Synthetic Feature 생성
        step5_start = time.time()
        user_data = user_data.with_columns(
            (pl.col("skill") * 2 + pl.col("Correct")).alias("skill_with_answer")
        )
        step5_end = time.time()
        print(f"Step 5 (Synthetic Feature 생성): {step5_end - step5_start:.4f} 초")

        # Step 6 - 입력 스킬 매핑
        step6_start = time.time()
        next_skills = [skill_map_dict.get(skill, -1) for skill in input_data.skill_list]
        if -1 in next_skills:
            raise HTTPException(
                status_code=400,
                detail="One or more skills in skill_list are not present in the data.",
            )
        step6_end = time.time()
        print(f"Step 6 (입력 스킬 매핑): {step6_end - step6_start:.4f} 초")

        # Step 7 - 모델 예측
        step7_start = time.time()
        features = user_data["skill_with_answer"].to_list()
        questions = user_data["skill"].to_list()

        for i in range(len(next_skills)):
            features.append(next_skills[i] * 2 + input_data.correct_list[i])
            questions.append(next_skills[i])

        features_tensor = torch.tensor(features, dtype=torch.long).unsqueeze(0)
        questions_tensor = torch.tensor(questions, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            pred_res, _, _, _ = model(features_tensor, questions_tensor)
            next_preds = pred_res.squeeze(0)[-len(next_skills) :]
        step7_end = time.time()
        print(f"Step 7 (모델 예측): {step7_end - step7_start:.4f} 초")

        # 결과 생성
        step8_start = time.time()
        result = [
            {skill: next_preds[idx].item()}
            for idx, skill in enumerate(input_data.skill_list)
        ]
        step8_end = time.time()
        print(f"Step 8 (결과 생성): {step8_end - step8_start:.4f} 초")

        # 전체 소요 시간
        end_time = time.time()
        print(f"전체 소요 시간: {end_time - start_time:.4f} 초")

        return {"predictions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
