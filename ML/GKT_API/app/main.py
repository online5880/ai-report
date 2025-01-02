from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import mlflow.pytorch
import os
import polars as pl
import sys
import boto3


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


# 예제 사용
local_csv_path = "/code/app/filtered_combined_user_data.csv"
s3_bucket = "big9-project-01"
s3_file_key = "data/tbl_app_testhisdtl/filtered_combined_user_data.csv"

try:
    data = get_csv_file(local_csv_path, s3_bucket, s3_file_key)
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
app = FastAPI()


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
        user_id = input_data.user_id
        user_data = data.filter(pl.col("UserID") == user_id)

        # Step 0 - 정렬: 가장 오래된 기록부터 정렬
        user_data = user_data.sort(["UserID", "CreDate"])  # "CreDate" 컬럼을 기준으로 정렬

        # Step 2 - Enumerate skill id
        skill_map = user_data["f_mchapter_id"].unique().sort().to_list()
        skill_map_dict = {value: idx for idx, value in enumerate(skill_map)}
        user_data = user_data.with_columns(
            pl.col("f_mchapter_id")
            .replace(skill_map_dict)
            .cast(pl.Int32)
            .alias("skill")
        )

        # correct 생성 (O -> 1, X -> 0)
        user_data = user_data.with_columns(
            pl.col("Correct").replace({"O": 1, "X": 0}).cast(pl.Int32).alias("Correct")
        )

        # Step 3 - Cross skill id with answer to form a synthetic feature
        user_data = user_data.with_columns(
            (pl.col("skill") * 2 + pl.col("Correct")).alias("skill_with_answer")
        )

        # 팩토라이징되지 않은 입력 스킬을 기존 skill_map에 따라 팩토라이징
        next_skills = [
            skill_map_dict.get(skill, -1) for skill in input_data.skill_list
        ]  # -1은 미정의된 스킬

        # 매핑되지 않은 스킬이 있는지 확인
        if -1 in next_skills:
            raise HTTPException(
                status_code=400,
                detail="One or more skills in skill_list are not present in the data.",
            )

        # correct_list와 next_skills 확인
        next_answers = input_data.correct_list
        if len(next_skills) != len(next_answers):
            raise HTTPException(
                status_code=400,
                detail="skill_list and correct_list must have the same length.",
            )

        # 유저 풀이 시퀀스 및 다음 문제 정의
        features = user_data["skill_with_answer"].to_list()
        questions = user_data["skill"].to_list()

        for i in range(0, len(next_skills)):
            features.append(next_skills[i] * 2 + next_answers[i])
            questions.append(next_skills[i])

        features_tensor = torch.tensor(features, dtype=torch.long).unsqueeze(0)
        questions_tensor = torch.tensor(questions, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            pred_res, _, _, _ = model(
                features_tensor, questions_tensor
            )  # 입력값과 동일한 디바이스에서 수행
            next_preds = pred_res.squeeze(0)[-len(next_skills) :]  # 다음 문제에 해당하는 예측값

        result = []
        for idx, skill in enumerate(input_data.skill_list):
            result.append({skill: next_preds[idx].item()})

        # 예측 결과 반환
        return {"predictions": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
