from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import polars as pl
import time
from .model_utils import load_model, predict_model
from .AWS_utils import get_rds_data
from .data_utils import prepare_data
from .config import DB_CONFIG, BASE_DIR
import sys
import os

sys.path.insert(0, os.path.join(BASE_DIR, "model"))

# 모델 및 데이터 로드
# data = get_rds_data(db_config = DB_CONFIG)
model = load_model()

# FastAPI 인스턴스 생성
app = FastAPI(docs_url="/docs", openapi_url="/api/gkt/openapi.json")

@app.get("/health")
def health_check():
    return {"status": "ok"}

# 입력 데이터 모델 정의
class InputData(BaseModel):
    user_id: str
    skill_list: list
    correct_list: list

@app.post("/api/gkt")
async def predict(input_data: InputData):
    try:
        start_time = time.time()

        # 유저 데이터 필터링
        user_data = get_rds_data(db_config = DB_CONFIG, user_id = input_data.user_id)
        user_data = user_data.sort("cre_date")

        # 스킬 매핑
        skill_map = user_data["f_mchapter_id"].unique().sort().to_list()
        skill_map_dict = {value: idx for idx, value in enumerate(skill_map)}

        user_data = user_data.with_columns(
            pl.col("f_mchapter_id")
            .replace(skill_map_dict)
            .cast(pl.Int32)
            .alias("skill")
        )
        user_data = user_data.with_columns(
            pl.col("correct").replace({"O": 1, "X": 0}).cast(pl.Int32).alias("correct")
        )
        user_data = user_data.with_columns(
            (pl.col("skill") * 2 + pl.col("correct")).alias("skill_with_answer")
        )

        # 입력 데이터 변환
        next_skills = [skill_map_dict.get(skill, -1) for skill in input_data.skill_list]
        if -1 in next_skills:
            raise HTTPException(
                status_code=400,
                detail="One or more skills in skill_list are not present in the data.",
            )

        features_tensor, questions_tensor = prepare_data(user_data, next_skills, input_data)

        # 모델 예측
        next_preds = predict_model(model, features_tensor, questions_tensor, next_skills)

        # 결과 생성
        result = [
            {skill: next_preds[idx].item()}
            for idx, skill in enumerate(input_data.skill_list)
        ]

        print(f"전체 소요 시간: {time.time() - start_time:.4f} 초")
        return {"predictions": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/gkt/confusion-matrix")
async def get_confusion_matrix(input_data: InputData):
    try:
        threshold = 0.5
        user_data = get_rds_data(db_config = DB_CONFIG, user_id = input_data.user_id)
        user_data = user_data.sort("cre_date")

        # 스킬 매핑
        skill_map = user_data["f_mchapter_id"].unique().sort().to_list()
        skill_map_dict = {value: idx for idx, value in enumerate(skill_map)}

        user_data = user_data.with_columns(
            pl.col("f_mchapter_id")
            .replace(skill_map_dict)
            .cast(pl.Int32)
            .alias("skill")
        )
        user_data = user_data.with_columns(
            pl.col("correct").replace({"O": 1, "X": 0}).cast(pl.Int32).alias("correct")
        )
        user_data = user_data.with_columns(
            (pl.col("skill") * 2 + pl.col("correct")).alias("skill_with_answer")
        )

        next_skills = [skill_map_dict.get(skill, -1) for skill in input_data.skill_list]
        if -1 in next_skills:
            raise HTTPException(
                status_code=400,
                detail="One or more skills in skill_list are not present in the data.",
            )

        features_tensor, questions_tensor = prepare_data(user_data, next_skills, input_data)

        next_preds = predict_model(model, features_tensor, questions_tensor, next_skills)
        confusion_results = []

        for i, pred in enumerate(next_preds.tolist()):
            pred_result = 1 if pred >= threshold else 0
            if pred_result == input_data.correct_list[i]:
                if pred_result == 1:
                    analysis = "개념 확립 (정답 확신)"
                else:
                    analysis = "개념 확립 (오답 확신)"
            else:
                if pred_result == 1:
                    analysis = "실수 (과신)"
                else:
                    analysis = "찍음 (운 좋게 맞춤)"

            confusion_results.append({
                "skill": input_data.skill_list[i],
                "predicted_probability": pred,
                "predicted_result": pred_result,
                "actual_result": input_data.correct_list[i],
                "analysis": analysis
            })

        return {"confusion_matrix": confusion_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))