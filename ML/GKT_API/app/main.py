import mlflow.pyfunc
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class GKTModelWrapper:
    def __init__(self, model_path):
        # 모델 로드 및 환경 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()

    def predict(self, model_input):
        # 입력 데이터 처리
        features = torch.tensor(model_input["features"], dtype=torch.long).unsqueeze(0).to(self.device)
        questions = torch.tensor(model_input["questions"], dtype=torch.long).unsqueeze(0).to(self.device)
        next_skills = torch.tensor(model_input["next_skills"], dtype=torch.long).unsqueeze(0).to(self.device)
        next_answers = model_input["next_answers"]

        # 다음 문제의 feature 생성
        next_features = [
            skill * 2 + answer for skill, answer in zip(model_input["next_skills"], next_answers)
        ]
        next_features_tensor = torch.tensor(next_features, dtype=torch.long).unsqueeze(0).to(self.device)

        # 모델 추론
        with torch.no_grad():
            # 이전 풀이 기록으로 상태 업데이트
            pred_res, _, _, _ = self.model(features, questions)

            # 다음 문제에 대해 모델 예측 수행
            pred_res_next, _, _, _ = self.model(next_features_tensor, next_skills)

        # 결과 비교 및 반환
        predictions = [
            {
                "skill": skill,
                "predicted": float(pred),
                "actual": actual
            }
            for skill, pred, actual in zip(
                model_input["next_skills"], pred_res_next.squeeze(0).tolist(), next_answers
            )
        ]
        return {"predictions": predictions}

# MLflow 모델 저장 (주석 처리)
# mlflow.pyfunc.save_model(
#     path="gkt_model",
#     python_model=GKTModelWrapper(),
#     artifacts={"model_path": "path/to/saved_model.pth"},
# )

# 요청 본문에 사용할 데이터 모델 정의
class RequestBody(BaseModel):
    features: List[int]  # feature 벡터
    questions: List[int]  # 질문 ID 리스트
    next_skills: List[int]  # 다음 문제의 스킬 리스트
    next_answers: List[int]  # 다음 문제의 실제 정답 리스트

model = GKTModelWrapper(model_path="app/model.pth")

# POST 요청을 처리하는 API 엔드포인트 정의
@app.post("/api/gkt")
def post_data(request_body: RequestBody):
    # GKTModelWrapper 인스턴스 생성 (로컬 모델 사용)
    # 예측 수행
    predictions = model.predict(request_body.dict())

    return {"predictions": predictions}
