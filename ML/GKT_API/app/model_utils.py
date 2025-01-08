import torch
import os
import sys
from .config import BASE_DIR, MODEL_PATH

sys.path.insert(0, os.path.join(BASE_DIR, "model"))


def load_model():
    """
    MLflow에서 모델을 로드하는 함수.
    Returns:
        torch.nn.Module: 로드된 PyTorch 모델.
    """
    # logged_model = "runs:/446b1a8e75ff4263a59f168a5605ba90/best_model"
    # model = mlflow.pytorch.load_model(logged_model, map_location=torch.device("cpu"))
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
    return model


def predict_model(active_model, features_tensor, questions_tensor, next_skills):
    """
    모델 예측을 수행하는 함수.
    Args:
        active_model (torch.nn.Module): 예측에 사용할 모델.
        features_tensor (torch.Tensor): 입력 피처 텐서.
        questions_tensor (torch.Tensor): 입력 질문 텐서.
        next_skills (list): 다음 스킬 리스트.
    Returns:
        torch.Tensor: 예측 결과.
    """
    with torch.no_grad():
        pred_res, _, _, _ = active_model(features_tensor, questions_tensor)
        next_preds = pred_res.squeeze(0)[-len(next_skills):]
    return next_preds
