import os
import requests
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# 환경 변수 로드 (GitHub Actions에서 전달)
MLFLOW_SERVER_URI = os.getenv("MLFLOW_SERVER_URI")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
EXPERIMENT_NAME = "Iris_Classification_Experiment"
MODEL_NAME = "Iris_Classifier"


def send_slack_notification(status, message):
    """Slack 알림 전송"""
    if not SLACK_WEBHOOK_URL:
        print("Slack Webhook URL이 설정되지 않았습니다.")
        return

    payload = {"text": f"MLflow 작업 상태: {status}\n{message}"}
    response = requests.post(SLACK_WEBHOOK_URL, json=payload)
    if response.status_code == 200:
        print("Slack 알림 성공")
    else:
        print(f"Slack 알림 실패: {response.status_code}, {response.text}")


def train_model():
    """모델 학습 및 MLflow 로깅"""
    try:
        # MLflow 설정
        mlflow.set_tracking_uri(MLFLOW_SERVER_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        # 데이터 로드 및 학습
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.2
        )
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # MLflow 실행
        with mlflow.start_run() as run:
            # 서명 및 입력 예제 생성
            input_example = pd.DataFrame(X_test, columns=data.feature_names)
            signature = infer_signature(X_test, model.predict(X_test))

            # 모델 저장 및 메트릭 기록
            accuracy = model.score(X_test, y_test)
            mlflow.sklearn.log_model(
                model, "model", signature=signature, input_example=input_example
            )
            mlflow.log_metric("accuracy", accuracy)

            # Run 정보 출력
            run_id = run.info.run_id
            artifact_uri = mlflow.get_artifact_uri("model")
            print(f"Run ID: {run_id}")
            print(f"Artifact URI: {artifact_uri}")

            send_slack_notification(
                status="성공",
                message=f"모델 학습 성공\nRun ID: {run_id}\nAccuracy: {accuracy:.2f}",
            )
            return run_id, artifact_uri
    except Exception as e:
        send_slack_notification(status="실패", message=f"모델 학습 중 오류 발생: {str(e)}")
        raise


def register_model(run_id, artifact_uri):
    """MLflow 모델 레지스트리에 등록"""
    client = MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except Exception:
        print(f"Model {MODEL_NAME} already exists. Skipping creation.")

    # 모델 버전 생성
    try:
        model_version = client.create_model_version(
            name=MODEL_NAME, source=artifact_uri, run_id=run_id
        )
        print(f"Model version {model_version.version} created.")
        send_slack_notification(
            status="성공",
            message=f"모델 등록 성공\nModel: {MODEL_NAME}\nVersion: {model_version.version}",
        )

        # 모델을 'Staging' 단계로 전환
        client.transition_model_version_stage(
            name=MODEL_NAME, version=model_version.version, stage="Staging"
        )
        print(f"Model version {model_version.version} moved to Staging.")
        send_slack_notification(
            status="성공",
            message=f"모델 Staging 단계로 전환 완료\nModel: {MODEL_NAME}\nVersion: {model_version.version}",
        )
    except Exception as e:
        send_slack_notification(status="실패", message=f"모델 등록 중 오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    run_id, artifact_uri = train_model()
    register_model(run_id, artifact_uri)
