# import json
# import os
# from typing import List, Dict
# from fastapi import APIRouter, HTTPException
# from fastapi_app.schemas import RecommendRequest, RecommendResponse
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import pandas as pd

# router = APIRouter()

# # 프로젝트 루트 경로 설정
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_DIR = os.path.join(BASE_DIR, "data")


# def load_embeddings(file_path: str) -> np.ndarray:
#     """임베딩 파일 로드"""
#     return np.load(file_path)


# def compute_similarities(embeddings: np.ndarray) -> np.ndarray:
#     """코사인 유사도 계산"""
#     return cosine_similarity(embeddings)


# def recommend_concepts(target_concept_id: int, similarities: np.ndarray, id_to_idx: Dict[int, int],
#                        idx_to_id: Dict[int, int], learning_order: Dict[int, int], top_k: int):
#     """
#     유사도 기반 개념 추천
#     """
#     if target_concept_id not in id_to_idx:
#         raise ValueError(f"타겟 개념 ID {target_concept_id}가 존재하지 않습니다.")
#     if target_concept_id not in learning_order:
#         raise ValueError(f"타겟 개념 ID {target_concept_id}의 학습 순서 정보가 없습니다.")

#     target_idx = id_to_idx[target_concept_id]

#     # 특정 개념 ID 제외 처리
#     exclude_self_ids = {14201779, 14201784, 14201792, 14201818, 14201871, 14201877}
#     if target_concept_id in exclude_self_ids:
#         return [target_concept_id], [1.0]

#     # 유사도 기반 정렬
#     all_indices = np.argsort(-similarities[target_idx])
#     similar_indices = [idx for idx in all_indices if idx != target_idx]

#     # 유사도 0.7 이상 필터링
#     filtered_indices = [
#         idx for idx in similar_indices if similarities[target_idx][idx] >= 0.7
#     ]

#     # 학습 순서 조건
#     target_order = learning_order[target_concept_id]
#     filtered_indices_by_order = [
#         idx for idx in filtered_indices if learning_order.get(idx_to_id[idx], float('inf')) < target_order
#     ]

#     # 상위 K개 결과 반환
#     top_k_indices = filtered_indices_by_order[:top_k]
#     top_k_ids = [idx_to_id[idx] for idx in top_k_indices]
#     top_k_similarities = [similarities[target_idx][idx] for idx in top_k_indices]

#     return top_k_ids, top_k_similarities

# def map_recommendations_to_lecture_data(top_k_ids: List[int], lecture_df_path: str):
#     """
#     추천된 ID를 lecture_df에서 매핑하여 반환
#     """
#     lecture_df = pd.read_csv(lecture_df_path)

#     # 추천된 개념 ID에 해당하는 데이터를 검색
#     mapped_data = lecture_df[lecture_df['f_mchapter_id'].isin(top_k_ids)].to_dict(orient='records')

#     return mapped_data

# @router.post("/", response_model=RecommendResponse)
# def recommend(request: RecommendRequest):
#     """
#     추천 API 엔드포인트
#     """
#     try:
#         # 데이터 파일 경로 로드
#         id_to_idx_path = os.path.join(DATA_DIR, "id_to_idx.json")
#         idx_to_id_path = os.path.join(DATA_DIR, "idx_to_id.json")
#         learning_order_path = os.path.join(DATA_DIR, "learning_order.json")
#         lecture_df_path = os.path.join(DATA_DIR, "lecture_df.csv")

#         id_to_idx = {int(k): int(v) for k, v in json.load(open(id_to_idx_path)).items()}
#         idx_to_id = {int(k): int(v) for k, v in json.load(open(idx_to_id_path)).items()}
#         learning_order = {int(k): int(v) for k, v in json.load(open(learning_order_path)).items()}

#         # 한 번만 출력
#         if request.target_concept_id == id_to_idx[next(iter(id_to_idx))]:  # 첫 요청 시 출력
#             print("Loaded id_to_idx:", id_to_idx)
#             print("Loaded idx_to_id:", idx_to_id)
#             print("Loaded learning_order:", learning_order)

#         # 임베딩 로드 및 유사도 계산
#         embeddings = load_embeddings(request.embedding_path)
#         similarities = compute_similarities(embeddings)

#         # 추천 실행
#         top_k_ids, _ = recommend_concepts(
#             target_concept_id=request.target_concept_id,
#             similarities=similarities,
#             id_to_idx=id_to_idx,
#             idx_to_id=idx_to_id,
#             learning_order=learning_order,
#             top_k=request.top_k,
#         )
#         print(f"Recommended Top K IDs for Target {request.target_concept_id}: {top_k_ids}")

#         # 추천 결과 매핑
#         mapped_results = map_recommendations_to_lecture_data(top_k_ids, lecture_df_path)

#         print(f"Mapped Recommendations for Target {request.target_concept_id}: {mapped_results}")

#         # 결과 반환
#         return {"recommendations": mapped_results}

#     except ValueError as ve:
#         print(f"ValueError: {ve}")
#         raise HTTPException(status_code=400, detail=str(ve))
#     except Exception as e:
#         print(f"Unexpected error: {e}")
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


import json
import os
from fastapi import APIRouter, HTTPException
from fastapi_app.schemas import RecommendRequest, RecommendResponse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

router = APIRouter()

# 경로 설정
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 파일 위치
PROJECT_ROOT = os.path.abspath(
    os.path.join(CURRENT_FILE_DIR, "..", "..")
)  # 프로젝트 루트 디렉토리
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")  # models 디렉토리
DATA_DIR = os.path.join(PROJECT_ROOT, "fastapi_app", "data")  # 데이터 디렉토리
EMBEDDING_PATH = os.path.join(MODELS_DIR, "trained_node_embeddings.npy")

# 전역 변수
EMBEDDINGS = None
SIMILARITIES = None


def load_embeddings(file_path: str) -> np.ndarray:
    """임베딩 파일 로드"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embedding file not found at {file_path}")
    return np.load(file_path)

def compute_similarities(embeddings: np.ndarray) -> np.ndarray:
    """코사인 유사도 계산"""
    return cosine_similarity(embeddings)


def initialize_data():
    """임베딩 및 유사도 데이터 초기화"""
    global EMBEDDINGS, SIMILARITIES
    print(f"Initializing data with EMBEDDING_PATH: {EMBEDDING_PATH}")
    if not os.path.exists(EMBEDDING_PATH):
        raise FileNotFoundError(f"Embedding file not found at {EMBEDDING_PATH}")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found at {DATA_DIR}")

    # 임베딩 로드 및 유사도 계산
    EMBEDDINGS = load_embeddings(EMBEDDING_PATH)
    SIMILARITIES = compute_similarities(EMBEDDINGS)
    print("Embeddings and similarities initialized successfully")


# 서버 시작 시 데이터 로드
try:
    initialize_data()
except Exception as e:
    print(f"[Error] Failed to initialize data: {e}")
    raise


def recommend_concepts(
    target_concept_id, similarities, id_to_idx, idx_to_id, learning_order, top_k
):
    """
    타겟 개념에 대해 유사도가 높은 개념 추천
    """
    if target_concept_id not in id_to_idx or target_concept_id not in learning_order:
        return [], []

    target_idx = id_to_idx[target_concept_id]
    # 개념에 대한 시작 노드는 이전에 배운 후행 개념이 없으므로 예외 처리
    exclude_self_ids = {14201779, 14201784, 14201792, 14201818, 14201871, 14201877}
    if target_concept_id in exclude_self_ids:
        return [target_concept_id], [1.0]

    all_indices = np.argsort(-similarities[target_idx])
    similar_indices = [idx for idx in all_indices if idx != target_idx]
    filtered_indices = [
        idx for idx in similar_indices if similarities[target_idx][idx] >= 0.7
    ]

    target_order = learning_order[target_concept_id]
    filtered_indices_by_order = [
        idx
        for idx in filtered_indices
        if learning_order.get(idx_to_id[idx], float("inf")) < target_order
    ]

    if target_concept_id == 14201781:
        fallback_indices_by_order = [
            idx
            for idx in similar_indices
            if learning_order.get(idx_to_id[idx], float("inf")) < target_order
        ]
        if fallback_indices_by_order:
            best_idx = fallback_indices_by_order[0]
            return [idx_to_id[best_idx]], [similarities[target_idx][best_idx]]

        return [], []

    top_k_indices = filtered_indices_by_order[:top_k]
    top_k_ids = [idx_to_id[idx] for idx in top_k_indices]
    top_k_similarities = [similarities[target_idx][idx] for idx in top_k_indices]

    return top_k_ids, top_k_similarities


def process_predictions(
    predictions, similarities, id_to_idx, idx_to_id, learning_order, top_k
):
    """
    이해도가 0.5 이하인 중단원을 타겟으로 하여 유사한 개념을 추천.
    """
    low_confidence_concepts = [
        int(concept)
        for prediction in predictions
        for concept, confidence in prediction.items()
        if confidence < 0.5
    ]
    grouped_recommendations = []

    for target_concept_id in low_confidence_concepts:
        similar_concepts, _ = recommend_concepts(
            target_concept_id, similarities, id_to_idx, idx_to_id, learning_order, top_k
        )
        grouped_recommendations.append([target_concept_id] + similar_concepts)

    return grouped_recommendations


def map_recommendations_to_lecture_data(grouped_recommendations, lecture_df_path):
    """
    grouped_recommendations 리스트의 모든 ID를 lecture_df에서 찾아 출력
    """
    lecture_df = pd.read_csv(lecture_df_path)
    results = []

    for group in grouped_recommendations:
        target_id = group[0]
        similar_ids = group[1:]
        target_data = lecture_df[lecture_df["f_mchapter_id"] == target_id].to_dict(
            orient="records"
        )
        similar_data = lecture_df[
            lecture_df["f_mchapter_id"].isin(similar_ids)
        ].to_dict(orient="records")
        results.append({"target": target_data, "similar": similar_data})

    return results


@router.post("/", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    """
    추천 API 엔드포인트
    """
    try:
        # 사용자 정의 예외 처리: top_k 값 검증
        if request.top_k > 5:
            raise HTTPException(
                status_code=400, detail="The 'top_k' value must not exceed 5."
            )

        id_to_idx_path = os.path.join(DATA_DIR, "id_to_idx.json")
        idx_to_id_path = os.path.join(DATA_DIR, "idx_to_id.json")
        learning_order_path = os.path.join(DATA_DIR, "learning_order.json")
        lecture_df_path = os.path.join(DATA_DIR, "lecture_df.csv")

        id_to_idx = {int(k): int(v) for k, v in json.load(open(id_to_idx_path)).items()}
        idx_to_id = {int(k): int(v) for k, v in json.load(open(idx_to_id_path)).items()}
        learning_order = {
            int(k): int(v) for k, v in json.load(open(learning_order_path)).items()
        }

        grouped_recommendations = process_predictions(
            request.predictions,
            SIMILARITIES,
            id_to_idx,
            idx_to_id,
            learning_order,
            request.top_k,
        )

        mapped_results = map_recommendations_to_lecture_data(
            grouped_recommendations, lecture_df_path
        )

        return {"recommendations": mapped_results}

    except HTTPException as he:
        raise he
    except FileNotFoundError as fe:
        raise HTTPException(status_code=404, detail=f"File not found: {str(fe)}")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
