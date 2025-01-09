from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import pandas as pd

def recommend_concepts(target_concept_id, similarities, id_to_idx, idx_to_id, learning_order, top_k):
    """
    타겟 개념에 대해 유사도가 높은 개념 추천
    """
    if target_concept_id not in id_to_idx or target_concept_id not in learning_order:
        return [], []
    
    target_idx = id_to_idx[target_concept_id]

    # 개념 시작 노드는 자기 자신만 추천
    exclude_self_ids = {14201779, 14201784, 14201792, 14201818, 14201871, 14201877}
    if target_concept_id in exclude_self_ids:
        return [target_concept_id], [1.0]

    # 유사도 기준 정렬
    all_indices = np.argsort(-similarities[target_idx])
    similar_indices = [idx for idx in all_indices if idx != target_idx]

    # 유사도 0.7 이상 필터링
    filtered_indices = [
        idx for idx in similar_indices if similarities[target_idx][idx] >= 0.7
    ]

    # 학습 순서 필터링 (타겟 개념보다 이전에 배운 개념만 추천하도록 설정)
    target_order = learning_order[target_concept_id]
    filtered_indices_by_order = [
        idx for idx in filtered_indices if learning_order.get(idx_to_id[idx], float('inf')) < target_order
    ]

    # 타겟 ID가 14201781일 경우
    if target_concept_id == 14201781:
        # 유사도 조건(0.7)을 만족하지 못하더라도 학습 순서를 만족하는 개념 중 가장 유사도가 높은 한 개를 추천
        fallback_indices_by_order = [
            idx for idx in similar_indices if learning_order.get(idx_to_id[idx], float('inf')) < target_order
        ]
        if fallback_indices_by_order:
            best_idx = fallback_indices_by_order[0]  
            return [idx_to_id[best_idx]], [similarities[target_idx][best_idx]]

        return [], []  # 학습 순서를 만족하는 개념이 없는 경우

    # 상위 K개 제한 (일반적인 경우)
    top_k_indices = filtered_indices_by_order[:top_k]
    top_k_ids = [idx_to_id[idx] for idx in top_k_indices]
    top_k_similarities = [similarities[target_idx][idx] for idx in top_k_indices]

    return top_k_ids, top_k_similarities

def compute_similarities(embeddings):
    """임베딩을 이용하여 코사인 유사도 계산"""
    return cosine_similarity(embeddings)

def load_embeddings(file_path):
    """임베딩 파일 로드"""
    return np.load(file_path)

def process_predictions(predictions, similarities, id_to_idx, idx_to_id, learning_order, top_k):
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
        # 타겟 ID와 추천 결과를 하나의 리스트로 묶어서 추가
        grouped_recommendations.append([target_concept_id] + similar_concepts)

    return grouped_recommendations


def map_recommendations_to_lecture_data(grouped_recommendations, lecture_df_path):
    """
    grouped_recommendations 리스트의 모든 ID를 lecture_df에서 찾아 출력
    """
    # CSV 파일 읽기
    lecture_df = pd.read_csv(lecture_df_path)

    # 결과를 저장할 리스트
    results = []

    for group in grouped_recommendations:
        target_id = group[0]  # 타겟 개념
        similar_ids = group[1:]  # 유사한 개념 리스트

        # 타겟 개념 데이터 가져오기
        target_data = lecture_df[lecture_df['f_mchapter_id'] == target_id].to_dict(orient="records")

        # 유사 개념 데이터 가져오기
        similar_data = lecture_df[lecture_df['f_mchapter_id'].isin(similar_ids)].to_dict(orient="records")

        # 결과 저장
        results.append({"target": target_data, "similar": similar_data})

    return results


if __name__ == "__main__":
    # 학습된 임베딩 파일 경로
    embedding_file = "../models/trained_node_embeddings.npy"
    id_to_idx_file = "../fastapi_app/data/id_to_idx.json"
    idx_to_id_file = "../fastapi_app/data/idx_to_id.json"
    learning_order_file = "../fastapi_app/data/learning_order.json"
    lecture_df_path = "../fastapi_app/data/lecture_df.csv"
    
    
    # graph 생성 시 배치된 인덱스와 행렬 계산 때 배정된 인덱스 매핑
    with open(id_to_idx_file, "r") as f:
        id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(idx_to_id_file, "r") as f:
        idx_to_id = {int(k): int(v) for k, v in json.load(f).items()}
    with open(learning_order_file, "r") as f:
        learning_order = {int(k): int(v) for k, v in json.load(f).items()}

    # 학습된 노드 임베딩 불러오기
    embeddings = load_embeddings(embedding_file)

    # 코사인 유사도 계산
    similarities = compute_similarities(embeddings)

    # predictions 데이터 처리
    predictions= {
        "predictions": [
            {
                "14201781": 0.127743005752563
            },
            {
                "14201897": 0.8259633779525757
            },
            {
                "14201897": 0.8356612324714661
            },
            {
                "14201897": 0.8063239455223083
            },
            {
                "14201897": 0.8274922370910645
            },
            {
                "14201869": 0.98056304693222046
            },
            {
                "14201868": 0.9563564658164978
            },
            {
                "14201869": 0.9187106490135193
            },
            {
                "14201868": 0.9601629972457886
            },
            {
                "14201857": 0.6699686050415
            }
        ]
    }

    # # API 요청 데이터 정의
    # api_url = "http://0.0.0.0:8100/api/gkt"  # GKT 모델 API 엔드포인트
    # user_history = {
    #     "user_id": 12345,
    #     "question_history": [
    #         {"question_id": "14201897", "answer_correct": True},
    #         {"question_id": "14201869", "answer_correct": False},
    #         {"question_id": "14201868", "answer_correct": True}
    #     ]
    # }

    # # API 요청
    # try:
    #     response = requests.post(api_url, json=user_history)
    #     response.raise_for_status()  # HTTP 오류 발생 시 예외 처리

    #     # API 응답 데이터
    #     predictions = response.json()  # JSON 형식으로 파싱
    #     print("API 응답 데이터:", json.dumps(predictions, indent=4))

    # except requests.exceptions.RequestException as e:
    #     print(f"API 요청 실패: {e}")
    #     exit(1)


    # 상위 추천 개수 설정
    top_k = 2

    # 이해도가 낮은 중단원들에 대한 추천 실행
    grouped_recommendations = process_predictions(
        predictions["predictions"], similarities, id_to_idx, idx_to_id, learning_order, top_k
    )

    # 추천 결과를 lecture_df와 매핑
    mapped_results = map_recommendations_to_lecture_data(grouped_recommendations, lecture_df_path)

    # 결과 출력
    for result in mapped_results:
        print("현재 부족한 개념:")
        for data in result["target"]:
            print(data)
        print("함께 복습하면 좋은 강의:")
        for data in result["similar"]:
            print(data)
        print("-" * 76)

