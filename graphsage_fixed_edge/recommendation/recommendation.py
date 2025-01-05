from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import pandas as pd

def recommend_concepts(target_concept_id, similarities, id_to_idx, idx_to_id, learning_order, top_k):
    """
    타겟 개념에 대해 유사도가 높은 개념 추천
    """
    if target_concept_id not in id_to_idx:
        raise ValueError(f"타겟 개념 ID {target_concept_id}가 존재하지 않습니다.")
    if target_concept_id not in learning_order:
        raise ValueError(f"타겟 개념 ID {target_concept_id}의 학습 순서 정보가 없습니다.")
    
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


def map_concept_to_ltitle(concept_ids, csv_file, column_name="f_mchapter_id"):
    """
    개념 ID를 f_mchapter_id 컬럼 값으로 매핑하고 해당 l_title 출력
    """
    # CSV 파일 로드
    df = pd.read_csv(csv_file)

    # 데이터 타입 변환
    df[column_name] = df[column_name].astype(str)

    results = {}
    for concept_id in concept_ids:
        concept_id_str = str(concept_id)
        print(f"비교 대상 concept_id: {concept_id_str}")  # 비교 대상 출력
        matching_rows = df[df[column_name] == concept_id_str]
        if not matching_rows.empty:
            results[concept_id] = matching_rows["l_title"].tolist()
        else:
            print(f"매핑 실패: concept_id {concept_id_str}는 CSV에서 찾을 수 없음.")  # 매핑 실패 출력
            results[concept_id] = []
    return results



if __name__ == "__main__":
    # 학습된 임베딩 파일 경로
    embedding_file = "trained_node_embeddings.npy"
    id_to_idx_file = "id_to_idx.json"
    idx_to_id_file = "idx_to_id.json"
    learning_order_file = "learning_order.json"  # 학습 순서 파일(복습 개념만 추천하도록 설정)
    lecture_csv_file = "lecture.csv"  # 매핑할 파일 경로

    # graph 생성 시 배치된 인덱스와 행렬 계산 때 배정된 인덱스 매핑핑
    with open(id_to_idx_file, "r") as f:
        id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(idx_to_id_file, "r") as f:
        idx_to_id = {int(k): int(v) for k, v in json.load(f).items()}
    with open(learning_order_file, "r") as f:
        learning_order = {int(k): int(v) for k, v in json.load(f).items()}

    # 타겟 개념 ID 입력
    target_concept_id = int(input("타겟 개념 ID를 입력하세요: "))

    if target_concept_id not in learning_order:
        print(f"타겟 개념 ID {target_concept_id}의 학습 순서 정보가 없습니다.")
        exit(1)

    # 상위 추천 개수 설정
    top_k = 5

    # 학습된 노드 임베딩 불러오기
    embeddings = load_embeddings(embedding_file)

    # 코사인 유사도 계산
    similarities = compute_similarities(embeddings)

    # 타겟 개념과 유사한 개념 추천
    similar_concepts, similar_scores = recommend_concepts(
        target_concept_id, similarities, id_to_idx, idx_to_id, learning_order, top_k=top_k
    )
    
    # 개념 ID를 f_mchapter_nm으로 매핑하여 l_title 확인
    if similar_concepts:
        mapping_results = map_concept_to_ltitle(similar_concepts, lecture_csv_file)
        print(f"타겟 개념 {target_concept_id}과 유사한 복습 개념:")
        for concept, score in zip(similar_concepts, similar_scores):
            ltitles = mapping_results.get(concept, [])
            if ltitles:
                print(f"개념 ID: {concept}, 유사도: {score:.4f}, 매핑된 l_title: {', '.join(map(str, ltitles))}")
            else:
                print(f"개념 ID: {concept}, 유사도: {score:.4f}, 매핑된 l_title: 없음")
    else:
        print(f"타겟 개념 {target_concept_id}과 유사한 복습 개념이 없습니다.")
