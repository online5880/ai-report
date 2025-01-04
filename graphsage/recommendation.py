from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

def recommend_concepts(target_concept_id, similarities, id_to_idx, idx_to_id, top_k=5):
    """
    타겟 개념에 대해 유사도가 높은 개념 추천
    - target_concept_id: 타겟 개념의 실제 ID
    - similarities: 코사인 유사도 행렬
    - id_to_idx: 노드 ID -> 배열 인덱스 매핑
    - idx_to_id: 배열 인덱스 -> 노드 ID 매핑
    - top_k: 상위 추천 개수
    """
    if target_concept_id not in id_to_idx:
        raise ValueError(f"타겟 개념 ID {target_concept_id}가 존재하지 않습니다.")
    
    target_idx = id_to_idx[target_concept_id]
    all_indices = np.argsort(-similarities[target_idx])
    
    # 유사도가 비슷한 상위 K개 선택
    similar_indices = [idx for idx in all_indices if idx != target_idx][:top_k]
    similar_ids = [idx_to_id[idx] for idx in similar_indices]
    return similar_ids

def compute_similarities(embeddings):
    """임베딩을 이용하여 코사인 유사도 계산"""
    return cosine_similarity(embeddings)

def load_embeddings(file_path):
    """임베딩 파일 로드"""
    return np.load(file_path)

if __name__ == "__main__":
    # 학습된 임베딩 파일 다운로드드
    embedding_file = "trained_node_embeddings.npy"
    
    # ID 매핑 정보 로드(노드 생성시 부여되는 인덱스와 행렬 생성 시 부여되는 인덱스 매핑과정)
    id_to_idx_file = "id_to_idx.json"
    idx_to_id_file = "idx_to_id.json"

    # JSON 파일 로드 및 키/값 정수형 변환
    with open(id_to_idx_file, "r") as f:
        id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(idx_to_id_file, "r") as f:
        idx_to_id = {int(k): int(v) for k, v in json.load(f).items()}

    # 타겟 개념 ID 입력
    target_concept_id = int(input("타겟 개념 ID를 입력하세요: "))
    
    # 상위 추천 개수 설정
    top_k = 5
    
    # 임베딩 불러오기
    embeddings = load_embeddings(embedding_file)
    
    # 코사인 유사도 계산
    similarities = compute_similarities(embeddings)
    
    # 타겟 개념과 유사한 개념 추천
    similar_concepts = recommend_concepts(target_concept_id, similarities, id_to_idx, idx_to_id, top_k=top_k)
    
    print(f"타겟 개념 {target_concept_id}과 유사한 상위 {top_k} 개념: {similar_concepts}")
