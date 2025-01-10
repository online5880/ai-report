from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import pandas as pd
import random

def evaluate_precision(target_concept_id, recommended_concepts, label_dict):
    """Precision@5 평가"""
    target_label = label_dict.get(target_concept_id, None)
    if target_label is None:
        raise ValueError(f"타겟 개념 ID {target_concept_id}의 라벨을 찾을 수 없습니다.")
    
    correct_count = 0
    for concept_id in recommended_concepts:
        concept_label = label_dict.get(concept_id, None)
        if concept_label == target_label:
            correct_count += 1
    
    precision_at_k = correct_count / len(recommended_concepts)
    return precision_at_k

def recommend_concepts(target_concept_id, similarities, id_to_idx, idx_to_id, top_k):
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
    
    # 타겟 노드를 제외하고 상위 K개 선택
    similar_indices = [idx for idx in all_indices if idx != target_idx]
    similar_ids = [idx_to_id[idx] for idx in similar_indices]
    
    # 상위 K개 제한
    top_k_indices = similar_indices[:top_k]
    top_k_ids = [idx_to_id[idx] for idx in top_k_indices]
    top_k_similarities = [similarities[target_idx][idx] for idx in top_k_indices]
    
    return top_k_ids, top_k_similarities

def compute_similarities(embeddings):
    """임베딩을 이용하여 코사인 유사도 계산"""
    return cosine_similarity(embeddings)

def load_embeddings(file_path):
    """임베딩 파일 로드"""
    return np.load(file_path)

def load_labels(label_file):
    """라벨 파일 로드"""
    labels_df = pd.read_csv(label_file)
    labels_dict = dict(zip(labels_df['mchapter_id'], labels_df['label']))
    return labels_dict

def evaluate_precision(target_concept_id, recommended_concepts, label_dict):
    """Precision@5 평가"""
    target_label = label_dict.get(target_concept_id, None)
    if target_label is None:
        raise ValueError(f"타겟 개념 ID {target_concept_id}의 라벨을 찾을 수 없습니다.")
    
    correct_count = 0
    for concept_id in recommended_concepts:
        concept_label = label_dict.get(concept_id, None)
        if concept_label == target_label:
            correct_count += 1
    
    precision_at_k = correct_count / len(recommended_concepts)
    return precision_at_k

if __name__ == "__main__":
    # 학습된 임베딩 파일 경로
    embedding_file = "../models/trained_node_embeddings.npy"
    id_to_idx_file = "id_to_idx.json"
    idx_to_id_file = "idx_to_id.json"
    label_file = "node_labels.csv"  # 개념 ID와 라벨이 포함된 파일

    # JSON 파일 로드 및 키/값 정수형 변환
    with open(id_to_idx_file, "r") as f:
        id_to_idx = {int(k): int(v) for k, v in json.load(f).items()}
    with open(idx_to_id_file, "r") as f:
        idx_to_id = {int(k): int(v) for k, v in json.load(f).items()}

    # 라벨 로드
    label_dict = load_labels(label_file)

    # 타겟 개념 ID 랜덤으로 20개 선택
    all_concept_ids = list(label_dict.keys())
    random_targets = random.sample(all_concept_ids, 20)

    # 상위 추천 개수 설정
    top_k = 5

    # 임베딩 불러오기
    embeddings = load_embeddings(embedding_file)

    # 코사인 유사도 계산
    similarities = compute_similarities(embeddings)

    precision_values = []
    
    # 랜덤 타겟에 대해 Precision@5 계산
    for target_concept_id in random_targets:
        # 타겟 개념과 유사한 개념 추천
        similar_concepts, similar_scores = recommend_concepts(
            target_concept_id, similarities, id_to_idx, idx_to_id, top_k=top_k
        )
        
        # Precision 평가
        precision = evaluate_precision(target_concept_id, similar_concepts, label_dict)
        precision_values.append(precision)
    
    # 평균 Precision@5 출력
    average_precision = np.mean(precision_values)
    print(f"랜덤으로 선택된 20개의 타겟에 대한 Precision@5 평균: {average_precision:.4f}")
