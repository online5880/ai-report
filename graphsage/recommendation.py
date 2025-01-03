from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recommend_concepts(target_concept, similarities, top_k=5):
    """
    타겟 개념에 대해 유사도가 높은 개념 추천
    - target_concept: 타겟 개념의 인덱스
    - similarities: 코사인 유사도 행렬
    - top_k: 상위 추천 개수
    """
    # 유사도를 기반으로 정렬된 인덱스 가져오기
    all_concepts = np.argsort(-similarities[target_concept])
    
    # 타겟 노드를 제외하고 상위 K개 선택
    similar_concepts = [idx for idx in all_concepts if idx != target_concept][:top_k]
    return similar_concepts

def compute_similarities(embeddings):
    """임베딩을 이용하여 코사인 유사도 계산"""
    return cosine_similarity(embeddings)
