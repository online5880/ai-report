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
import numpy as np

# 학습된 임베딩 파일 불러오기
def load_embeddings(file_path):
    """
    임베딩 파일 로드
    - file_path: 임베딩이 저장된 파일 경로 (.npy 형식 가정)
    """
    return np.load(file_path)

# 위에서 정의한 함수 활용
if __name__ == "__main__":
    # 학습된 임베딩 파일 경로
    embedding_file = "trained_node_embeddings.npy"
    
    # 타겟 개념 인덱스 입력
    target_concept = int(input("타겟 개념 인덱스를 입력하세요: "))
    
    # 상위 추천 개수 설정
    top_k = 5  # 기본값 5
    
    # 임베딩 불러오기
    embeddings = load_embeddings(embedding_file)
    
    # 코사인 유사도 계산
    similarities = compute_similarities(embeddings)
    
    # 타겟 개념과 유사한 개념 추천
    similar_concepts = recommend_concepts(target_concept, similarities, top_k=top_k)
    
    print(f"타겟 개념 {target_concept}과 유사한 상위 {top_k} 개념: {similar_concepts}")
