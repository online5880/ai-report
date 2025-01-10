import pandas as pd
import numpy as np
import json
from collections import defaultdict

# 1. 임베딩 파일 로드
def load_embeddings(file_path):
    """임베딩 파일 로드"""
    return np.load(file_path)

# 2. 성취기준 코드 그룹별 노드 쌍 평균 코사인 유사도 계산
def evaluate_similarity_by_code(code_to_nodes, embeddings, id_to_idx):
    """
    성취기준 코드 그룹별 노드 쌍 평균 유사도 계산
    """
    group_similarities = {}

    for code, nodes in code_to_nodes.items():
        # 유효한 노드만 필터링
        valid_nodes = [node for node in nodes if str(node) in id_to_idx]
        if len(valid_nodes) > 1:  # 그룹 내 유효 노드가 2개 이상일 경우에만 계산
            similarities = []
            for i in range(len(valid_nodes)):
                for j in range(i + 1, len(valid_nodes)):
                    # 노드 ID -> 임베딩 인덱스 변환
                    idx_i = id_to_idx[str(valid_nodes[i])]
                    idx_j = id_to_idx[str(valid_nodes[j])]

                    # 코사인 유사도 계산
                    sim = np.dot(embeddings[idx_i], embeddings[idx_j]) / (
                        np.linalg.norm(embeddings[idx_i]) * np.linalg.norm(embeddings[idx_j])
                    )
                    similarities.append(sim)
            group_similarities[code] = np.mean(similarities)
        else:
            group_similarities[code] = None  # 계산 불가능한 경우 None 처리
    
    return group_similarities

# 3. 성취기준 코드별 노드 그룹화
def group_nodes_by_code(df):
    """
    성취기준 코드별로 노드를 그룹화
    """
    node_to_codes = df.groupby("f_mchapter_id")["성취기준코드"].apply(list).to_dict()
    code_to_nodes = defaultdict(list)
    for node, codes in node_to_codes.items():
        for code in codes:
            code_to_nodes[code].append(node)
    return code_to_nodes

# 메인 실행
if __name__ == "__main__":
    # 파일 경로 설정
    embedding_file = "../models/trained_node_embeddings.npy"  # 임베딩 파일 경로
    id_to_idx_file = "id_to_idx.json"  # ID -> Index 매핑 파일
    lecture_file = "merged_final_data_new.csv"  # CSV 파일 경로

    # 매핑 파일 로드
    with open(id_to_idx_file, "r") as f:
        id_to_idx = {str(k): int(v) for k, v in json.load(f).items()}  # Key를 str로 변환

    # 학습된 노드 임베딩 불러오기
    embeddings = load_embeddings(embedding_file)

    # 성취기준 코드별 데이터 로드 및 그룹화
    df = pd.read_csv(lecture_file)
    code_to_nodes = group_nodes_by_code(df)

    # 성취기준 코드 그룹별 평균 코사인 유사도 계산
    group_similarities = evaluate_similarity_by_code(code_to_nodes, embeddings, id_to_idx)

    # 전체 평균 계산
    valid_similarities = [sim for sim in group_similarities.values() if sim is not None]
    overall_mean_similarity = np.mean(valid_similarities)

    # 결과 출력
    print(f"누락된 노드 수: {len(set(df['f_mchapter_id'].astype(str)) - set(id_to_idx.keys()))}")
    print(f"누락된 노드 ID: {set(df['f_mchapter_id'].astype(str)) - set(id_to_idx.keys())}")
    print("성취기준 코드 그룹별 평균 코사인 유사도:")
    for code, sim in group_similarities.items():
        print(f"성취기준 코드: {code}, 평균 유사도: {sim:.4f}" if sim is not None else f"성취기준 코드: {code}, 평균 유사도: 계산 불가")
    print(f"성취기준 코드 그룹 내 전체 평균 코사인 유사도: {overall_mean_similarity:.4f}")
