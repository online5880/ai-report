import json
import torch
import numpy as np
import dgl

# 1. ID 맵 파일 읽기 (학생, 중단원, 강의에 대한 인덱싱 정보)
with open('id_map.json', 'r') as f:
    id_to_index = json.load(f)

# 2. 학생-중단원 관계 데이터 읽기
with open('user_to_mchapter_data.json', 'r') as f:
    student_to_concept_data = json.load(f)

# 3. 중단원-강의 관계 데이터 읽기
with open('mchapter_to_mcode_data.json', 'r') as f:
    mchapter_to_mcode_data = json.load(f)

# 4. 학생-중단원 관계에서 엣지 리스트 생성
edges_student_to_concept = [
    (id_to_index['students'][student], id_to_index['concepts'][concept]) 
    for student, concept in student_to_concept_data
]

# 5. 중단원-강의 관계에서 엣지 리스트 생성
edges_concept_to_lecture = [
    (id_to_index['concepts'][concept], id_to_index['lectures'][lecture])
    for concept, lecture in mchapter_to_mcode_data
]

# 6. 그래프 생성
edges = {
    ('student', 'understands', 'concept'): edges_student_to_concept,
    ('concept', 'teaches', 'lecture'): edges_concept_to_lecture,
}

# 노드 타입별로 필요한 노드 개수 정의
num_nodes_dict = {
    'student': len(id_to_index['students']),
    'concept': len(id_to_index['concepts']),
    'lecture': len(id_to_index['lectures']),
}

# DGL 그래프 생성
graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

# 노드 특성 설정
# 학생 노드 특성 (예시로 10차원 랜덤 값 설정)
graph.nodes['student'].data['feat'] = torch.randn(num_nodes_dict['student'], 10)

# 강의 노드 특성 (예시로 20차원 랜덤 값 설정)
graph.nodes['lecture'].data['feat'] = torch.randn(num_nodes_dict['lecture'], 20)

# 중단원 노드 특성 설정
# npy 파일 로드
# 중단원 노드 특성 설정

# 중단원 노드 특성 설정
file_path = "embedding/reduced_mchapter_embeddings_94.npy"  # 임베딩 데이터 경로
node_features = np.load(file_path, allow_pickle=True).item()  # 임베딩 데이터 로드
ids = node_features['ids']  # 중단원 ID
embeddings = node_features['embeddings']  # 중단원 임베딩

# Debug: ID 형식 변환 및 교집합 계산
ids_as_strings = list(map(str, ids))  # 임베딩 IDs를 문자열로 변환
concept_ids_in_embeddings = set(ids_as_strings)  # 문자열로 변환된 IDs
concept_ids_in_graph = set(id_to_index['concepts'].keys())  # id_map.json의 concepts IDs

common_ids = concept_ids_in_embeddings.intersection(concept_ids_in_graph)
print(f"Matching concepts after type conversion: {len(common_ids)}")

# 공통 ID를 기준으로 임베딩 정렬
sorted_embeddings = [embeddings[ids.index(int(concept_id))] for concept_id in common_ids]

# 리스트를 NumPy 배열로 변환한 후 Torch 텐서로 변환
concept_node_features = torch.tensor(np.array(sorted_embeddings), dtype=torch.float32)
graph.nodes['concept'].data['feat'] = concept_node_features

# 그래프 정보 출력
print(graph)

# 학생, 중단원, 강의 노드의 데이터 크기 확인
print("Student node feature shape:", graph.nodes['student'].data['feat'].shape)
print("Concept node feature shape:", graph.nodes['concept'].data['feat'].shape)
print("Lecture node feature shape:", graph.nodes['lecture'].data['feat'].shape)

'''
그래프 생성 결과
Matching concepts after type conversion: 34
Graph(num_nodes={'concept': 34, 'lecture': 60, 'student': 32946},
      num_edges={('concept', 'teaches', 'lecture'): 68, ('student', 'understands', 'concept'): 232806},
      metagraph=[('concept', 'lecture', 'teaches'), ('student', 'concept', 'understands')])
Student node feature shape: torch.Size([32946, 10])
Concept node feature shape: torch.Size([34, 94])
Lecture node feature shape: torch.Size([60, 20])
'''