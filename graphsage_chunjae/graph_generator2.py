# 중단원 -> 학생 엣지 추가
import json
import torch
import numpy as np
import dgl
from sklearn.decomposition import PCA
import torch.nn as nn

# 1. ID 맵 파일 읽기 (학생, 중단원, 강의에 대한 인덱싱 정보)
with open('id_map.json', 'r') as f:
    id_to_index = json.load(f)

# 2. 학생-중단원 관계 데이터 읽기
with open('student_to_concept_data.json', 'r') as f:
    student_to_concept_data = json.load(f)

# 3. 중단원-강의 관계 데이터 읽기
with open('concept_to_lecture_data.json', 'r') as f:
    concept_to_lecture_data = json.load(f)

# 4. 학생-중단원 관계에서 엣지 리스트 생성
edges_student_to_concept = [
    (id_to_index['students'][student], id_to_index['concepts'][concept]) 
    for student, concept in student_to_concept_data
    if student in id_to_index['students'] and concept in id_to_index['concepts']
]

# 5. 중단원-강의 관계에서 엣지 리스트 생성
edges_concept_to_lecture = [
    (id_to_index['concepts'][concept], id_to_index['lectures'][lecture])
    for concept, lecture in concept_to_lecture_data
    if concept in id_to_index['concepts'] and lecture in id_to_index['lectures']
]

# 6. 그래프 생성
edges = {
    ('student', 'understands', 'concept'): edges_student_to_concept,
    ('concept', 'teaches', 'lecture'): edges_concept_to_lecture,
    ('concept', 'understood_by', 'student'): [(dst, src) for src, dst in edges_student_to_concept],
}

# 노드 타입별로 필요한 노드 개수 정의
num_nodes_dict = {
    'student': len(id_to_index['students']),
    'concept': len(id_to_index['concepts']),
    'lecture': len(id_to_index['lectures']),
}

# DGL 그래프 생성
graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

# 7. 학생 노드 특성 (중단원 이해도 추가)
with open('embedding/transformed_student_to_concept_data.json', 'r') as f:  # JSON 파일 경로
    student_concept_scores = json.load(f)

# 7.1 중단원 ID 인덱싱
concept_ids = sorted(list(id_to_index['concepts'].keys()))  # 중단원 ID를 정렬
concept_id_to_index = {concept_id: idx for idx, concept_id in enumerate(concept_ids)}  # ID → Index 매핑

# 7.2 학생 노드 특성 초기화
num_students = len(id_to_index['students'])
num_concepts = len(concept_ids)  # 중단원 수
student_features = np.zeros((num_students, num_concepts))  # 학생 노드 특성 초기화 (학생 수 x 중단원 수)

# 7.3 이해도 벡터 생성
for student_id, concept_scores in student_concept_scores.items():
    if student_id in id_to_index['students']:  # 유효한 학생 ID만 처리
        student_idx = id_to_index['students'][student_id]  # 학생 인덱스
        for concept_id, score in concept_scores.items():
            if concept_id in concept_id_to_index:  # 유효한 중단원 ID만 처리
                concept_idx = concept_id_to_index[concept_id]  # 중단원 인덱스
                student_features[student_idx, concept_idx] = score  # 이해도 값 할당

print("Student features shape:", student_features.shape)
# 7.4 DGL 그래프에 학생 노드 특성 추가
graph.nodes['student'].data['feat'] = torch.tensor(student_features, dtype=torch.float32)

# 8. 강의 노드 특성 (예시로 20차원 랜덤 값 설정)
graph.nodes['lecture'].data['feat'] = torch.randn(num_nodes_dict['lecture'], 20)

# 9. 중단원 노드 특성 설정
file_path = "embedding/reduced_mchapter_embeddings_6.npy"
node_features = np.load(file_path, allow_pickle=True).item()
print("Loaded node features keys:", node_features.keys())
print("IDs:", node_features['ids'][:5])  # 일부 ID 확인
print("Embeddings shape:", node_features['embeddings'].shape)

ids = node_features['ids']  # 중단원 ID
embeddings = node_features['embeddings']  # 중단원 임베딩
print("로드된 embeddings shape:", embeddings.shape)  # 예: (34, 6)

# Debug: ID 형식 변환 및 교집합 계산
ids_as_strings = list(map(str, ids))  # 임베딩 IDs를 문자열로 변환
concept_ids_in_embeddings = set(ids_as_strings)  # 문자열로 변환된 IDs
concept_ids_in_graph = set(id_to_index['concepts'].keys())  # id_map.json의 concepts IDs

common_ids = concept_ids_in_embeddings.intersection(concept_ids_in_graph)
print(f"Matching concepts after type conversion: {len(common_ids)}")

# 공통 ID를 기준으로 임베딩 정렬 및 누락된 ID 처리
default_embedding = np.mean(embeddings, axis=0)  # 평균 임베딩
sorted_embeddings = [
    embeddings[ids.index(int(concept_id))] if concept_id in ids_as_strings else default_embedding
    for concept_id in id_to_index['concepts'].keys()
]

# 6차원으로 축소된 데이터 사용
concept_node_features = torch.tensor(np.array(sorted_embeddings), dtype=torch.float32)
graph.nodes['concept'].data['feat'] = concept_node_features

# 그래프 정보 출력
print(graph)

# 학생, 중단원, 강의 노드의 데이터 크기 확인
print("Student node feature shape:", graph.nodes['student'].data['feat'].shape)
print("Concept node feature shape:", graph.nodes['concept'].data['feat'].shape)
print("Lecture node feature shape:", graph.nodes['lecture'].data['feat'].shape)

# 샘플 데이터 검증
print("Sample edges (student to concept):", edges_student_to_concept[:5])
print("Sample edges (concept to lecture):", edges_concept_to_lecture[:5])

# 누락된 중단원 ID 로그 출력
missing_concepts = set(id_to_index['concepts'].keys()).difference(concept_ids_in_embeddings)
print(f"Missing concepts in embeddings: {len(missing_concepts)}")

def generate_graph():
    return graph

# === 그래프 준비 ===
g = generate_graph()

# 그래프 구조 확인
print("Node types:", g.ntypes)  # 노드 타입 확인
print("Edge types:", g.etypes)  # 엣지 타입 확인
print("Graph schema:", g)       # 그래프 전체 스키마 확인

print("Student node degrees:")
print(g.in_degrees(etype="understands"))
print(g.out_degrees(etype="understands"))


target_dim = 128  # 목표 차원

# 학생 임베딩 변환
student_features = graph.nodes['student'].data['feat']
student_transform = nn.Linear(student_features.shape[1], target_dim)  # 34 → 128 변환
student_features_transformed = student_transform(student_features)

# 중단원 임베딩 변환
concept_features = graph.nodes['concept'].data['feat']
concept_transform = nn.Linear(concept_features.shape[1], target_dim)  # 6 → 128 변환
concept_features_transformed = concept_transform(concept_features)

# 강의 임베딩 변환
lecture_features = graph.nodes['lecture'].data['feat']
lecture_transform = nn.Linear(lecture_features.shape[1], target_dim)  # 20 → 128 변환
lecture_features_transformed = lecture_transform(lecture_features)

# 변환된 임베딩을 그래프에 다시 설정
graph.nodes['student'].data['feat'] = student_features_transformed
graph.nodes['concept'].data['feat'] = concept_features_transformed
graph.nodes['lecture'].data['feat'] = lecture_features_transformed

# 결과 확인
print("Transformed student features shape:", graph.nodes['student'].data['feat'].shape)
print("Transformed concept features shape:", graph.nodes['concept'].data['feat'].shape)
print("Transformed lecture features shape:", graph.nodes['lecture'].data['feat'].shape)

print(f"Common IDs: {len(common_ids)}")
print(f"Missing concepts: {len(missing_concepts)}")
print("Student features input shape:", student_features.shape)
print("Concept features input shape:", concept_features.shape)
print("Lecture features input shape:", lecture_features.shape)

