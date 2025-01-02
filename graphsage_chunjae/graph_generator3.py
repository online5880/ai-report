# 학생-중단원 엣지 가중치(이해도) 추가, 강의 노드 특성 임베딩 추가 
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

# 3. 학생-중단원 이해도 데이터 읽기
with open('embedding/transformed_student_to_concept_data.json', 'r') as f:
    student_concept_scores = json.load(f)

# 4. 중단원-강의 관계 데이터 읽기
with open('mchapter_to_mcode_data.json', 'r') as f:
    concept_to_lecture_data = json.load(f)

# # 5. 학생-중단원 엣지 리스트 생성
# edges_student_to_concept = []
# understanding_scores = []  # 학생-중단원 이해도 가중치 저장

# # 학습 기록을 바탕으로 엣지 생성
# for student_id, concept_id in student_to_concept_data:
#     if student_id in id_to_index['students'] and concept_id in id_to_index['concepts']:
#         # 학습한 중단원에 대한 이해도 점수 가져오기
#         if student_id in student_concept_scores and concept_id in student_concept_scores[student_id]:
#             score = student_concept_scores[student_id][concept_id]
#             if score > 0:  # 이해도가 0 이상인 경우에만 엣지 생성
#                 student_idx = id_to_index['students'][student_id]
#                 concept_idx = id_to_index['concepts'][concept_id]
#                 edges_student_to_concept.append((student_idx, concept_idx))
#                 understanding_scores.append(score)
# 4. 학생-중단원 관계에서 엣지 리스트 생성
edges_student_to_concept = [
    (id_to_index['students'][student], id_to_index['concepts'][concept]) 
    for student, concept in student_to_concept_data
    if student in id_to_index['students'] and concept in id_to_index['concepts']
]

# 6. 중단원-강의 엣지 리스트 생성
edges_concept_to_lecture = [
    (id_to_index['concepts'][concept], id_to_index['lectures'][lecture])
    for concept, lecture in concept_to_lecture_data
    if concept in id_to_index['concepts'] and lecture in id_to_index['lectures']
]

# 7. 그래프 생성
edges = {
    ('student', 'understands', 'concept'): edges_student_to_concept,
    ('concept', 'teaches', 'lecture'): edges_concept_to_lecture,
    # ('concept', 'understood_by', 'student'): [(dst, src) for src, dst in edges_student_to_concept],
}

num_nodes_dict = {
    'student': len(id_to_index['students']),
    'concept': len(id_to_index['concepts']),
    'lecture': len(id_to_index['lectures']),
}

graph = dgl.heterograph(edges, num_nodes_dict=num_nodes_dict)

# # 8. 엣지 가중치 추가
# graph.edges['understands'].data['weight'] = torch.tensor(understanding_scores, dtype=torch.float32)

# 9. 학생 노드 특성 설정
num_students = len(id_to_index['students'])
num_concepts = len(id_to_index['concepts'])
student_features = np.zeros((num_students, num_concepts))

for student_id, concept_scores in student_concept_scores.items():
    if student_id in id_to_index['students']:
        student_idx = id_to_index['students'][student_id]
        for concept_id, score in concept_scores.items():
            if concept_id in id_to_index['concepts']:
                concept_idx = id_to_index['concepts'][concept_id]
                student_features[student_idx, concept_idx] = score

graph.nodes['student'].data['feat'] = torch.tensor(student_features, dtype=torch.float32)

# 10. 중단원 노드 특성 설정
with open("embedding/reduced_mchapter_embeddings_6.npy", "rb") as f:
    node_features = np.load(f, allow_pickle=True).item()

ids = node_features['ids']
embeddings = node_features['embeddings']
default_embedding = np.mean(embeddings, axis=0)
sorted_embeddings = [
    embeddings[ids.index(int(concept_id))] if str(concept_id) in map(str, ids) else default_embedding
    for concept_id in id_to_index['concepts'].keys()
]

concept_node_features = torch.tensor(np.array(sorted_embeddings), dtype=torch.float32)
graph.nodes['concept'].data['feat'] = concept_node_features

# 11. 강의 노드 특성 설정
lecture_embeddings_path = "embedding/reduced_lecture_embeddings_23.npy"
lecture_node_features = np.load(lecture_embeddings_path, allow_pickle=True).item()
lecture_ids = lecture_node_features['ids']  # IDs는 문자열 또는 정수일 수 있음
lecture_embeddings = lecture_node_features['embeddings']

# 강의 ID 순서에 따라 정렬
default_lecture_embedding = np.mean(lecture_embeddings, axis=0)
lecture_features_sorted = [
    lecture_embeddings[lecture_ids.index(lecture_id)] if lecture_id in lecture_ids else default_lecture_embedding
    for lecture_id in id_to_index['lectures'].keys()
]

# 강의 노드에 임베딩 설정
graph.nodes['lecture'].data['feat'] = torch.tensor(np.array(lecture_features_sorted), dtype=torch.float32)

# 12. 그래프 정보 출력
print(graph)
print("Student node feature shape:", graph.nodes['student'].data['feat'].shape)
print("Concept node feature shape:", graph.nodes['concept'].data['feat'].shape)
print("Lecture node feature shape:", graph.nodes['lecture'].data['feat'].shape)
# print("Edge weights for 'understands':", graph.edges['understands'].data['weight'][:5])

def generate_graph():
    return graph

# === 그래프 준비 ===
g = generate_graph()

# 그래프 구조 확인
print("Node types:", g.ntypes)
print("Edge types:", g.etypes)
print("Graph schema:", g)

print("Student features:", g.nodes['student'].data.keys())
print("Concept features:", g.nodes['concept'].data.keys())
print("Lecture features:", g.nodes['lecture'].data.keys())


