import dgl  # DGL(DiGL) 라이브러리, 그래프 처리 및 학습용 도구
import torch as th  # PyTorch, 딥러닝 모델 구현과 학습을 위한 라이브러리
import torch.nn as nn  # 신경망 구축을 위한 PyTorch 모듈
import torch.nn.functional as F  # 활성화 함수 및 추가 연산을 위한 PyTorch 모듈
from sklearn.decomposition import PCA  # 주성분 분석(PCA)을 위한 라이브러리, 차원 축소용
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
import numpy as np  # 수치 연산 및 배열 처리를 위한 라이브러리
from dgl.nn import HeteroGraphConv, SAGEConv  # DGL에서 제공하는 GNN 계층(SAGEConv, HeteroGraphConv)
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler  # DataLoader, 샘플링 도구
from graph_generator3 import generate_graph  # 기존 그래프 생성 함수 가져오기

# === Over-sample teaches edges by duplicating or adding synthetic connections ===
def add_teaches_edges(graph, num_edges=1000):
    concept_nodes = graph.nodes('concept')
    lecture_nodes = graph.nodes('lecture')

    # 랜덤으로 연결 생성
    src = np.random.choice(concept_nodes, num_edges)
    dst = np.random.choice(lecture_nodes, num_edges)

    graph.add_edges(src, dst, etype=('concept', 'teaches', 'lecture'))
    return graph

# === Oversample nodes ===
def oversample_nodes(graph, node_type, oversample_factor):
    """
    Oversample nodes of a specific type by duplicating them in the graph.
    """
    original_nodes = graph.nodes(node_type)
    oversampled_nodes = original_nodes.repeat(oversample_factor)
    return oversampled_nodes

# === GraphSAGE 모델 정의 ===
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, dropout):
        super().__init__()  # 부모 클래스 초기화
        self.layers = nn.ModuleList()  # GNN 계층을 저장하는 리스트

        # 첫 번째 GNN 계층 정의
        self.layers.append(HeteroGraphConv({
            'understands': SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean'),
            'teaches': SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        }, aggregate='mean'))  # 각 엣지 타입에 대해 개별적인 SAGEConv 정의 후 결합

        # 각 노드 타입에 대해 입력 피처를 변환하는 선형 계층 정의
        self.fc_student = nn.Linear(in_feats['student'], hidden_feats)  # 학생 노드의 입력 피처 변환
        self.fc_concept = nn.Linear(in_feats['concept'], hidden_feats)  # 개념 노드의 입력 피처 변환
        self.fc_lecture = nn.Linear(in_feats['lecture'], hidden_feats)  # 강의 노드의 입력 피처 변환

        # 추가 GNN 계층을 정의 (num_layers 수만큼 반복)
        for _ in range(num_layers - 1):  # 첫 번째 계층 이후로 숨김층 추가
            self.layers.append(HeteroGraphConv({
                'understands': SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean'),
                'teaches': SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
            }))

        # 출력 계층 정의
        self.layers.append(HeteroGraphConv({
            'understands': SAGEConv(hidden_feats, out_feats, aggregator_type='mean'),
            'teaches': SAGEConv(hidden_feats, out_feats, aggregator_type='mean')
        }))

        # 드롭아웃 계층 정의
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        student_feats = self.fc_student(features['student'])
        concept_feats = self.fc_concept(features['concept'])
        lecture_feats = self.fc_lecture(features['lecture'])

        features = {
            'student': student_feats,
            'concept': concept_feats,
            'lecture': lecture_feats
        }

        for i, layer in enumerate(self.layers):
            x = layer(g, features)
            if i != len(self.layers) - 1:
                x = {ntype: F.relu(feat) for ntype, feat in x.items()}
                x = {ntype: self.dropout(feat) for ntype, feat in x.items()}
            features = x

        return x

# === Main Code ===
if __name__ == "__main__":
    g = generate_graph()

    # Step 1: Add more teaches edges to improve connectivity
    g = add_teaches_edges(g, num_edges=10000)

    # Step 2: Oversample nodes to ensure lecture nodes are sampled more frequently
    oversampled_concept_nodes = oversample_nodes(g, 'concept', oversample_factor=20)
    oversampled_lecture_nodes = oversample_nodes(g, 'lecture', oversample_factor=20)

    print("Total teaches edges in graph:", g.num_edges(('concept', 'teaches', 'lecture')))

    # Step 3: Adjust sampler to force lecture inclusion
    sampler = dgl.dataloading.MultiLayerNeighborSampler([
    {
        ('student', 'understands', 'concept'): 10,
        ('concept', 'teaches', 'lecture'): 0
    },  # Block 0: understands 엣지만 샘플링
    {
        ('student', 'understands', 'concept'): 0,
        ('concept', 'teaches', 'lecture'): 5
    }   # Block 1: teaches 엣지만 샘플링
    ])

    # Step 4: Include oversampled nodes in dataloader
    dataloader = DataLoader(
        g,
        {
            'student': g.nodes('student'),
            'concept': oversampled_concept_nodes,
            'lecture': oversampled_lecture_nodes
        },
        sampler,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        print(f"--- Step {step} ---")
        for i, block in enumerate(blocks):
            print(f"--- Block {i} ---")
            print("Edge types sampled in this block:", block.etypes)
            print("Source nodes in block:", block.srcdata['_ID'])
            print("Destination nodes in block:", block.dstdata['_ID'])
            print(f"Understands edges in block: {block.num_edges(('student', 'understands', 'concept'))}")
            print(f"Teaches edges in block: {block.num_edges(('concept', 'teaches', 'lecture'))}")
