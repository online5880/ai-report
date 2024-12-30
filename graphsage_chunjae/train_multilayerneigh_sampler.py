import dgl  # DGL(DiGL) 라이브러리, 그래프 처리 및 학습용 도구
import torch as th  # PyTorch, 딥러닝 모델 구현과 학습을 위한 라이브러리
import torch.nn as nn  # 신경망 구축을 위한 PyTorch 모듈
import torch.nn.functional as F  # 활성화 함수 및 추가 연산을 위한 PyTorch 모듈
from sklearn.decomposition import PCA  # 주성분 분석(PCA)을 위한 라이브러리, 차원 축소용
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
import numpy as np  # 수치 연산 및 배열 처리를 위한 라이브러리
from dgl.nn import HeteroGraphConv, SAGEConv  # DGL에서 제공하는 GNN 계층(SAGEConv, HeteroGraphConv)
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler  # DGL의 데이터 로더 및 샘플링 도구
from graph_generator3 import generate_graph  # 기존 그래프 생성 함수 가져오기

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

# === 메인 코드 ===
if __name__ == "__main__":
    g = generate_graph()  # 기존 graph_generator에서 그래프 생성 함수 호출
    print("=== Initial Graph Information ===")
    print("Graph schema:", g)
    print("Number of edges (concept -> lecture):", g.num_edges(('concept', 'teaches', 'lecture')))
    
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    sampler = MultiLayerNeighborSampler([
        {
            ('student', 'understands', 'concept'): 50,  # 1-hop 'understands' 샘플링
            ('concept', 'teaches', 'lecture'): 20       # 1-hop 'teaches' 샘플링
        },
        {
            ('student', 'understands', 'concept'): 30, # 2-hop 'understands' 샘플링
            ('concept', 'teaches', 'lecture'): 10       # 2-hop 'teaches' 샘플링
        }
    ])


    dataloader = DataLoader(
        g,
        {'student': g.nodes('student')},  # 타겟 노드
        sampler,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    print("=== Graph Information ===")
    print("Number of concept nodes:", g.num_nodes('concept'))
    print("Number of lecture nodes:", g.num_nodes('lecture'))
    print("Number of edges (concept -> lecture):", g.num_edges(('concept', 'teaches', 'lecture')))

    src, dst = g.edges(etype=('concept', 'teaches', 'lecture'))
    print("Source concept nodes:", src)
    print("Destination lecture nodes:", dst)

    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        print(f"--- Step {step} ---")
        print("Input nodes:", input_nodes)
        print("Output nodes:", output_nodes)
        for i, block in enumerate(blocks):
            print(f"--- Block {i} ---")
            for etype in block.etypes:
                print(f"Edge type: {etype}, Number of edges: {block.num_edges(etype)}")
                # Lecture 노드와 teaches 엣지 디버깅
            if 'lecture' in block.ntypes:
                print("Lecture nodes in current block:", block.num_nodes('lecture'))
            else:
                print("No lecture nodes sampled in current block.")
            
            teaches_edges = block.num_edges(('concept', 'teaches', 'lecture')) if ('concept', 'teaches', 'lecture') in block.etypes else 0
            print(f"Teaches edges in block: {teaches_edges}")

        # # 블록 데이터에서 피처 추출
        # try:
        #     features = {
        #         ntype: blocks[0].srcdata['feat'][ntype].to(device)  # 'feat' 아래의 ntype 접근
        #         for ntype in blocks[0].srcdata['feat'].keys()  # 'feat' 키 안의 노드 타입 반복
        #     }
        #     print("Extracted features for each node type:", {ntype: feat.shape for ntype, feat in features.items()})
        # except KeyError as e:
        #     print(f"KeyError occurred: {e}")
        #     print("Available srcdata keys and content:")
        #     for ntype, data in blocks[0].srcdata.items():
        #         print(f"Node type: {ntype}, Keys: {data.keys()}")
        #     raise