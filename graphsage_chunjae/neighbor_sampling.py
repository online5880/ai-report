import dgl  # DGL(DiGL) 라이브러리, 그래프 처리 및 학습용 도구
import torch as th  # PyTorch, 딥러닝 모델 구현과 학습을 위한 라이브러리
import torch.nn as nn  # 신경망 구축을 위한 PyTorch 모듈
import torch.nn.functional as F  # 활성화 함수 및 추가 연산을 위한 PyTorch 모듈
import numpy as np  # 수치 연산 및 배열 처리를 위한 라이브러리
from dgl.nn import HeteroGraphConv, SAGEConv  # DGL에서 제공하는 GNN 계층(SAGEConv, HeteroGraphConv)
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler  # DataLoader, 샘플링 도구
from graph_generator3 import generate_graph  # 기존 그래프 생성 함수 가져오기

# === Custom Negative Sampler ===
class CustomNegativeSampler:
    def __init__(self, k):
        self.k = k

    def __call__(self, pos_graph, pos_eids):
        neg_src, neg_dst = [], []
        for etype in pos_graph.canonical_etypes:
            if etype in [('student', 'understands', 'concept'), ('concept', 'teaches', 'lecture')]:
                src, dst = pos_graph.edges(form='eid', etype=etype)
                neg_src.append(src.repeat(self.k))
                neg_dst.append(dst.repeat(self.k))
        neg_graph = dgl.heterograph({
            etype: (src, dst)
            for etype, (src, dst) in zip(pos_graph.canonical_etypes, zip(neg_src, neg_dst))
        }, num_nodes_dict=pos_graph.num_nodes())
        return neg_graph

# === Oversample nodes ===
def oversample_nodes(graph, node_type, oversample_factor):
    """
    Oversample nodes of a specific type by duplicating them in the graph.
    """
    original_nodes = graph.nodes(node_type)
    oversampled_nodes = original_nodes.repeat(oversample_factor)
    return oversampled_nodes

def get_sampler():
    """
    Multi-layer neighbor sampler 설정을 반환
    """
    return MultiLayerNeighborSampler([
        {
            ('student', 'understands', 'concept'): 50,
            ('concept', 'teaches', 'lecture'): 300
        },  # Block 0: understands 엣지만 샘플링
        {
            ('student', 'understands', 'concept'): 50,
            ('concept', 'teaches', 'lecture'): 300
        }   # Block 1: teaches 엣지만 샘플링
    ])

def validate_edges(block, min_teaches=10, min_understands=20):
    teaches_count = block.num_edges('teaches')
    understands_count = block.num_edges('understands')
    
    if teaches_count < min_teaches or understands_count < min_understands:
        print(f"Block skipped due to insufficient edges: teaches={teaches_count}, understands={understands_count}")
        return False
    return True


# 샘플링 반복 로직에 조건 추가
def repeat_sampling(dataloader, min_teaches=10, min_understands=20, max_retries=5):
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        retries = 0
        valid_blocks = []
        
        while retries < max_retries and not valid_blocks:
            valid_blocks = [block for block in blocks if validate_edges(block, min_teaches, min_understands)]
            
            if not valid_blocks:
                retries += 1
                print(f"Retrying sampling... (Attempt {retries}/{max_retries})")
                input_nodes, output_nodes, blocks = next(dataloader)

        if not valid_blocks:
            print(f"Skipping batch {step} after {max_retries} retries due to insufficient edges.")
            continue

        yield input_nodes, output_nodes, valid_blocks

# === GraphSAGE 모델 정의 ===
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, dropout):
        super().__init__()  # 부모 클래스 초기화
        self.layers = nn.ModuleList()  # GNN 계층을 저장하는 리스트

        # 첫 번째 GNN 계층 정의
        self.layers.append(HeteroGraphConv({
            ('student', 'understands', 'concept'): SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean'),
            ('concept', 'teaches', 'lecture'): SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        }, aggregate='mean'))  # 각 엣지 타입에 대해 개별적인 SAGEConv 정의 후 결합

        # 각 노드 타입에 대해 입력 피처를 변환하는 선형 계층 정의
        self.fc_student = nn.Linear(in_feats['student'], hidden_feats)  # 학생 노드의 입력 피처 변환
        self.fc_concept = nn.Linear(in_feats['concept'], hidden_feats)  # 개념 노드의 입력 피처 변환
        self.fc_lecture = nn.Linear(in_feats['lecture'], hidden_feats)  # 강의 노드의 입력 피처 변환

        # 추가 GNN 계층을 정의 (num_layers 수만큼 반복)
        for _ in range(num_layers - 1):  # 첫 번째 계층 이후로 숨김층 추가
            self.layers.append(HeteroGraphConv({
                ('student', 'understands', 'concept'): SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean'),
                ('concept', 'teaches', 'lecture'): SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
            }))

        # 출력 계층 정의
        self.layers.append(HeteroGraphConv({
            ('student', 'understands', 'concept'): SAGEConv(hidden_feats, out_feats, aggregator_type='mean'),
            ('concept', 'teaches', 'lecture'): SAGEConv(hidden_feats, out_feats, aggregator_type='mean')
        }))

        # 드롭아웃 계층 정의
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, features):
        # 블록 리스트의 첫 번째 블록에서 입력 특성 변환
        print(f"Initial features keys: {features.keys()}")

        # 디버깅 코드 추가 - srcdata와 features 크기 확인
        for ntype in blocks[0].srcdata.keys():
            print(f"Block 0 srcdata[{ntype}] shape: {blocks[0].srcdata['_ID'][ntype].shape}")
            print(f"Features[{ntype}] shape: {features[ntype].shape}")

        # 각 블록의 srcdata['_ID']를 사용해 피처 정렬
        for ntype in features.keys():
            if ntype in blocks[0].srcdata['_ID']:
                indices = blocks[0].srcdata['_ID'][ntype]
                if indices.max() >= features[ntype].shape[0]:
                    print(f"Warning: Invalid indices for {ntype}. Skipping those out of range.")
                    indices = indices[indices < features[ntype].shape[0]]  # 유효한 인덱스만 선택
                features[ntype] = features[ntype][indices]

        # 기존 forward 코드 계속 진행
        student_feats = self.fc_student(features['student'])
        print(f"Student features shape: {student_feats.shape}")
        concept_feats = self.fc_concept(features['concept'])
        lecture_feats = self.fc_lecture(features['lecture'])

        features = {
            'student': student_feats,
            'concept': concept_feats,
            'lecture': lecture_feats
        }

        # 각 블록을 순회하며 처리
        for i, (block, layer) in enumerate(zip(blocks, self.layers)):
            x = layer(block, features)  # 각 블록에 대해 HeteroGraphConv 호출
            print(f"Block {i} output keys: {x.keys()}")  # Debugging: 각 블록의 출력 확인
            
            # 학생 노드의 출력 누락 처리
            if 'student' not in x:
                print(f"Warning: 'student' key missing in Block {i} output.")
                x['student'] = features['student']
            
            # 마지막 레이어가 아닌 경우 활성화 함수 및 드롭아웃 적용
            if i != len(self.layers) - 1:
                x = {ntype: F.relu(feat) for ntype, feat in x.items()}
                x = {ntype: self.dropout(feat) for ntype, feat in x.items()}
            features = x  # 업데이트된 피처를 다음 블록으로 전달

        return features  # 마지막 블록의 결과 반환


def filter_block_edges(blocks):
    filtered_blocks = []
    for i, block in enumerate(blocks):
        teaches_valid = block.num_edges('teaches') > 0
        understands_valid = block.num_edges('understands') > 0

        # 하나의 엣지가 유효하다면 블록을 추가
        if not (teaches_valid or understands_valid):
            print(f"Skipping Block {i} due to no valid edges.")
            continue

        print(f"Block {i} has valid edges. Proceeding.")
        filtered_blocks.append(block)

    return filtered_blocks


# === Main Code ===
if __name__ == "__main__":
    g = generate_graph()

    # Step 2: Oversample nodes to ensure lecture nodes are sampled more frequently
    oversampled_concept_nodes = oversample_nodes(g, 'concept', oversample_factor=30)
    oversampled_lecture_nodes = oversample_nodes(g, 'lecture', oversample_factor=30)

        # 결과 확인
    print("Oversampled concept nodes:", oversampled_concept_nodes)
    print(f"Oversampled concept nodes count: {len(oversampled_concept_nodes)}")

    print("Oversampled lecture nodes:", oversampled_lecture_nodes)
    print(f"Oversampled lecture nodes count: {len(oversampled_lecture_nodes)}")


    # Step 3: Adjust sampler to force lecture inclusion
    sampler = get_sampler()

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
    print("DataLoader initialized with oversampled nodes.")
    print(f"Concept nodes (oversampled): {oversampled_concept_nodes}")
    print(f"Lecture nodes (oversampled): {oversampled_lecture_nodes}")
    
    for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        # print(f"--- Step {step} ---")
        # print("Input nodes:", input_nodes)
        # print("Output nodes:", output_nodes)

        blocks = filter_block_edges(blocks)  # Filter blocks with no 'teaches' edges
        if not blocks:  # Skip batch if no valid blocks
            print(f"Skipping batch {step} due to no valid blocks.")
            continue

        # 디버깅: 모든 블록의 엣지 타입과 개수 확인
        for i, block in enumerate(blocks):
            print(f"Edge types: {block.etypes}")
            for etype in block.etypes:
                print(f"Edge type: {etype}, Count: {block.num_edges(etype)}")

            # Block 0의 srcdata 확인 (기존 코드에 포함됨)
            if i == 0:
                print(f"Block 0 srcdata keys: {block.srcdata.keys()}")
            else:
                print(f"Block 1 srcdata keys: {block.srcdata.keys()}")
