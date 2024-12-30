import dgl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from dgl.nn import HeteroGraphConv, SAGEConv
from dgl.dataloading import DataLoader, HeteroNeighborSampler
from graph_generator3 import generate_graph

# === GraphSAGE 모델 정의 ===
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, dropout):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(HeteroGraphConv({
            'understands': SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean'),
            'teaches': SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
        }, aggregate='mean'))

        self.fc_student = nn.Linear(in_feats['student'], hidden_feats)
        self.fc_concept = nn.Linear(in_feats['concept'], hidden_feats)
        self.fc_lecture = nn.Linear(in_feats['lecture'], hidden_feats)

        for _ in range(num_layers - 1):
            self.layers.append(HeteroGraphConv({
                'understands': SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean'),
                'teaches': SAGEConv(hidden_feats, hidden_feats, aggregator_type='mean')
            }))

        self.layers.append(HeteroGraphConv({
            'understands': SAGEConv(hidden_feats, out_feats, aggregator_type='mean'),
            'teaches': SAGEConv(hidden_feats, out_feats, aggregator_type='mean')
        }))

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
    g = generate_graph()
    print("=== Initial Graph Information ===")
    print("Graph schema:", g)
    print("Number of edges (concept -> lecture):", g.num_edges(('concept', 'teaches', 'lecture')))

    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    # HeteroNeighborSampler 설정
    sampler = HeteroNeighborSampler({
        ('student', 'understands', 'concept'): 50,  # 'understands' 엣지 이웃 샘플링
        ('concept', 'teaches', 'lecture'): 20      # 'teaches' 엣지 이웃 샘플링
    })

    dataloader = DataLoader(
        g,
        {'student': g.nodes('student')},  # 타겟 노드 설정
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
            if 'lecture' in block.ntypes:
                print("Lecture nodes in current block:", block.num_nodes('lecture'))
            else:
                print("No lecture nodes sampled in current block.")

            teaches_edges = block.num_edges(('concept', 'teaches', 'lecture')) if ('concept', 'teaches', 'lecture') in block.etypes else 0
            print(f"Teaches edges in block: {teaches_edges}")
