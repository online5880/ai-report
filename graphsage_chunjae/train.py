import torch
import torch.nn as nn
import torch.optim as optim
from neighbor_sampling import GraphSAGE, oversample_nodes, filter_block_edges
import dgl
from graph_generator3 import generate_graph
from dgl.dataloading import DataLoader, MultiLayerNeighborSampler
from dgl.dataloading.negative_sampler import GlobalUniform
from neighbor_sampling import get_sampler

# === Unsupervised Loss Function ===
class UnsupervisedLoss(nn.Module):
    def forward(self, pos_graph, neg_graph, node_embeddings):
        # Positive edges: 연결된 노드 쌍의 점수를 계산
        pos_score = torch.sum(
            node_embeddings[pos_graph.srcdata['_ID']] * node_embeddings[pos_graph.dstdata['_ID']], dim=-1
        )
        pos_loss = -torch.log(torch.sigmoid(pos_score)).mean()

        # Negative edges: 연결되지 않은 노드 쌍의 점수를 계산
        neg_score = torch.sum(
            node_embeddings[neg_graph.srcdata['_ID']] * node_embeddings[neg_graph.dstdata['_ID']], dim=-1
        )
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score)).mean()

        return pos_loss + neg_loss

# === Train function ===
def train_model():
    # Step 1: Graph and Data Preparation
    g = generate_graph()

    # Add more teaches edges to improve connectivity
    

    # Oversample nodes to ensure lecture nodes are sampled more frequently
    oversampled_concept_nodes = oversample_nodes(g, 'concept', oversample_factor=30)
    oversampled_lecture_nodes = oversample_nodes(g, 'lecture', oversample_factor=30)

    # Feature dimensions
    in_feats = {
        'student': 34,  # 학생 노드의 피처 수
        'concept': 6,   # 개념 노드의 피처 수
        'lecture': 23   # 강의 노드의 피처 수
    }

    hidden_feats = 128  # 숨김층 피처 크기
    out_feats = 64      # 출력 피처 크기
    num_layers = 3      # GraphSAGE 계층 수
    dropout = 0.5       # 드롭아웃 비율

    # Step 2: Model Initialization
    model = GraphSAGE(in_feats, hidden_feats, out_feats, num_layers, dropout)
    model.train()

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = UnsupervisedLoss()

    # Negative sampler
    neg_sampler = GlobalUniform(k=2)  # 각 엣지당 5개의 부정 샘플 생성

    # Step 3: Dataloader setup
    sampler = get_sampler()

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

    # Step 4: Training Loop
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0

        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            # 디버깅 코드 추가: 샘플링 결과 확인
            print(f"Epoch {epoch}, Step {step}")
            print(f"Input Nodes: {input_nodes}")
            print(f"Output Nodes: {output_nodes}")
            for i, block in enumerate(blocks):
                print(f"--- Block {i} ---")
                print(f"Edge types: {block.etypes}")
                for etype in block.etypes:
                    print(f"Edge type: {etype}, Count: {block.num_edges(etype)}")
                # 추가 디버깅: 노드 정보
                print(f"Block {i} number of src nodes: {block.num_src_nodes()}")
                print(f"Block {i} number of dst nodes: {block.num_dst_nodes()}")
                # 추가 디버깅: src와 dst의 노드 타입별 개수
                for ntype in block.srctypes:
                    print(f"Source nodes of type {ntype}: {block.num_src_nodes(ntype)}")
                for ntype in block.dsttypes:
                    print(f"Destination nodes of type {ntype}: {block.num_dst_nodes(ntype)}")

            blocks = filter_block_edges(blocks)  # Apply filter to remove unwanted edges

            if not blocks:  # 필터링 후 blocks가 비어있으면 스킵
                print(f"Skipping batch {step} due to no valid blocks.")
                continue

            print(f"--- Step {step} ---")

            input_features = {
                ntype: blocks[0].srcdata['feat'][ntype]
                for ntype in blocks[0].srcdata['feat']
            }
            # 디버깅 코드: Block 0의 src 노드 수와 Features['student']의 크기를 출력
            print(f"Block 0 number of src nodes: {blocks[0].num_src_nodes('student')}")
            print(f"Features['student'] shape: {input_features['student'].shape}")
            
            # Forward pass to get embeddings
            embeddings = model(blocks, input_features)

            # Positive and negative graphs
            pos_graph = blocks[0]  # Positive graph (1-hop)
            pos_eids = {
                etype: pos_graph.edges(form='eid', etype=etype)
                for etype in pos_graph.canonical_etypes
            }  # Extract edge IDs for all edge types
            
            if any(len(eids) == 0 for eids in pos_eids.values()):
                print("No edges available for negative sampling, skipping step.")
                continue
            
            neg_graph = neg_sampler(pos_graph, pos_eids)  # Use the negative sampler

            # Loss 계산
            loss = loss_fn(pos_graph, neg_graph, embeddings['student'])
            epoch_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    train_model()



