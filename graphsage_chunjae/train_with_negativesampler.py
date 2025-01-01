import torch
import torch.nn as nn
import torch.optim as optim
from neighbor_sampling import GraphSAGE, oversample_nodes, CustomNegativeSampler
from graph_generator3 import generate_graph
from dgl.dataloading import DataLoader
from neighbor_sampling import get_sampler

# === Meta Path Loss Function ===
def meta_path_loss(student_embeddings, concept_embeddings, lecture_embeddings, blocks):
    loss = 0
    for block in blocks:
        student_to_concept = block.edges['understands'].data['weight']
        concept_to_lecture = block.edges['teaches'].data['weight']
        meta_path_score = torch.sum(student_to_concept * concept_to_lecture, dim=-1)
        loss += -torch.log(torch.sigmoid(meta_path_score)).mean()
    return loss

# === Train function ===
def train_model():
    g = generate_graph()
    oversampled_concept_nodes = oversample_nodes(g, 'concept', oversample_factor=30)
    oversampled_lecture_nodes = oversample_nodes(g, 'lecture', oversample_factor=30)

    in_feats = {'student': 34, 'concept': 6, 'lecture': 23}
    model = GraphSAGE(in_feats, hidden_feats=128, out_feats=64, num_layers=3, dropout=0.5)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    neg_sampler = CustomNegativeSampler(k=2)
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

    for epoch in range(10):
        epoch_loss = 0
        for step, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            input_features = {ntype: blocks[0].srcdata['feat'][ntype] for ntype in blocks[0].srcdata['feat']}
            embeddings = model(blocks, input_features)

            pos_graph = blocks[0]
            neg_graph = neg_sampler(pos_graph, None)
            supervised_loss = -torch.log(torch.sigmoid(pos_graph.edges['understands'].data['weight'])).mean()
            meta_loss = meta_path_loss(embeddings['student'], embeddings['concept'], embeddings['lecture'], blocks)
            loss = supervised_loss + meta_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

if __name__ == "__main__":
    train_model()
