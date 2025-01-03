import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl

def link_prediction_loss_hinge(g, embeddings):
    """링크 예측용 힌지 손실 계산"""
    pos_edges = g.edges()
    pos_src, pos_dst = pos_edges[0], pos_edges[1]
    pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)

    neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
        g, num_samples=len(pos_src)
    )
    neg_scores = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

    loss = torch.mean(F.relu(1.0 - pos_scores + neg_scores))
    return loss

def train_model(model, g, features, epochs, lr):
    """모델 학습 함수"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_values = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(g, features)
        loss = link_prediction_loss_hinge(g, embeddings)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    plt.plot(range(1, epochs + 1), loss_values, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Reduction Over Epochs")
    plt.legend()
    plt.show()