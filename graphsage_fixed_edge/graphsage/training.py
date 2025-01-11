import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import dgl

def link_prediction_loss_hinge(g, embeddings):
    """링크 예측용 힌지 손실 계산"""
    pos_edges = g.edges()
    pos_src, pos_dst = pos_edges[0], pos_edges[1]
    # 긍정 엣지 점수 계산
    pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)

    # 긍정 엣지와 같은 수의 부정 엣지(negative edges) 샘플링링
    neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
        g, num_samples=len(pos_src)  # 긍정 엣지 수와 동일한 개수 샘플링링
    )

    # 부정 엣지 점수 계산 (내적)
    neg_scores = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)
    
    # 긍정 점수와 부정 점수의 크기가 동일하도록 조정정
    min_size = min(len(pos_scores), len(neg_scores))
    pos_scores = pos_scores[:min_size]
    neg_scores = neg_scores[:min_size]

    # 힌지 손실 함수 계산
    loss = torch.mean(F.relu(1.0 - pos_scores + neg_scores))
    
    # 손실값과 긍정, 부정 엣지의 평균 점수 반환
    return loss, pos_scores.mean().item(), neg_scores.mean().item()


def train_model(model, g, features, epochs, lr):
    """모델 학습 함수"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_values = []
    pos_score_values = []
    neg_score_values = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        embeddings = model(g, features)
        loss, pos_score_mean, neg_score_mean = link_prediction_loss_hinge(g, embeddings)
        loss.backward()
        optimizer.step()
        
        # 에포크별 손실 및 점수 저장장
        loss_values.append(loss.item())
        pos_score_values.append(pos_score_mean)
        neg_score_values.append(neg_score_mean)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
              f"Positive Score: {pos_score_mean:.4f}, Negative Score: {neg_score_mean:.4f}, "
              f"Score Difference: {pos_score_mean - neg_score_mean:.4f}")

    # 학습 손실값 변화 그래프
    plt.plot(range(1, epochs + 1), loss_values, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Reduction Over Epochs")
    plt.legend()
    plt.show()

    # 긍정 엣지와 부정 엣지 점수 차이 시각화
    score_differences = [p - n for p, n in zip(pos_score_values, neg_score_values)]
    plt.plot(range(1, epochs + 1), score_differences, label="Positive-Negative Score Difference")
    plt.xlabel("Epochs")
    plt.ylabel("Score Difference")
    plt.title("Positive vs. Negative Score Difference Over Epochs")
    plt.legend()
    plt.show()