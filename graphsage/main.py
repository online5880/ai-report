import os
import torch
import numpy as np
import dgl
from sklearn.metrics import roc_auc_score
from data_loader import load_graph
from model import ConceptGraphSAGE
from training import train_model
from recommendation import recommend_concepts, compute_similarities

def get_positive_and_negative_edges(g, embeddings):
    # Positive edges
    pos_edges = g.edges()
    pos_src, pos_dst = pos_edges[0], pos_edges[1]
    pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)

    # Negative edges
    neg_src, neg_dst = dgl.sampling.global_uniform_negative_sampling(
        g, num_samples=len(pos_src)
    )
    neg_scores = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

    return pos_scores, neg_scores

def compute_auc(model, g, features):
    model.eval()
    with torch.no_grad():
        embeddings = model(g, features)

    pos_scores, neg_scores = get_positive_and_negative_edges(g, embeddings)

    # Labels and scores
    labels = torch.cat([torch.ones(len(pos_scores)), torch.zeros(len(neg_scores))]).numpy()
    scores = torch.cat([pos_scores, neg_scores]).numpy()

    # Compute AUC
    return roc_auc_score(labels, scores)

def mean_cosine_similarity(target_idx, recommended, similarities, k):
    """
    Mean Cosine Similarity@K 계산
    """
    top_k_similarities = [similarities[target_idx, rec_idx] for rec_idx in recommended[:k]]
    return np.mean(top_k_similarities)

def weighted_mean_similarity(target_idx, recommended, similarities, k):
    """
    Weighted Mean Cosine Similarity@K 계산
    """
    weights = [1 / np.log2(i + 2) for i in range(k)]
    top_k_similarities = [similarities[target_idx, rec_idx] for rec_idx in recommended[:k]]
    weighted_sum = np.sum([w * sim for w, sim in zip(weights, top_k_similarities)])
    return weighted_sum / np.sum(weights)

def main():
    # 1. 그래프 및 데이터 준비
    print("그래프를 로드 중입니다...")
    g = load_graph()
    features = g.ndata['features'] 

    # 2. 모델 초기화 및 학습
    model = ConceptGraphSAGE(features.shape[1], 64, 32)
    train_model(model, g, features, epochs=500, lr=0.001)

    # 3. 학습된 모델 저장 및 노드 임베딩 계산
    torch.save(model.state_dict(), "./trained_model.pth")
    embeddings = model(g, features).detach().numpy()
    np.save("./trained_node_embeddings.npy", embeddings)

    # 4. AUC 계산
    auc = compute_auc(model, g, features)
    print(f"AUC: {auc}")

    # 5. Mean Cosine Similarity 및 Weighted Mean Similarity 계산
    similarities = compute_similarities(embeddings)
    target_concept = 0  # 예: 첫 번째 노드
    # 타겟 노드 제외한 상위 5개 추천 노드 생성
    recommended = [idx for idx in np.argsort(-similarities[target_concept]) if idx != target_concept][:5]
    mean_sim = mean_cosine_similarity(target_concept, recommended, similarities, k=5)
    weighted_sim = weighted_mean_similarity(target_concept, recommended, similarities, k=5)

    print(f"Mean Cosine Similarity@5: {mean_sim}")
    print(f"Weighted Mean Similarity@5: {weighted_sim}")

if __name__ == "__main__":
    main()

