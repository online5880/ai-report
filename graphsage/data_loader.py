import json
import dgl
import torch
import numpy as np

def load_graph(json_path="graph_schema.json"):
    """그래프 정보(노드-특성 포함함, 엣지)가 담긴 JSON 파일로부터 DGL 그래프를 로드"""
    with open(json_path, "r") as file:
        data = json.load(file)

    nodes = data["nodes"]
    edges = data["edges"]

    # 노드 ID 매핑
    id_to_idx = {node["id"]: idx for idx, node in enumerate(nodes)}

    # 노드 임베딩 배열 생성
    node_features = np.array([node["embedding"] for node in nodes])

    # 누락된 노드를 제외한 엣지 목록 생성
    filtered_edges = [
        edge for edge in edges
        if edge["source"] in id_to_idx and edge["target"] in id_to_idx
    ]

    source_nodes = [id_to_idx[edge["source"]] for edge in filtered_edges]
    target_nodes = [id_to_idx[edge["target"]] for edge in filtered_edges]

    # DGL 그래프 생성
    g = dgl.graph((source_nodes, target_nodes))

    # 노드 특성 추가
    valid_node_features = torch.tensor(node_features, dtype=torch.float32)[:g.num_nodes()]
    g.ndata['features'] = valid_node_features

    return g
