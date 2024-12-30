# import json
# import dgl
# import torch as th


# def load_graph_from_json(json_file):
#     with open(json_file, 'r') as f:
#         data = json.load(f)

#     # 노드와 엣지 데이터 추출
#     nodes = data["nodes"]
#     edges = data["edges"]

#     # 노드 ID와 타입 추출
#     node_ids = [node["id"] for node in nodes]
#     node_types = [node["type"] for node in nodes]

#     # 노드 타입을 숫자로 매핑
#     type_mapping = {t: i for i, t in enumerate(set(node_types))}
#     type_data = [type_mapping[t] for t in node_types]

#     # 엣지 데이터
#     src_ids = [edge["src"] for edge in edges]
#     dst_ids = [edge["dst"] for edge in edges]

#     # DGL 그래프 생성
#     g = dgl.graph((src_ids, dst_ids))
#     g.ndata["type"] = th.tensor(type_data)  # 노드 타입 추가
#     print(f"Loaded graph with {g.num_nodes()} nodes and {g.num_edges()} edges.")
#     return g, type_mapping

# load_graph_from_json('G.json')

import json
import dgl
import torch as th

def load_graph_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 노드와 엣지 데이터 추출
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    # 노드 ID와 타입 추출
    node_ids = [node["id"] for node in nodes]
    node_types = [node["type"] for node in nodes]

    # 노드 ID와 타입을 숫자로 매핑
    id_mapping = {nid: i for i, nid in enumerate(node_ids)}
    type_mapping = {t: i for i, t in enumerate(set(node_types))}
    type_data = [type_mapping[t] for t in node_types]

    # 엣지 데이터 변환
    src_ids = []
    dst_ids = []
    weights = []  # 가중치가 없는 엣지의 경우 기본값으로 설정

    for edge in edges:
        src = edge.get("source")  # 'source'로 처리
        dst = edge.get("target")  # 'target'으로 처리
        weight = edge.get("weight", 1.0)  # 가중치가 없으면 기본값 1.0 사용

        # 노드 ID를 숫자로 변환
        if src in id_mapping and dst in id_mapping:
            src_ids.append(id_mapping[src])
            dst_ids.append(id_mapping[dst])
            weights.append(weight)

    # DGL 그래프 생성
    g = dgl.graph((src_ids, dst_ids))
    g.ndata["type"] = th.tensor(type_data)  # 노드 타입 추가
    g.edata["weight"] = th.tensor(weights, dtype=th.float32)  # 엣지 가중치 추가
    print(f"Loaded graph with {g.num_nodes()} nodes and {g.num_edges()} edges.")
    return g, type_mapping

# JSON 파일 불러오기
graph, type_mapping = load_graph_from_json('G.json')
print("Type mapping:", type_mapping)


