import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

# GraphSAGE 모델 정의
class ConceptGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(ConceptGraphSAGE, self).__init__()
        # 집계함수로 pool 사용: 이외 mean, gcn, lstm 등 존재
        self.conv1 = dglnn.SAGEConv(in_feats, hidden_feats, aggregator_type='pool')
        self.conv2 = dglnn.SAGEConv(hidden_feats, out_feats, aggregator_type='pool')

    def forward(self, g, features):
        # 순전파
        h = self.conv1(g, features)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# 테스트용 코드
if __name__ == "__main__":
    print("GraphSAGE 모델 정의 완료.")