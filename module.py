from torch_geometric import nn as gnn
import torch
from torch import nn
from torch.nn import functional as F


# Internal graph convolution feature module
class SubGcnFeature(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.gcn1 = gnn.SGConv(c_in, hidden_size, K=3)
        self.gcn2 = gnn.SGConv(hidden_size, 64, K=3)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 64 // 4),
            nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(64 // 4, nc)
        )

    def forward(self, graph):
        h = F.relu(self.gcn1(graph.x, graph.edge_index))
        h_avg = gnn.global_mean_pool(h, graph.batch)
        h_gcn2 = F.relu(self.gcn2(h, graph.edge_index))
        logits = self.classifier(h_gcn2)
        return h_avg, logits


# External graph convolution
class GraphNet(nn.Module):
    def __init__(self, c_in, hidden_size, nc):
        super().__init__()
        self.bn_0 = gnn.BatchNorm(c_in)
        self.gcn_1 = gnn.GCNConv(c_in, hidden_size)
        self.bn_1 = gnn.BatchNorm(hidden_size)
        self.gcn_2 = gnn.GraphConv(hidden_size, 64)
        self.bn_2 = gnn.BatchNorm(64)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 64 // 4),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64 // 4, nc)
        )

    def forward(self, graph):
        x_normalization = self.bn_0(graph.x)
        h = self.gcn_1(x_normalization, graph.edge_index)
        h = self.bn_1(F.relu(h))
        h = F.relu(self.gcn_2(h, graph.edge_index))
        # logits = self.classifier(h + x_normalization)
        logits = self.classifier(h)
        return logits

# 1DCNN
class CNN(nn.Module):
    def __init__(self, c_in, hidden_size_1d_cnn, nc):
        super().__init__()


        self.conv1d1 = nn.Conv1d(c_in, hidden_size_1d_cnn, kernel_size=10)
        self.conv1d2 = nn.Conv1d(hidden_size_1d_cnn, 256, kernel_size=10)
        self.conv1d3 = nn.Conv1d(256, 64, kernel_size=11)
        self.conv1d4 = nn.Conv1d(64, 32, kernel_size=11)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32, 32 // 2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32 // 2, nc)
        )

    def forward(self, graph):
        # 1D CNN forward pass
        x = graph.x
        x = x.view(32185, 1, 39)
        h_1d_cnn1 = F.relu(self.conv1d1(x))
        h_1d_cnn2 = F.relu(self.conv1d2(h_1d_cnn1))
        h_1d_cnn3 = F.relu(self.conv1d3(h_1d_cnn2))
        h_1d_cnn4 = F.relu(self.conv1d4(h_1d_cnn3))
        logits = self.classifier(h_1d_cnn4.view(32185, 32))

        return logits



