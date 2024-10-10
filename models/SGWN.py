import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, global_mean_pool



class SGWConv(nn.Module):
    def __init__(self, in_features, out_features, r, num_nodes,  bias=True):
        super(SGWConv, self).__init__()
        # self.Lev = Lev
        # self.crop_len = (Lev - 1) * num_nodes
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(r * num_nodes, 1).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * num_nodes, 1))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        x = torch.matmul(x, self.weight)

        # Decomposition
        x = torch.sparse.mm(torch.cat(d_list, dim=0), x)

        # Hadamard product in spectral domain
        x = self.filter * x

        # Reconstruction
        x = torch.sparse.mm(torch.cat(d_list, dim=1), x)
        if self.bias is not None:
            x += self.bias

        return x


class SGWN(nn.Module):
    def __init__(self, feature, nhid, out_channel, r, num_nodes, dropout_prob=0.5):
        super(SGWN, self).__init__()
        self.GConv1 = SGWConv(feature, nhid, r, num_nodes)
        self.Bn1 = BatchNorm(1024)
        self.GConv2 = SGWConv(nhid, 1024, r, num_nodes)
        self.Bn2 = BatchNorm(1024)

        self.fc = nn.Sequential(nn.Linear(1024, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Dropout(dropout_prob),
                                nn.Linear(512, out_channel),
                                nn.BatchNorm1d(out_channel))

    def forward(self, data, d_list):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x has shape [num_nodes, num_input_features]

        x = self.GConv1(x, d_list)
        x = self.Bn1(x)
        x = F.relu(x)

        x = self.GConv2(x, d_list)
        x = self.Bn2(x)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc(x)

        return x
