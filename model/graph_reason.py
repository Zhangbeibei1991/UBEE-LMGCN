import torch.nn as nn
import torch
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, gcn_dim, dep_embed_dim, device, reason_nums=2, drop_rate=0.5, pooling='avg'):
        super(GraphConvLayer, self).__init__()

        self.gcn_dim = gcn_dim
        self.dep_embed_dim = dep_embed_dim
        self.pooling = pooling

        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = Edgeupdate(gcn_dim, self.dep_embed_dim, dropout_ratio=0.5)

        self.reason_module = [nn.Sequential(
            nn.Conv2d(in_channels=self.dep_embed_dim, out_channels=self.dep_embed_dim, kernel_size=(5, 5),
                      padding="same"),
            nn.BatchNorm2d(self.dep_embed_dim),
            nn.ReLU(inplace=True)).to(device=device) for _ in range(reason_nums)]
        self.drop = nn.Dropout(drop_rate)

    def forward(self, weight_adj, gcn_inputs):
        """
        :param weight_adj: [batch, seq, seq, dim_e]
        :param gcn_inputs: [batch, seq, dim]
        :return:
        """
        batch, seq, dim = gcn_inputs.shape
        weight_adj = weight_adj.permute(0, 3, 1, 2)  # [batch, dim_e, seq, seq]
        for reason_layer in self.reason_module:
            weight_adj = reason_layer(weight_adj)
        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.dep_embed_dim, seq, dim)
        Ax = torch.matmul(weight_adj, gcn_inputs)  # [batch, dim_e, seq, dim]
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        # Edge update weight_adj[batch, dim_e, seq, seq]
        weight_adj = weight_adj.permute(0, 2, 3, 1).contiguous()  # [batch, seq, seq, dim_e]
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)
        return node_outputs.squeeze(dim=0), edge_outputs.squeeze(dim=0)


class Edgeupdate(nn.Module):
    def __init__(self, hidden_dim, dim_e, dropout_ratio=0.5):
        super(Edgeupdate, self).__init__()
        self.hidden_dim = hidden_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        self.W = nn.Linear(self.hidden_dim * 2 + self.dim_e, self.dim_e)

    def forward(self, edge, node1, node2):
        """
        :param edge: [batch, seq, seq, dim_e]
        :param node: [batch, seq, seq, dim]
        :return:
        """

        node = torch.cat([node1, node2], dim=-1)  # [batch, seq, seq, dim * 2]
        edge = self.W(torch.cat([edge, node], dim=-1))
        return edge  # [batch, seq, seq, dim_e]


import dgl.function as fn
import dgl


class MixHopConv(nn.Module):
    r"""
    Description
    -----------
    MixHop Graph Convolutional layer from paper `MixHop: Higher-Order Graph Convolutional Architecturesvia Sparsified Neighborhood Mixing
     <https://arxiv.org/pdf/1905.00067.pdf>`__.
    .. math::
        H^{(i+1)} =\underset{j \in P}{\Bigg\Vert} \sigma\left(\widehat{A}^j H^{(i)} W_j^{(i)}\right),
    where :math:`\widehat{A}` denotes the symmetrically normalized adjacencymatrix with self-connections,
    :math:`D_{ii} = \sum_{j=0} \widehat{A}_{ij}` its diagonal degree matrix,
    :math:`W_j^{(i)}` denotes the trainable weight matrix of different MixHop layers.
    Parameters
    ----------
    in_dim : int
        Input feature size. i.e, the number of dimensions of :math:`H^{(i)}`.
    out_dim : int
        Output feature size for each power.
    p: list
        List of powers of adjacency matrix. Defaults: ``[0, 1, 2]``.
    dropout: float, optional
        Dropout rate on node features. Defaults: ``0``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    batchnorm: bool, optional
        If True, use batch normalization. Defaults: ``False``.
    """

    def __init__(
            self,
            in_dim,
            out_dim,
            p=[0, 1, 2],
            dropout=0,
            activation=None,
            batchnorm=False,
    ):
        super(MixHopConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.p = p
        self.activation = activation
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim * len(p))

        # define weight dict for each power j
        self.weights = nn.ModuleDict(
            {str(j): nn.Linear(in_dim, out_dim, bias=False) for j in p}
        )

    def forward(self, graph, feats):
        with graph.local_scope():
            # assume that the graphs are undirected and graph.in_degrees() is the same as graph.out_degrees()
            degs = graph.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
            max_j = max(self.p) + 1
            outputs = []
            for j in range(max_j):

                if j in self.p:
                    output = self.weights[str(j)](feats)
                    outputs.append(output)

                feats = feats * norm
                graph.ndata["h"] = feats
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                feats = graph.ndata.pop("h")
                feats = feats * norm

            final = torch.cat(outputs, dim=1)

            if self.batchnorm:
                final = self.bn(final)

            if self.activation is not None:
                final = self.activation(final)

            final = self.dropout(final)

            return final


import math


def gcn_msg(edge):
    # msg = edge.src['h'] * edge.src['norm']
    node_dim, edge_dim = edge.src['h'].size(-1), edge.data["ew"].size(-1)
    src_data = edge.src['h'].unsqueeze(dim=1).repeat([1, edge_dim, 1])
    msg = edge.data["ew"].unsqueeze(dim=-1).repeat([1, 1, node_dim]) * src_data
    return {'m': msg.view(-1, node_dim * edge_dim)}


def gcn_reduce(node):
    node_dim, edge_dim = 768, 50
    accum = torch.sum(torch.sum(node.mailbox['m'], 1).view(-1, edge_dim, node_dim), dim=1)
    return {'h': accum}


class NodeApplyModule(nn.Module):
    def __init__(self, out_feats, activation=None, bias=True):
        super(NodeApplyModule, self).__init__()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.bias = None
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            stdv = 1. / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, nodes):
        h = nodes.data['h']
        if self.bias is not None:
            h = h + self.bias
        if self.activation:
            h = self.activation(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 dropout,
                 edge_dim=50,
                 bias=True):
        super(GCNLayer, self).__init__()
        self.dim_e = edge_dim
        self.in_feats = in_feats
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = 0.
        self.node_update = NodeApplyModule(out_feats, activation, bias)
        self.W = nn.Linear(self.in_feats * 2 + self.dim_e, self.dim_e)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, h, g):
        if self.dropout:
            h = self.dropout(h)
        g.ndata['h'] = torch.mm(h, self.weight)
        g.update_all(gcn_msg, gcn_reduce, self.node_update)
        g.apply_edges(self.edge_update)
        h = g.ndata.pop('h')
        return h, g

    def edge_update(self, edge_feats):
        dist_h = edge_feats.dst["h"]
        src_h = edge_feats.src["h"]
        edge_ew = edge_feats.data['ew']
        edge_ew = self.W(torch.cat([dist_h, edge_ew, src_h], dim=-1))
        return {"ew": edge_ew}


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GCNLayer(in_feats, n_hidden, activation, dropout))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GCNLayer(n_hidden, n_hidden, activation, dropout))
        self.trans_layer = nn.Linear(n_hidden * (n_layers + 1), n_hidden)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, features):
        h = features
        outputs = [h]
        for layer in self.layers:
            h, graph = layer(h, graph)
            h = self.dropout(h)
            graph.edata["ew"] = self.dropout(graph.edata["ew"])
            outputs.append(h)

        outputs = torch.cat(outputs, dim=-1)
        h = self.dropout(self.trans_layer(outputs))
        return h
