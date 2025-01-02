import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import random
import numpy as np
import os
from itertools import chain
from torch import Tensor
from torch.nn import Parameter as Param

from torch_geometric.nn.inits import uniform, glorot_orthogonal
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
import torch.autograd as autograd
from torch.nn import Linear
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.utils import dropout_edge, to_dense_adj
from torch import linalg as LA
from torch_geometric.nn import knn_graph, Sequential
from torch_geometric.nn.conv import MessagePassing, GATConv, GatedGraphConv, SAGEConv, GraphConv
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LeakyReLU, Sigmoid, BatchNorm1d as BN, Conv1d, Dropout, Tanh, Softmax
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.transforms import ToUndirected
import random
to_undirected = ToUndirected()
torch.manual_seed(0)

def generate_colorable_graph(num_nodes, max_degree):
    """
    Generates a random graph with a specified number of nodes
    and ensures it is colorable with max_degree + 1 colors.
    
    Args:
        num_nodes (int): Number of nodes in the graph.
        max_degree (int): Maximum degree for any node.
        
    Returns:
        G (networkx.Graph): A random graph.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    for node in range(num_nodes):
        # Randomly connect to other nodes while ensuring max_degree
        possible_neighbors = [n for n in range(num_nodes) if n != node and not G.has_edge(node, n)]
        random.shuffle(possible_neighbors)
        num_edges = min(max_degree - G.degree[node], len(possible_neighbors))
        for neighbor in possible_neighbors[:num_edges]:
            if G.degree[neighbor] < max_degree:
                G.add_edge(node, neighbor)
    
    return G

def generate_colorable_graph_list(num_graphs, num_nodes, max_degree):
    Graph_list = []
    for i in range(num_graphs):
        G = generate_colorable_graph(num_nodes,max_degree)
        graph = from_networkx(G)
        graph.x = torch.ones(num_nodes,2)    
        graph = to_undirected(graph)
        Graph_list.append(graph)

    return Graph_list


class GatedGraphConv(MessagePassing):
    r"""The gated graph convolution operator from the `"Gated Graph Sequence
    Neural Networks" <https://arxiv.org/abs/1511.05493>`_ paper

    .. math::
        \mathbf{h}_i^{(0)} &= \mathbf{x}_i \, \Vert \, \mathbf{0}

        \mathbf{m}_i^{(l+1)} &= \sum_{j \in \mathcal{N}(i)} e_{j,i} \cdot
        \mathbf{\Theta} \cdot \mathbf{h}_j^{(l)}

        \mathbf{h}_i^{(l+1)} &= \textrm{GRU} (\mathbf{m}_i^{(l+1)},
        \mathbf{h}_i^{(l)})

    up to representation :math:`\mathbf{h}_i^{(L)}`.
    The number of input channels of :math:`\mathbf{x}_i` needs to be less or
    equal than :obj:`out_channels`.
    :math:`e_{j,i}` denotes the edge weight from source node :obj:`j` to target
    node :obj:`i` (default: :obj:`1`)

    Args:
        out_channels (int): Size of each output sample.
        num_layers (int): The sequence length :math:`L`.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"add"`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`

    """
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True, **kwargs):
        super().__init__(aggr=aggr, **kwargs)

        self.out_channels = out_channels
        self.num_layers = num_layers

        self.weight = Param(Tensor(num_layers, out_channels, out_channels))
        self.rnn = torch.nn.GRUCell(out_channels, out_channels, bias=bias)
        self.bn = BN(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        #super().reset_parameters()
        #uniform(self.out_channels, self.weight)
        glorot_orthogonal(uniform(self.out_channels, self.weight), scale=1.0)
        self.rnn.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:

        if x.size(-1) > self.out_channels:
            raise ValueError('The number of input channels is not allowed to '
                             'be larger than the number of output channels')

        if x.size(-1) < self.out_channels:
            zero = x.new_zeros(x.size(0), self.out_channels - x.size(-1))
            x = torch.cat([x, zero], dim=1)

        for i in range(self.num_layers):
            m = torch.matmul(x, self.weight[i])
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            m = self.propagate(edge_index, x=m, edge_weight=edge_weight,
                               size=None)
            #x = self.bn(x)
            x = self.rnn(m, x)
            
        return x

    def message(self, x_j: Tensor, edge_weight: OptTensor):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.out_channels}, '
                f'num_layers={self.num_layers})')


class PICAN(torch.nn.Module):
    """
    Define the graph neural network model, with options including the 
    conv_layer  = 'Gated Graph Neural Network - GGNN', 'GraphConv', and 'GraphSage'
    """
    def __init__(self, output_channel_dim, num_of_layers, K, conv_layer):
        super(PICAN, self).__init__()
        self.output_channel_dim = output_channel_dim
        self.num_of_layers = num_of_layers
        self.K = K
        self.conv_layer = conv_layer
        if self.conv_layer == 'GGNN':
            self.conv = GatedGraphConv(self.output_channel_dim,self.num_of_layers) 
        elif self.conv_layer == 'GraphConv':
            modules = []
            for o in range(num_of_layers):
                if o == 0:
                    modules.append((GraphConv(1,self.output_channel_dim),'x, edge_index -> x'))
                    modules.append(BN(self.output_channel_dim)) 
                    modules.append(Tanh())
                else:
                    modules.append((GraphConv(self.output_channel_dim,self.output_channel_dim),'x, edge_index -> x'))
                    modules.append(BN(self.output_channel_dim)) 
                    modules.append(Tanh())
            self.conv = Sequential('x, edge_index',modules)
            
        elif self.conv_layer == 'GraphSage':
            modules = []
            for o in range(num_of_layers):
                if o == 0:
                    modules.append((SAGEConv(1,self.output_channel_dim,aggr='add'),'x, edge_index -> x'))
                    modules.append(BN(self.output_channel_dim))
                    modules.append(Tanh())  
                else:
                    modules.append((SAGEConv(self.output_channel_dim,self.output_channel_dim,aggr='add'),'x, edge_index -> x'))
                    modules.append(BN(self.output_channel_dim)) 
                    modules.append(Tanh())       
            self.conv = Sequential('x, edge_index',modules)

        elif self.conv_layer == 'TransConv':
            print('Transformer conv')

        else:
            print('conv_type is important')
        self.readout = Seq(Lin(self.output_channel_dim,self.K),Softmax(dim=-1))
        #self.readout = Seq(BN(self.output_channel_dim),Lin(self.output_channel_dim,int(self.output_channel_dim/2)),Lin(int(self.output_channel_dim/2),self.K),BN(self.K),Softmax(dim=-1))


    def forward(self, data):
        x0, edge_attr, edge_index, batch = torch.unsqueeze(data.x[:,0],-1), data.edge_attr, data.edge_index, data.batch
        if self.conv_layer == 'GGNN':
            x1 = self.conv(x = x0, edge_index = edge_index, edge_weight=edge_attr)
        else:
            x1 = self.conv(x = x0, edge_index = edge_index)
        out = self.readout(x1)     
        return out

def get_adjacency_matrix(data, torch_device, torch_dtype):
    """
        Generate adjacency matrix from pytorch geometric data
        return_type torch.tensor
    """
    G = to_networkx(data)
    adj = nx.linalg.graphmatrix.adjacency_matrix(G).todense()
    adj_ = torch.tensor(adj).type(torch_dtype).to(torch_device)

    return adj_

def loss_func_mod(probs, adj_tensor):
    loss_ = torch.mul(adj_tensor.squeeze(-1), (probs @ torch.transpose(probs,-1,1))).sum()
    return loss_

def hard_decision_eval(one_hot_channel, edge_indexx):
    cost_ = 0
    coloring_ = one_hot_channel.view(edge_indexx.shape[0],-1)
    for i in range(edge_indexx.shape[0]):
        edge_index =  torch.argwhere(edge_indexx[i])
        coloring = coloring_[i]
        u = edge_index[:,0]
        v = edge_index[:,1]
        cost_ += torch.sum((1*(coloring[u] == coloring[v])*(u != v))/2)
    #print(cost_)
    return cost_

def loss_func_color_hard(coloring, nx_graph):
    cost_ = 0
    for (u, v) in nx_graph.edges:
        #print(u)
        cost_ += 1*(coloring[int(u)] == coloring[int(v)])*(u != v)

    return cost_

def train(model,train_loader,optimizer,torch_device,N,K):
    model.train()
    total_loss = 0
    count = 0
    for data in train_loader:
        data = data.to(torch_device)
        optimizer.zero_grad()
        out = model(data)
        adj_tensor = to_dense_adj(data.edge_index, batch=data.batch)      
        loss = loss_func_mod(out.view(-1,N,K), adj_tensor)
        total_loss += (loss.item())/data.num_graphs
        count = count+1
        loss.backward()
        optimizer.step()
    total = total_loss / count 
    return total

def test(model,validation_loader,optimizer,torch_device,N,K):
    model.eval()
    total_loss = 0
    hard = 0
    count = 0
    power_weight = []
    for data in validation_loader:
        data = data.to(torch_device)
        with torch.no_grad():
            out = model(data)
            adj_tensor = to_dense_adj(data.edge_index, batch=data.batch)      
            loss = loss_func_mod(out.view(-1,N,K), adj_tensor)
            total_loss += (loss.item())/data.num_graphs
            hard += hard_decision_eval(torch.argmax(out,-1),adj_tensor)
            count = count+1
    total = total_loss / count  
    #print(torch.argmax(out[0:N,0:K],dim=-1)) 
    return total, out[0:N,0:K], hard


def trainmodel(name,model,num_epochs,scheduler, train_loader, validation_loader, optimizer,torch_device,N,K):
    loss_ = []
    losst_ = []
    hard_ = []
    power_weight = []
    for epoch in range(1,num_epochs):
        losst = train(model,train_loader,optimizer,torch_device,N,K)
        loss1, out, hard= test(model,validation_loader,optimizer,torch_device,N,K)
        loss_.append(loss1)
        losst_.append(losst)
        hard_.append(hard.to('cpu'))
        if (loss1 == min(loss_)):
           torch.save(model, 'models/softloss'+str(name))
        if (hard == min(hard_)):
           torch.save(model, 'models/hardloss'+str(name))
        print('Epoch {:03d}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val hard error: {:.1f}'.format(
            epoch, losst, loss1, hard))
        scheduler.step()
    print(out)
    return loss_, losst_, hard_

