from turtle import forward
from unittest import skip
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

from codescholar.utils.train_utils import get_device


class SubgraphEmbedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, args):
        super(SubgraphEmbedder, self).__init__()
        self.encoder = BasicGNN(input_dim, hidden_dim, hidden_dim, args)
        self.margin = args.margin
        self.use_intersection = False

        self.classifier = nn.Sequential(
            nn.Linear(1, 2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, emb_targets, emb_queries):
        return emb_targets, emb_queries
    
    def criterion(self, pred, labels):
        """Loss function for subgraph ordering in embedding space.
            error = amount of violation (if b is a subgraph of a).
            For + examples, train to minimize error -> 0;
            For - examples, train to minimize error to be atleast self.margin
        """
        emb_targets, emb_queries = pred
        
        # sum(||max{0, z_q - z_u}||_2^2))
        error = torch.sum(
            torch.max(
                torch.zeros_like(
                    emb_targets,
                    device=get_device()),
                emb_queries - emb_targets
            )**2, dim=1)
        
        margin = self.margin

        # rewrite loss for -ve examples
        error[labels == 0] = torch.max(
            torch.tensor(0.0, device=get_device()),
            margin - error)[labels == 0]
        
        relation_loss = torch.sum(error)
        
        return relation_loss
    
    def predict(self, pred):
        """Inference API: predict if queries are subgraphs of targets

        Args:
            pred (List<emb_t, emb_q>): embeddings of pairs of graphs
        """
        emb_targets, emb_queries = pred
        is_subgraph = torch.sum(
            torch.max(
                torch.zeros_like(
                    emb_targets,
                    device=emb_targets.device),
                emb_queries - emb_targets
            )**2, dim=1)

        return is_subgraph
        

class BasicGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(BasicGNN, self).__init__()
        self.dropout = args.dropout
        self.n_layers = args.n_layers
        self.skip = args.skip
        self.agg_type = args.agg_type

        # add a preprocessor
        self.feat_preprocess = None

        # MODULE: INPUT
        self.pre_mp = nn.Sequential(nn.Linear(input_dim, hidden_dim))

        # MODULES(k): Graph Aggregation/Convolution
        agg_module = self.get_agg_layer(type=args.agg_type)
        self.aggregates = nn.ModuleList()

        # add learnable skip params
        if args.skip == 'learnable':
            self.learnable_skip = nn.Parameter(torch.ones(
                self.n_layers, self.n_layers))

        for layer in range(args.n_layers):
            if args.skip == 'all' or args.skip == "learnable":
                # a layer can get input from any of it's preceding layers
                # layer_input =  hidden_dim * # previous-layers
                hidden_input_dim = hidden_dim * (layer + 1)
            else:
                hidden_input_dim = hidden_dim

            self.aggregates.append(agg_module(hidden_input_dim, hidden_dim))
        
        # MODULE: OUTPUT
        post_input_dim = hidden_dim * (args.n_layers + 1)

        self.post_mp = nn.Sequential(
            nn.Linear(post_input_dim, hidden_dim),
            nn.Dropout(args.dropout),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim)
        )
    
    def get_agg_layer(self, type):
        # graph convolution
        if type == "GCN":
            return pyg_nn.GCNConv

        # graph isomorphism + weighted edges
        elif type == "GIN":
            return lambda i, h: WeightedGINConv(
                nn.Sequential(
                    nn.Linear(i, h),
                    nn.ReLU(),
                    nn.Linear(h, h)
                ))
        
        else:
            print("unrecognized model type")

    def forward(self, data):
        
        # preprocess (if reqd)
        if self.feat_preprocess is not None:
            if not hasattr(data, "preprocessed"):
                data = self.feat_preprocess(data)
                data.preprocessed = True

        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        
        # pre mlp
        x = self.pre_mp(x)
        all_emb = x.unsqueeze(1)
        emb = x

        # aggregate-combine loop (k iterations)
        for i in range(len(self.aggregates)):
            # print(f"Running layer {i}")
            
            # aggregate
            if self.skip == "learnable":
                skip_vals = self.learnable_skip[i, : i + 1]
                skip_vals = skip_vals.unsqueeze(0).unsqueeze(-1)
                curr_emb = all_emb * torch.sigmoid(skip_vals)  # select inputs
                curr_emb = curr_emb.view(x.size(0), -1)
                x = self.aggregates[i](curr_emb, edge_index)
            elif self.skip == "all":
                x = self.aggregates[i](emb, edge_index)
            else:
                x = self.aggregates[i](x, edge_index)
            
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # combine
            emb = torch.cat((emb, x), 1)
        
        # pooling
        # x = pyg_nn.global_mean_pool(x, batch)
        emb = pyg_nn.global_add_pool(emb, batch)

        # post MLP
        emb = self.post_mp(emb)

        return emb
            
    def loss(self, pred, label):
        return F.nll_loss(pred, label)


# pytorch geom implementation of GIN + weighted edges
# adapted from NeuroMatch [#TODO Add reference]
class WeightedGINConv(pyg_nn.MessagePassing):
    def __init__(self, nn, eps=0, train_eps=False, **kwargs):
        super(WeightedGINConv, self).__init__(aggr='add', **kwargs)
        self.nn = nn
        self.initial_eps = eps

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_index, edge_weight = pyg_utils.remove_self_loops(
            edge_index,
            edge_weight)
        out = self.nn((1 + self.eps) * x + self.propagate(
            edge_index, x=x,
            edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_weight):
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
