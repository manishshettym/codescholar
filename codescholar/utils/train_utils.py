import torch
import torch.optim as optim

import networkx as nx
from transformers import RobertaTokenizer, RobertaModel
from deepsnap.graph import Graph as DSGraph

from codescholar.utils.graph_utils import GraphEdgeLabel, GraphNodeLabel


def get_device():
    global device_cache

    if device_cache is None:
        if torch.cuda.is_available():
            # print("GPU is available!!!")
            device_cache = torch.device("cuda")
        else:
            device_cache = torch.device("cpu")

    return device_cache


def build_model(model_type, args):
    # build model
    model = model_type(1, args.hidden_dim, args)
    model.to(get_device())

    if args.test and args.model_path:
        model.load_state_dict(
            torch.load(args.model_path, map_location=get_device())
        )

    return model


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)

    if args.opt == 'adam':
        optimizer = optim.Adam(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(
            filter_fn, lr=args.lr, momentum=0.95,
            weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(
            filter_fn, lr=args.lr, weight_decay=weight_decay)

    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step,
            gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart)

    return scheduler, optimizer


device_cache = None
codebert_name = "microsoft/codebert-base"
CodeBertTokenizer = RobertaTokenizer.from_pretrained(codebert_name)
CodeBertModel = RobertaModel.from_pretrained(codebert_name).to(get_device())
CodeBertModel.eval()


def featurize_graph(g, anchor=None):
    if anchor is not None:
        pagerank = nx.pagerank(g)
        clustering_coeff = nx.clustering(g)
    
        for v in g.nodes:
            g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])
            node_type_name = g.nodes[v]['ast_type']
            try:
                node_span = g.nodes[v]['span']
            except KeyError:
                raise KeyError(f"key error in node {v}")

            if isinstance(node_type_name, str):
                try:
                    node_type_val = GraphNodeLabel[node_type_name].value
                except KeyError:
                    node_type_val = GraphNodeLabel['Other'].value

                g.nodes[v]["ast_type"] = torch.tensor([node_type_val])
            
            if isinstance(node_span, str):
                tokens_ids = CodeBertTokenizer.encode(
                    node_span, truncation=True)
                tokens_tensor = torch.tensor(tokens_ids, device=get_device())
                
                with torch.no_grad():
                    context_embeddings = CodeBertModel(tokens_tensor[None, :])[0]
                
                # torch.cuda.empty_cache()

                g.nodes[v]["node_span"] = torch.mean(
                    context_embeddings,
                    dim=1)
                            
            g.nodes[v]["node_degree"] = torch.tensor([g.degree(v)])
            g.nodes[v]["node_pagerank"] = torch.tensor([pagerank[v]])
            g.nodes[v]["node_cc"] = torch.tensor([clustering_coeff[v]])
    
    for e in g.edges:
        edge_type_name = g.edges[e]['flow_type']
        
        if isinstance(edge_type_name, str):
            edge_type_val = GraphEdgeLabel[edge_type_name].value
            g.edges[e]["flow_type"] = torch.tensor([edge_type_val])
    
    return DSGraph(g)
