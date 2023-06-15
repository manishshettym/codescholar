import torch
import torch.optim as optim

import re
import networkx as nx
from deepsnap.graph import Graph as DSGraph

from codescholar.utils.graph_utils import GraphEdgeLabel, GraphNodeLabel


def get_device(device_id=None):
    if device_id is None:
        if torch.cuda.is_available():
            # print("GPU is available!!!")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_id}")

    return device


def build_model(model_type, args, device_id=None):
    # build model
    model = model_type(1, args.hidden_dim, args, device_id=device_id)
    model.to(get_device(device_id))

    if args.test and args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location=get_device(device_id)))

    return model


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)

    if args.opt == "adam":
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "sgd":
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == "rmsprop":
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == "adagrad":
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)

    if args.opt_scheduler == "none":
        return None, optimizer
    elif args.opt_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == "cos":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)

    return scheduler, optimizer


def featurize_graph(g, feat_tokenizer, feat_model, anchor=None, device_id=None):
    assert len(g.nodes) > 0
    assert len(g.edges) > 0

    if anchor is not None:
        pagerank = nx.pagerank(g)
        clustering_coeff = nx.clustering(g)

        for v in g.nodes:
            g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])
            node_type_name = g.nodes[v]["ast_type"]
            try:
                node_span = g.nodes[v]["span"]
            except KeyError:
                raise KeyError(f"key error in node {v}")

            if isinstance(node_type_name, str):
                try:
                    node_type_val = GraphNodeLabel[node_type_name].value
                except KeyError:
                    node_type_val = GraphNodeLabel["Other"].value

                g.nodes[v]["ast_type"] = torch.tensor([node_type_val])

            if isinstance(node_span, str):
                # remove format #TODO @manishs: is this is better/worse
                node_span = re.sub("\s+", " ", node_span)

                tokens_ids = feat_tokenizer.encode(node_span, truncation=True)
                tokens_tensor = torch.tensor(tokens_ids, device=get_device(device_id))

                with torch.no_grad():
                    context_embeddings = feat_model(tokens_tensor[None, :])[0]

                # torch.cuda.empty_cache()

                g.nodes[v]["node_span"] = torch.mean(context_embeddings, dim=1)

            g.nodes[v]["node_degree"] = torch.tensor([g.degree(v)])
            g.nodes[v]["node_pagerank"] = torch.tensor([pagerank[v]])
            g.nodes[v]["node_cc"] = torch.tensor([clustering_coeff[v]])

    for e in g.edges:
        edge_type_name = g.edges[e]["flow_type"]

        if isinstance(edge_type_name, str):
            edge_type_val = GraphEdgeLabel[edge_type_name].value
            g.edges[e]["flow_type"] = torch.tensor([edge_type_val])

    # Note: no need to sort the nodes of the graph
    # to maintain an order. GNN is permutation invariant.

    return DSGraph(g)
