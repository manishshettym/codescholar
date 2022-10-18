import random

import numpy as np
import torch
import torch.optim as optim
import scipy.stats as stats

from deepsnap.batch import Batch
from deepsnap.graph import Graph as DSGraph

from codescholar.representation.featurizer import FeatureAugment


device_cache = None


def get_device():
    global device_cache
    if device_cache is None:
        device_cache = torch.device("cuda") if torch.cuda.is_available() \
            else torch.device("cpu")
        # device_cache = torch.device("cpu")
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


def sample_neigh(graphs, size):
    """random bfs walk to find neighborhood graphs of a set size
    """
    ps = np.array([len(g) for g in graphs], dtype=np.float)
    ps /= np.sum(ps)
    dist = stats.rv_discrete(values=(np.arange(len(graphs)), ps))

    while True:
        idx = dist.rvs()
        # graph = random.choice(graphs)
        graph = graphs[idx]
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])

        while len(neigh) < size and frontier:
            new_node = random.choice(list(frontier))
            # new_node = max(sorted(frontier))
            assert new_node not in neigh
            neigh.append(new_node)
            visited.add(new_node)
            frontier += list(graph.neighbors(new_node))
            frontier = [x for x in frontier if x not in visited]

        if len(neigh) == size:
            return graph, neigh


def batch_nx_graphs(graphs, anchors=None):
    augmenter = FeatureAugment()
    
    if anchors is not None:
        for anchor, g in zip(anchors, graphs):
            for v in g.nodes:
                g.nodes[v]["node_feature"] = torch.tensor([float(v == anchor)])

    batch = Batch.from_data_list([DSGraph(g) for g in graphs])
    batch = augmenter.augment(batch)
    batch = batch.to(get_device())

    return batch
