"""Plot an alignment matrix for matching a query subgraph in a target graph 
using the results of the subgraph matching model"""

import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
from deepsnap.batch import Batch

from codescholar.representation import config
from codescholar.search import search_config
from codescholar.sast.simplified_ast import get_simplified_ast
from codescholar.sast.visualizer import render_sast
from codescholar.sast.sast_utils import sast_to_prog, remove_node
from codescholar.representation import models, config
from codescholar.search import search_config
from codescholar.utils.train_utils import build_model, get_device, featurize_graph
from codescholar.utils.graph_utils import nx_to_program_graph, program_graph_to_nx

def get_anchor_neigh(args, graph, anchor):
    shortest_paths = sorted(nx.single_source_shortest_path_length(
        graph, anchor, cutoff=args.radius).items(), key = lambda x: x[1])
    neighbors = list(map(lambda x: x[0], shortest_paths))

    # if args.subgraph_sample_size != 0:
    #     neighbors = neighbors[: args.subgraph_sample_size]

    if len(neighbors) > 1:
        neigh = graph.subgraph(neighbors)
        neigh = featurize_graph(neigh, anchor=anchor)
        return neigh
    else:
        return featurize_graph(graph, anchor=anchor)


def mask_var_names(sast):
    for node in sast.get_ast_nodes_of_type('Name'):
        ptype = sast.parent(node)
        if not ptype or (ptype.ast_type != 'Call' and ptype.ast_type != 'Attribute'):
            node.span = "VAR"
    return sast


def get_alignmat(args, query, target):
    args.radius = 7
    model = build_model(models.SubgraphEmbedder, args)
    model.eval()
    model.share_memory()
    
    q = get_simplified_ast(query)
    module_nid = list(q.get_ast_nodes_of_type('Module'))[0].id
    remove_node(q, module_nid)
    # q = mask_var_names(q)
    
    t = get_simplified_ast(target)
    module_nid = list(t.get_ast_nodes_of_type('Module'))[0].id
    remove_node(t, module_nid)
    # t = mask_var_names(t)
    
    render_sast(q, './plots/q.png', spans=True, relpos=True)
    render_sast(t, './plots/t.png', spans=True, relpos=True)
    
    q = program_graph_to_nx(q, directed=True)
    t = program_graph_to_nx(t, directed=True)
        
    mat = np.zeros((len(q), len(t)))
    for i, u in enumerate(q.nodes):
        for j, v in enumerate(t.nodes):
            # print("query anchored at:", q.nodes[u]['span'])
            qneigh = get_anchor_neigh(args, q, u)
            
            # print("target anchored at:", t.nodes[v]['span'])
            tneigh = get_anchor_neigh(args, t, v)

            with torch.no_grad():
                qemb = model.encoder(Batch.from_data_list([qneigh]).to(get_device()))
                temb = model.encoder(Batch.from_data_list([tneigh]).to(get_device()))
                vio_score = model.predict((temb, qemb))
                mat[i][j] = torch.log(vio_score + 1e-7).item()    
    
    y_labels = [q.nodes[i]['span'] for i in q.nodes]
    x_labels = [t.nodes[i]['span'] for i in t.nodes]
    
    plt.figure(figsize=(15, 10))

    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.imshow(mat, interpolation="nearest")
    plt.colorbar()
    plt.savefig("./plots/alignment.png")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()
    
    with open('./data/query.py', 'r') as fp:
        query_prog = fp.read()
    
    with open('./data/target.py', 'r') as fp:
        target_prog = fp.read()
        
    args.test = True
    args.model_path = f"../../representation/ckpt/model.pt"

    get_alignmat(args, query_prog, target_prog)