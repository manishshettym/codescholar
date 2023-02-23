import os
import glob
import random
from tqdm import tqdm
import os.path as osp

import ast
import networkx as nx
import torch
import argparse
from deepsnap.batch import Batch
import torch.multiprocessing as mp

from codescholar.sast.simplified_ast import get_simplified_ast
from codescholar.representation import models, config
from codescholar.utils.graph_utils import program_graph_to_nx
from codescholar.utils.train_utils import (
    build_model, get_device, featurize_graph)
from codescholar.search import search_config


# ########## MULTI PROC ##########

def start_workers_process(in_queue, out_queue, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(
            target=generate_neighborhoods,
            args=(args, in_queue, out_queue)
        )
        worker.start()
        workers.append(worker)
    
    return workers


def start_workers_embed(model, in_queue, out_queue, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(
            target=generate_embeddings,
            args=(args, model, in_queue, out_queue)
        )
        worker.start()
        workers.append(worker)
    
    return workers


# ########## UTILITIES ##########

# returns a featurized (sampled) radial neighborhood for all nodes in the graph
def get_neighborhoods(args, graph):
    neighs = []

    # find each node's neighbors via SSSP
    for j, node in enumerate(graph.nodes):
        shortest_paths = sorted(nx.single_source_shortest_path_length(
            graph, node, cutoff=args.radius).items(), key = lambda x: x[1])
        neighbors = list(map(lambda x: x[0], shortest_paths))

        if args.subgraph_sample_size != 0:
            # NOTE: random sampling of radius-hop neighbors, 
            # results in nodes w/o any edges between them!!
            # Instead, sort neighbors by hops and chose top-K closest neighbors
            neighbors = neighbors[: args.subgraph_sample_size]

        if len(neighbors) > 1:
            # NOTE: G.subgraph([nodes]) returns the subG induced on [nodes]
            # i.e., the subG containing the nodes in [nodes] and 
            # edges between these nodes => in this case, a (sampled) radial n'hood
            neigh = graph.subgraph(neighbors)
            neigh = featurize_graph(neigh, anchor=0)
            neighs.append(neigh)
    
    return neighs


# load the graph for a source program
def process_program(path, format="source"):
    
    if format == "source":
        with open(path, 'r') as fp:
            try:
                source = fp.read()
                program = ast.parse(source)
            except:
                # print(f"[ast.parse] {path} is a bad file!")
                return None

        try:
            prog_graph = get_simplified_ast(source, dfg=False, cfg=False)
        except:
            return None

        if prog_graph is None:
            return None

        graph = program_graph_to_nx(prog_graph, directed=True)
    
    elif format == "graphs":
        graph = torch.load(path)
    
    return graph


def generate_embeddings(args, model, in_queue, out_queue):
    done = False
    while not done:
        msg, idx = in_queue.get()

        if msg == "done":
            done = True
            break

        neighs = torch.load(osp.join(args.processed_dir, f'data_{idx}.pt'))

        with torch.no_grad():
            emb = model.encoder(Batch.from_data_list(neighs).to(get_device()))
            torch.save(emb, osp.join(args.emb_dir, f'emb_{idx}.pt'))
        
        out_queue.put(("complete"))


def generate_neighborhoods(args, in_queue, out_queue):
    done = False
    while not done:
        msg, idx = in_queue.get()

        if msg == "done":
            done = True
            break

        if args.format == "source":
            raw_path = osp.join(args.source_dir, f'example_{idx}.py')
        else:
            raw_path = osp.join(args.graphs_dir, f'data_{idx}.pt')
        
        graph = process_program(raw_path, args.format)
        
        if graph is None:
            out_queue.put(("complete"))
            continue
        
        # cache/save graph for search
        if args.format == "source":
            torch.save(graph, osp.join(args.graphs_dir, f'data_{idx}.pt'))
        
        neighs = get_neighborhoods(args, graph)
        torch.save(neighs, osp.join(args.processed_dir, f'data_{idx}.pt'))

        del graph
        del neighs

        out_queue.put(("complete"))

def embed_main(args):

    if not osp.exists(osp.dirname(args.graphs_dir)):
        os.makedirs(osp.dirname(args.graphs_dir))

    if not osp.exists(osp.dirname(args.processed_dir)):
        os.makedirs(osp.dirname(args.processed_dir))

    if not osp.exists(osp.dirname(args.emb_dir)):
        os.makedirs(osp.dirname(args.emb_dir))

    # ######### PHASE1: PROCESS GRAPHS #########

    if args.format == "source":
        raw_paths = sorted(glob.glob(osp.join(args.source_dir, '*.py')))
    else:
        raw_paths = sorted(glob.glob(osp.join(args.graphs_dir, '*.pt')))

    # for idx, p in enumerate(raw_paths):
    #     os.rename(p, osp.join(args.source_dir, f"example_{idx}.py"))
    
    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers_process(in_queue, out_queue, args)

    for i in range(1010579, len(raw_paths)):
        in_queue.put(("idx", i))
        
    for _ in tqdm(range(1010579, len(raw_paths))):
        msg = out_queue.get()
    
    for _ in range(args.n_workers):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()

    # ######### PHASE2: EMBED GRAPHS #########

    # model = build_model(models.SubgraphEmbedder, args)
    # model.share_memory()

    # print("Moving model to device:", get_device())
    # model = model.to(get_device())
    # model.eval()

    # in_queue, out_queue = mp.Queue(), mp.Queue()
    # workers = start_workers_embed(model, in_queue, out_queue, args)

    # for i in range(len(raw_paths)):
    #     in_queue.put(("idx", i))
        
    # for _ in tqdm(range(len(raw_paths))):
    #     msg = out_queue.get()
    
    # for _ in range(args.n_workers):
    #     in_queue.put(("done", None))

    # for worker in workers:
    #     worker.join()


def main():
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    args.format = "source"  # {graphs, source}
    args.source_dir = f"../data/{args.dataset}/source/"
    args.graphs_dir = f"../data/{args.dataset}/graphs/"
    args.processed_dir = f"./tmp/{args.dataset}/processed/"
    args.emb_dir = f"./tmp/{args.dataset}/emb/"

    embed_main(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
