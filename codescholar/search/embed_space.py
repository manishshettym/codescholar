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


# radial sampling
def get_neighborhoods(args, graph):
    neighs = []

    # find each node's neighborhood
    for j, node in enumerate(graph.nodes):
        neigh = list(nx.single_source_shortest_path_length(
            graph, node, cutoff=args.radius).keys())

        if args.subgraph_sample_size != 0:
            neigh = random.sample(
                neigh, min(len(neigh), args.subgraph_sample_size))
        
        if len(neigh) > 1:
            neigh = graph.subgraph(neigh)
            # neigh.add_edge(0, 0)
            neigh = featurize_graph(neigh, anchor=0)
            neighs.append(neigh)
    
    return neighs


def process_program(path, format="source"):
    
    if format == "source":
        with open(path, 'r') as fp:
            source = fp.read()
        try:
            program = ast.parse(source)
        except:
            return None

        prog_graph = get_simplified_ast(source, dfg=False, cfg=False)
        graph = program_graph_to_nx(prog_graph, directed=True)
    
    elif format == "nx":
        graph = torch.load(path)
    
    return graph


def start_workers(model, raw_paths, in_queue, out_queue, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(
            target=generate_embeddings,
            args=(args, model, raw_paths, in_queue, out_queue)
        )
        worker.start()
        workers.append(worker)
    
    return workers


def generate_embeddings(args, model, raw_paths, in_queue, out_queue):
    done = False
    while not done:
        msg, idx = in_queue.get()

        if msg == "done":
            done = True
            break

        path = raw_paths[idx]
        graph = process_program(path, args.format)

        if graph is None:
            continue

        if args.use_full_graphs:
            datapoint = featurize_graph(graph, 0)
            neighs = [datapoint]
        else:
            neighs = get_neighborhoods(args, graph)
        
        # NOTE: expensive -- so better to do many at a time as a batch
        try:
            with torch.no_grad():
                emb = model.encoder(Batch.from_data_list(neighs).to(get_device()))
                torch.save(emb, osp.join(args.emb_dir, f'emb_{idx}.pt'))
        except RuntimeError:
            print(len(neighs), "was too much for me")
        
        out_queue.put(("complete"))


def embed_main(args):
    if not osp.exists(osp.dirname(args.emb_dir)):
        os.makedirs(osp.dirname(args.emb_dir))
    
    if args.format == "source":
        raw_paths = sorted(glob.glob(osp.join(args.source_dir, '*.py')))
    else:
        raw_paths = sorted(glob.glob(osp.join(args.source_dir, '*.pt')))

    model = build_model(models.SubgraphEmbedder, args)
    model.share_memory()

    print("Moving model to device:", get_device())
    model = model.to(get_device())
    model.eval()

    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers(model, raw_paths, in_queue, out_queue, args)

    for i in range(len(raw_paths)):
        in_queue.put(("idx", i))
        
    for _ in tqdm(range(len(raw_paths))):
        msg = out_queue.get()
    
    for _ in range(args.n_workers):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()


def main():
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    args.format = "nx"  # nx or source
    args.source_dir = f"../representation/tmp/{args.dataset}/train/graphs/"
    args.emb_dir = f"./tmp/{args.dataset}/emb/"

    embed_main(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
