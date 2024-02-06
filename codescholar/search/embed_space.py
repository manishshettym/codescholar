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
from transformers import RobertaTokenizer, RobertaModel

from codescholar.sast.simplified_ast import get_simplified_ast
from codescholar.representation import models, config
from codescholar.utils.graph_utils import program_graph_to_nx
from codescholar.utils.train_utils import build_model, get_device, featurize_graph
from codescholar.search import search_config
from codescholar.constants import DATA_DIR


# ########## MULTI PROC ##########


def start_workers_process(in_queue, out_queue, args, gpu):
    torch.cuda.set_device(gpu)
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(target=generate_neighborhoods, args=(args, in_queue, out_queue, gpu))
        worker.start()
        workers.append(worker)

    return workers


def start_workers_embed(in_queue, out_queue, args, gpu):
    torch.cuda.set_device(gpu)
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(target=generate_embeddings, args=(args, in_queue, out_queue, gpu))
        worker.start()
        workers.append(worker)

    return workers


# ########## UTILITIES ##########


# returns a featurized (sampled) radial neighborhood for (sampled) nodes in the graph
def get_neighborhoods(args, graph, feat_tokenizer, feat_model, device_id=None):
    neighs = []

    # choose top args.num_neighborhoods nodes with high degree to sample neighborhoods from
    sampled_nodes = sorted(graph.nodes, key=lambda x: graph.degree[x], reverse=True)[: args.num_neighborhoods]

    # find each node's neighbors via SSSP
    for j, node in enumerate(sampled_nodes):
        shortest_paths = sorted(nx.single_source_shortest_path_length(graph, node, cutoff=args.radius).items(), key=lambda x: x[1])
        neighbors = list(map(lambda x: x[0], shortest_paths))

        if args.neighborhood_size != 0:
            # NOTE: random sampling of radius-hop neighbors,
            # results in nodes w/o any edges between them!!
            # Instead, sort neighbors by hops and chose top-K closest neighbors
            neighbors = neighbors[: args.neighborhood_size]

        if len(neighbors) > 1:
            # NOTE: G.subgraph([nodes]) returns the subG induced on [nodes]
            # i.e., the subG containing the nodes in [nodes] and
            # edges between these nodes => in this case, a (sampled) radial n'hood

            neigh = graph.subgraph(neighbors)
            neigh = featurize_graph(neigh, feat_tokenizer, feat_model, anchor=node, device_id=device_id)
            neighs.append(neigh)

    return neighs


# load the graph for a source program
def process_program(path, format="source"):
    if format == "source":
        with open(path, "r") as fp:
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

        try:
            graph = program_graph_to_nx(prog_graph, directed=True)
        except:
            return None

    elif format == "graphs":
        graph = torch.load(path)

    return graph


# ########## PIPELINE FUNCTIONS ##########


def generate_embeddings(args, in_queue, out_queue, device_id=None):
    # print("Moving model to device:", get_device(device_id))
    model = build_model(models.SubgraphEmbedder, args, device_id=device_id)
    model.eval()

    done = False
    while not done:
        msg, idx = in_queue.get()

        if msg == "done":
            done = True
            break

        # read only graphs of processed programs
        try:
            neighs = torch.load(osp.join(args.processed_dir, f"data_{idx}.pt"))
        except:
            out_queue.put(("complete"))
            continue

        with torch.no_grad():
            emb = model.encoder(Batch.from_data_list(neighs).to(get_device(device_id)))
            torch.save(emb, osp.join(args.emb_dir, f"emb_{idx}.pt"))

        out_queue.put(("complete"))


def generate_neighborhoods(args, in_queue, out_queue, device_id=None):
    codebert_name = "microsoft/codebert-base"
    feat_tokenizer = RobertaTokenizer.from_pretrained(codebert_name)
    feat_model = RobertaModel.from_pretrained(codebert_name).to(get_device(device_id))
    feat_model.eval()

    done = False
    while not done:
        msg, idx = in_queue.get()

        if msg == "done":
            done = True
            break

        if args.format == "source":
            raw_path = osp.join(args.source_dir, f"example_{idx}.py")
        else:
            raw_path = osp.join(args.graphs_dir, f"data_{idx}.pt")

        graph = process_program(raw_path, args.format)

        if graph is None:
            out_queue.put(("complete"))
            continue

        # cache/save graph for search
        if args.format == "source":
            torch.save(graph, osp.join(args.graphs_dir, f"data_{idx}.pt"))

        neighs = get_neighborhoods(args, graph, feat_tokenizer, feat_model, device_id=device_id)
        torch.save(neighs, osp.join(args.processed_dir, f"data_{idx}.pt"))

        del graph
        del neighs

        out_queue.put(("complete"))


# ########## MAIN ##########


def embed_main(args):
    if not osp.exists(osp.dirname(args.graphs_dir)):
        os.makedirs(osp.dirname(args.graphs_dir))

    if not osp.exists(osp.dirname(args.processed_dir)):
        os.makedirs(osp.dirname(args.processed_dir))

    if not osp.exists(osp.dirname(args.emb_dir)):
        os.makedirs(osp.dirname(args.emb_dir))

    if args.format == "source":
        raw_paths = sorted(glob.glob(osp.join(args.source_dir, "*.py")))
    else:
        raw_paths = sorted(glob.glob(osp.join(args.graphs_dir, "*.pt")))

    num_gpus = torch.cuda.device_count()
    workers_list = []
    in_queues, out_queues = [], []

    # ######### PHASE1: PROCESS GRAPHS #########

    for gpu in range(num_gpus):
        in_queue, out_queue = mp.Queue(), mp.Queue()
        workers = start_workers_process(in_queue, out_queue, args, gpu)
        workers_list.append(workers)
        in_queues.append(in_queue)
        out_queues.append(out_queue)

    for i in range(len(raw_paths)):
        gpu = i % num_gpus
        in_queues[gpu].put(("idx", i))

    results_collected = 0
    pbar = tqdm(total=len(raw_paths), desc="Process Graphs")
    while results_collected < len(raw_paths):
        for gpu in range(num_gpus):
            while not out_queues[gpu].empty():
                msg = out_queues[gpu].get()
                results_collected += 1
                pbar.update(1)
    pbar.close()

    for in_queue in in_queues:
        for _ in range(args.n_workers):
            in_queue.put(("done", None))

    for workers in workers_list:
        for worker in workers:
            worker.join()

    # ######### PHASE2: EMBED GRAPHS #########

    workers_list = []
    in_queues, out_queues = [], []

    for gpu in range(num_gpus):
        in_queue, out_queue = mp.Queue(), mp.Queue()
        workers = start_workers_embed(in_queue, out_queue, args, gpu)
        workers_list.append(workers)
        in_queues.append(in_queue)
        out_queues.append(out_queue)

    for i in range(len(raw_paths)):
        gpu = i % num_gpus
        in_queues[gpu].put(("idx", i))

    results_collected = 0
    pbar = tqdm(total=len(raw_paths), desc="Embed Graphs")
    while results_collected < len(raw_paths):
        for gpu in range(num_gpus):
            while not out_queues[gpu].empty():
                msg = out_queues[gpu].get()
                results_collected += 1
                pbar.update(1)
    pbar.close()

    for in_queue in in_queues:
        for _ in range(args.n_workers):
            in_queue.put(("done", None))

    for workers in workers_list:
        for worker in workers:
            worker.join()


def main():
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    # data config
    args.format = "source"  # {graphs, source}
    args.source_dir = {DATA_DIR}/{args.dataset}/source/"
    args.graphs_dir = {DATA_DIR}/{args.dataset}/graphs/"
    args.processed_dir = {DATA_DIR}/{args.dataset}/processed/"
    args.emb_dir = {DATA_DIR}/{args.dataset}/emb/"

    # model config
    args.test = True
    args.model_path = f"../representation/ckpt/model.pt"

    embed_main(args)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
