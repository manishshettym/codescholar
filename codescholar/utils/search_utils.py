import os.path as osp
import numpy as np
import glob
import random
from typing import List
from itertools import chain

import torch
import networkx as nx
from elasticsearch import Elasticsearch


########## SEARCH MACROS ##########


def _reduce(lists):
    """merge a nested list of lists into a single list"""
    return chain.from_iterable(lists)


def _frontier(graph, node, type="neigh"):
    """return the frontier of a node.
    The frontier of a node is the set of nodes that are one hop away from the node.

    Args:
        graph: the graph to find the frontier in
        node: the node to find the frontier of
        type: the type of frontier to find
            'neigh': the neighbors of the node (default) = out in a directed graph
            'radial': the union of the outgoing and incoming frontiers
    """

    if type == "neigh":
        return set(graph.neighbors(node))
    elif type == "radial":
        return set(graph.successors(node)) | set(graph.predecessors(node))


########## ELASTIC SEARCH UTILS ##########


def ping_elasticsearch():
    """check if elasticsearch is running"""
    es = Elasticsearch("http://localhost:9200/")
    try:
        info = es.info()
    except:
        return False

    return True


def ping_elasticindex(index_name: str = "python_files"):
    """check if elasticsearch index exists"""
    es = Elasticsearch("http://localhost:9200/")
    try:
        info = es.indices.get(index=index_name)
    except:
        return False

    return True


########## SEARCH DISK UTILS ##########


def sample_programs(src_dir: str, k=10000, seed=24):
    np.random.seed(seed)
    files = [f for f in sorted(glob.glob(osp.join(src_dir, "*.pt")))]
    random_files = np.random.choice(files, min(len(files), k))
    random_index = [f.split("_")[-1][:-3] for f in random_files]

    return random_files, random_index


def graphs_from_embs(graph_dir, paths: List[str]) -> List:
    graphs = []
    for file in paths:
        graph_path = "data_" + file.split("_")[-1]
        graph_path = osp.join(graph_dir, graph_path)

        graphs.append(torch.load(graph_path, map_location=torch.device("cpu")))

    return graphs


# @cached(cache=LRUCache(maxsize=1000), key=lambda args, idx: hashkey(idx))
def read_graph(args, idx):
    graph_path = f"data_{idx}.pt"
    graph_path = osp.join(args.source_dir, graph_path)
    return torch.load(graph_path, map_location=torch.device("cpu"))


def read_prog(args, idx):
    prog_path = f"example_{idx}.py"
    prog_path = osp.join(args.prog_dir, prog_path)
    with open(prog_path, "r") as f:
        return f.read()


def read_embedding(args, idx):
    emb_path = f"emb_{idx}.pt"
    emb_path = osp.join(args.emb_dir, emb_path)
    return torch.load(emb_path, map_location=torch.device("cpu"))


def read_embeddings(args, prog_indices):
    embs = []
    for idx in prog_indices:
        embs.append(read_embedding(args, idx))

    return embs


def read_embeddings_batched(args, prog_indices):
    embs, batch_embs = [], []
    count = 0

    for i, idx in enumerate(prog_indices):
        batch_embs.append(read_embedding(args, idx))

        if i > 0 and i % args.batch_size == 0:
            embs.append(torch.cat(batch_embs, dim=0))
            count += len(batch_embs)
            batch_embs = []

    # add remaining embs as a batch
    if len(batch_embs) > 0:
        embs.append(torch.cat(batch_embs, dim=0))
        count += len(batch_embs)

    assert count == len(prog_indices)

    return embs


########## GRAPH HASH UTILS ##########

cached_masks = None


def vec_hash(v):
    global cached_masks
    if cached_masks is None:
        random.seed(2019)
        cached_masks = [random.getrandbits(32) for i in range(len(v))]

    v = [hash(v[i]) ^ mask for i, mask in enumerate(cached_masks)]
    return v


def wl_hash(g, dim=64):
    """weisfeiler lehman graph hash"""
    g = nx.convert_node_labels_to_integers(g)
    vecs = np.zeros((len(g), dim), dtype=int)

    for v in g.nodes:
        if g.nodes[v]["anchor"] == 1:
            vecs[v] = 1
            break

    for i in range(len(g)):
        newvecs = np.zeros((len(g), dim), dtype=int)
        for n in g.nodes:
            newvecs[n] = vec_hash(np.sum(vecs[list(g.neighbors(n)) + [n]], axis=0))
        vecs = newvecs

    return tuple(np.sum(vecs, axis=0))


######## IDIOM MINE UTILS ##########


def save_idiom(path, idiom):
    try:
        idiom = black.format_str(idiom, mode=black.FileMode())
    except:
        pass

    with open(path, "w") as fp:
        fp.write(idiom)


def _print_mine_logs(mine_summary):
    print("========== CODESCHOLAR MINE ==========")
    print(".")
    for size, hashed_idioms in mine_summary.items():
        print(f"├── size {size}")
        fin_idx = len(hashed_idioms.keys()) - 1

        for idx, (hash_id, count) in enumerate(hashed_idioms.items()):
            if idx == fin_idx:
                print(f"    └── [{idx}] {count} idiom(s)")
            else:
                print(f"    ├── [{idx}] {count} idiom(s)")
    print("==========+================+==========")


def _write_mine_logs(mine_summary, filepath):
    with open(filepath, "w") as fp:
        fp.write("========== CODESCHOLAR MINE ==========" + "\n")
        fp.write("." + "\n")
        for size, hashed_idioms in mine_summary.items():
            fp.write(f"├── size {size}" + "\n")
            fin_idx = len(hashed_idioms.keys()) - 1

            for idx, (hash_id, count) in enumerate(hashed_idioms.items()):
                if idx == fin_idx:
                    fp.write(f"    └── [{idx}] {count} idiom(s)" + "\n")
                else:
                    fp.write(f"    ├── [{idx}] {count} idiom(s)" + "\n")
        fp.write("==========+================+==========" + "\n")
