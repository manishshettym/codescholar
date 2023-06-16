import os.path as osp
import numpy as np
import glob
import random
from typing import List

import torch
import networkx as nx


def sample_programs(src_dir: str, k=10000, seed=24):
    np.random.seed(seed)
    files = [f for f in sorted(glob.glob(osp.join(src_dir, "*.pt")))]
    random_files = np.random.choice(files, min(len(files), k))
    random_index = [f.split("_")[-1][:-3] for f in random_files]

    return random_files, random_index


def read_embeddings(files):
    prog_embs, prog_sizes = [], []
    for file in files:
        embs = torch.load(file, map_location=torch.device("cpu"))
        prog_embs.append(embs)
        prog_sizes.append(len(embs))

    return prog_embs, prog_sizes


def sample_prog_embs(src_dir: str, k=10000, seed=24):
    random_files, _ = sample_programs(src_dir, k, seed)
    prog_embs, prog_sizes = read_embeddings(random_files)

    return prog_embs, random_files, prog_sizes


def graphs_from_embs(graph_dir, paths: List[str]) -> List:
    graphs = []
    for file in paths:
        graph_path = "data_" + file.split("_")[-1]
        graph_path = osp.join(graph_dir, graph_path)

        graphs.append(torch.load(graph_path, map_location=torch.device("cpu")))

    return graphs


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
