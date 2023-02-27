import os
import os.path as osp
import argparse
from typing import List
import black
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import networkx as nx
from deepsnap.batch import Batch
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch.multiprocessing as mp

from codescholar.representation import models, config
from codescholar.search import search_config
from codescholar.utils.search_utils import sample_programs, wl_hash
from codescholar.utils.train_utils import build_model, get_device, featurize_graph
from codescholar.utils.graph_utils import nx_to_program_graph
from codescholar.sast.visualizer import render_sast
from codescholar.sast.sast_utils import sast_to_prog

def save_idiom(path, idiom):
    try:
        idiom = black.format_str(idiom, mode=black.FileMode())
    except:
        pass
    
    with open(path, 'w') as fp:
        fp.write(idiom)

    
def _print_mine(idiommine):
    print("========== CODESCHOLAR MINE ==========")
    print(".")
    for size, hashed_idioms in idiommine.items():
        print(f"├── size {size}")
        fin_idx = len(hashed_idioms.keys()) - 1

        for idx, (hash_id, idioms) in enumerate(hashed_idioms.items()):
            if idx == fin_idx:
                print(f"    └── [{idx}] {len(idioms)} idiom(s)")
            else:
                print(f"    ├── [{idx}] {len(idioms)} idiom(s)")
    print("==========+================+==========")


######## DISK READING UTILS ##########
def read_graph(args, idx):
    graph_path = f'data_{idx}.pt'
    graph_path = osp.join(args.source_dir, graph_path)
    return torch.load(graph_path, map_location=torch.device('cpu'))


def read_embedding(args, idx):
    emb_path = f"emb_{idx}.pt"
    emb_path = osp.join(args.emb_dir, emb_path)
    return torch.load(emb_path, map_location=torch.device('cpu'))


def read_embeddings(args, prog_indices):
    embs = []
    for idx in prog_indices:
        embs.append(read_embedding(args, idx))
    
    return embs


######### IDIOM SEARCH ############

def init_search(args, prog_indices):

    ps = []
    for idx in tqdm(prog_indices, desc="[init_search]"):
        g = read_graph(args, idx)
        ps.append(len(g))
        del g

    ps = np.array(ps, dtype=float)
    ps /= np.sum(ps)
    graph_dist = stats.rv_discrete(values=(np.arange(len(ps)), ps))

    beam_sets = []
    for trial in range(args.n_trials):
        graph_idx = np.arange(len(ps))[graph_dist.rvs()]
        graph_idx = prog_indices[graph_idx]
        
        graph = read_graph(args, graph_idx)
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]
        
        # find frontier = {neighbors} - {itself} = {supergraphs}
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])

        beam_sets.append([(0, neigh, frontier, visited, graph_idx)])

    return beam_sets


def start_workers_grow(model, prog_indices, in_queue, out_queue, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(
            target=grow,
            args=(args, model, prog_indices, in_queue, out_queue)
        )
        worker.start()
        workers.append(worker)
    
    return workers


def grow(args, model, prog_indices, in_queue, out_queue):
    done = False
    embs = read_embeddings(args, prog_indices)

    while not done:
        msg, beam_set = in_queue.get()

        if msg == "done":
            del embs
            done = True
            break
        
        new_beams = []

        # STEP 1: Explore all candidate nodes in the beam_set
        for beam in beam_set:
            _, neigh, frontier, visited, graph_idx = beam
            graph = read_graph(args, graph_idx)

            if len(neigh) >= args.max_idiom_size or not frontier:
                continue

            cand_neighs = []

            # EMBED CANDIDATES
            for cand_node in frontier:
                cand_neigh = graph.subgraph(neigh + [cand_node])
                cand_neigh = featurize_graph(cand_neigh, neigh[0])
                cand_neighs.append(cand_neigh)
            
            cand_batch = Batch.from_data_list(cand_neighs).to(get_device())
            with torch.no_grad():
                cand_embs = model.encoder(cand_batch)
                
            # SCORE CANDIDATES
            for cand_node, cand_emb in zip(frontier, cand_embs):
                score, n_embs = 0, 0

                # for emb_batch in embs:
                for i in range(len(embs) // args.batch_size):
                    emb_batch = embs[i*args.batch_size : (i+1)*args.batch_size]
                    emb_batch = torch.cat(emb_batch, dim=0)
                    n_embs += len(emb_batch)

                    '''score = total_violation := #nhoods !containing cand.
                    1. get embed of target prog(s) [k, 64] where k=#nodes/points
                    2. get embed of cand [64]
                    3. is subgraph rel satisified: 
                            model.predict:= sum(max{0, prog_emb - cand}**2) [k]
                    4. is_subgraph: 
                            model.classifier:= logsoftmax(mlp) [k, 2]
                            logsoftmax \in [-inf (prob:0), 0 (prob:1)]
                    5. score = sum(argmax(is_subgraph)) 
                    '''
                    with torch.no_grad():
                        is_subgraph_rel = model.predict((
                                    emb_batch.to(get_device()),
                                    cand_emb))
                        is_subgraph = model.classifier(
                                is_subgraph_rel.unsqueeze(1))
                        score -= torch.sum(torch.argmax(
                                    is_subgraph, axis=1)).item()

                new_neigh = neigh + [cand_node]
                new_frontier = list(((
                    set(frontier) | set(graph.neighbors(cand_node)))
                    - visited) - set([cand_node]))
                new_visited = visited | set([cand_node])
                new_beams.append((
                    score, new_neigh, new_frontier,
                    new_visited, graph_idx))

        # STEP 2: Sort new beams by score (total_violation)
        new_beams = list(sorted(
            new_beams, key=lambda x: x[0]))[:args.n_beams]

        out_queue.put(("complete", new_beams))


def search(args, model, prog_indices):
    beam_sets = init_search(args, prog_indices)
    idiommine = defaultdict(lambda: defaultdict(list))

    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers_grow(model, prog_indices, in_queue, out_queue, args)

    while len(beam_sets) != 0:
        
        for beam_set in beam_sets:
            in_queue.put(("beam_set", beam_set))
        
        new_beam_sets = []
        for _ in tqdm(range(len(beam_sets))):
            msg, new_beams = out_queue.get()

            # Select candidates from the top scoring beam
            for new_beam in new_beams[:1]:
                score, neigh, frontier, visited, graph_idx = new_beam
                graph = read_graph(args, graph_idx)

                neigh_g = graph.subgraph(neigh).copy()
                neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))

                for v in neigh_g.nodes:
                    neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0

                idiommine[len(neigh_g)][wl_hash(neigh_g)].append(neigh_g)

            if len(new_beams) > 0:
                new_beam_sets.append(new_beams)
        
        beam_sets = new_beam_sets
        _print_mine(idiommine)

    for _ in range(args.n_workers):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()
    
    return finish_search(args, idiommine)


def finish_search(args, idiommine):        
    cand_patterns_uniq = []
    for idiom_size in range(
            args.min_idiom_size,
            args.max_idiom_size + 1):

        hashed_idioms = idiommine[idiom_size].items()
        hashed_idioms = list(sorted(
            hashed_idioms, key=lambda x: len(x[1]), reverse=True))

        for _, idioms in hashed_idioms[:args.rank]:
            # choose any one because they all map to the same hash
            cand_patterns_uniq.append(random.choice(idioms))
            # for idiom in idioms:
                # cand_patterns_uniq.append(idiom)
    return cand_patterns_uniq


def main():
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    args.source_dir = f"../data/{args.dataset}/graphs/"
    args.emb_dir = f"./tmp/{args.dataset}/emb/"
    args.searchspace_dir = f"./tmp/{args.dataset}/searchspace/"
    args.idiom_g_dir = f"./results/idioms/graphs/"
    args.idiom_p_dir = f"./results/idioms/progs/"

    if not osp.exists(args.idiom_g_dir):
        os.makedirs(args.idiom_g_dir)
    
    if not osp.exists(args.idiom_p_dir):
        os.makedirs(args.idiom_p_dir)

    # init search space = sample K programs
    _, prog_indices = sample_programs(args.emb_dir, k=args.prog_samples, seed=4)
    
    # init model
    model = build_model(models.SubgraphEmbedder, args)
    model.eval()
    model.share_memory()

    # search for idioms
    out_graphs = search(args, model, prog_indices)
    count_by_size = defaultdict(int)

    for idiom in out_graphs:
        pat_len, pat_count = len(idiom), count_by_size[len(idiom)]
        file = "idiom_{}_{}".format(pat_len, pat_count)
        
        path = f"{args.idiom_g_dir}{file}.png"
        sast = nx_to_program_graph(idiom)
        render_sast(sast, path, spans=True, relpos=True)

        path = f"{args.idiom_p_dir}{file}.py"
        prog = sast_to_prog(sast).replace('#', '_')
        save_idiom(path, prog)

        count_by_size[len(idiom)] += 1


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
