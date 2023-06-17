import os
import os.path as osp
import argparse
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from itertools import chain
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

import torch
import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from deepsnap.batch import Batch
import scipy.stats as stats
import torch.multiprocessing as mp
from transformers import RobertaTokenizer, RobertaModel

from codescholar.sast.simplified_ast import get_simplified_ast
from codescholar.sast.visualizer import render_sast
from codescholar.sast.sast_utils import sast_to_prog, remove_node
from codescholar.representation import models, config
from codescholar.search import search_config
from codescholar.utils.search_utils import sample_programs, wl_hash, save_idiom, _print_mine_logs, _write_mine_logs
from codescholar.utils.train_utils import build_model, get_device, featurize_graph
from codescholar.utils.graph_utils import nx_to_program_graph, program_graph_to_nx
from codescholar.utils.perf import perftimer

######### MACROS ############


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


######## DISK UTILS ##########


def _save_idiom_generation(args, idiommine_gen):
    hashed_idioms = idiommine_gen.items()
    hashed_idioms = list(sorted(hashed_idioms, key=lambda x: len(x[1]), reverse=True))
    count = 0

    for _, idioms in hashed_idioms:  # [:args.rank]:
        # # choose any one because they all map to the same hash
        # idiom = random.choice(idioms)

        idioms = list(sorted(idioms, key=lambda x: x[1], reverse=True))
        idiom, score, holes = idioms[0]

        freq = len(idioms)
        file = "idiom_{}_{}_{}_{}_{}".format(len(idiom), count, freq, int(score), holes)

        path = f"{args.idiom_g_dir}{file}.png"
        sast = nx_to_program_graph(idiom)

        # NOTE @manishs: when growing graphs in all directions
        # the root can get misplaced. Find the root node
        # by looking for the node with no incoming edges!
        root = [n for n in sast.all_nodes() if sast.incoming_neighbors(n) == []][0]
        sast.root_id = root.id
        render_sast(sast, path, spans=True, relpos=True)

        path = f"{args.idiom_p_dir}{file}.py"
        prog = sast_to_prog(sast).replace("#", "_")
        save_idiom(path, prog)
        count += 1


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


######### INIT ############


# init_search for --mode m (idiom mine)
def init_search_m(args, prog_indices):
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

        graph = read_graph(args, graph_idx)  # TODO: convert to undirected?
        start_node = random.choice(list(graph.nodes))
        neigh = [start_node]

        # TODO: convert to undirected search like --mode g
        # find frontier = {neighbors} - {itself} = {supergraphs}
        frontier = list(set(graph.neighbors(start_node)) - set(neigh))
        visited = set([start_node])

        beam_sets.append([(0, 0, neigh, frontier, visited, graph_idx)])

    return beam_sets


# init_search for --mode g (idiom seed-graph-search)
def init_search_g(args, prog_indices, seed):
    beam_sets = []
    count = 0

    # generate seed graph for query
    seed_sast = get_simplified_ast(seed)
    if seed_sast is None:
        raise ValueError("Seed program is invalid!")

    module_nid = list(seed_sast.get_ast_nodes_of_type("Module"))[0].id
    remove_node(seed_sast, module_nid)

    seed_graph = program_graph_to_nx(seed_sast, directed=True)

    for idx in tqdm(prog_indices, desc="[init_search]"):
        if count >= args.max_init_beams:
            continue

        graph = read_graph(args, idx)

        # find all matches of the seed graph in the program graph
        # uses exact subgraph isomorphism - not that expensive because query is small (2-3 nodes)
        node_match = lambda n1, n2: n1["span"] == n2["span"] and n1["ast_type"] == n2["ast_type"]
        DiGM = DiGraphMatcher(graph, seed_graph, node_match=node_match)
        seed_matches = list(DiGM.subgraph_isomorphisms_iter())

        # no matches
        if len(seed_matches) == 0:
            continue

        # randomly select one of the matches as the starting point
        neigh = list(random.choice(seed_matches).keys())

        # find frontier = {successors} U {predecessors} - {itself} = {supergraphs}
        frontier = set(_reduce(list(_frontier(graph, n, type="radial") for n in neigh))) - set(neigh)
        visited = set(neigh)

        beam_sets.append([(0, 0, neigh, frontier, visited, idx)])
        count += 1

    return beam_sets


######### GROW ############


def start_workers_grow(prog_indices, in_queue, out_queue, args, gpu):
    torch.cuda.set_device(gpu)
    workers = []
    for _ in tqdm(range(args.n_workers), desc="[workers]"):
        worker = mp.Process(target=grow, args=(args, prog_indices, in_queue, out_queue, gpu))
        worker.start()
        workers.append(worker)

    return workers


def score_candidate_freq(args, model, embs, cand_emb, device_id=None):
    """score candidate program embedding against target program embeddings
    in a batched manner.

    Algorithm:
    softmax based classifier on top of emb-diff
        score = #nhoods !containing cand.
        = count(cand_emb - embs > 0)
            --> classifier: 0/1
    """
    score = 0

    for emb_batch in embs:
        with torch.no_grad():
            is_subgraph_rel = model.predict((emb_batch.to(get_device(device_id)), cand_emb))
            is_subgraph = model.classifier(is_subgraph_rel.unsqueeze(1))
            score += torch.sum(torch.argmax(is_subgraph, axis=1)).item()

    return score


def grow(args, prog_indices, in_queue, out_queue, device_id=None):
    embs = read_embeddings_batched(args, prog_indices)

    codebert_name = "microsoft/codebert-base"
    feat_tokenizer = RobertaTokenizer.from_pretrained(codebert_name)
    feat_model = RobertaModel.from_pretrained(codebert_name).to(get_device(device_id))
    feat_model.eval()
    
    # print("Moving model to device:", get_device(device_id))
    model = build_model(models.SubgraphEmbedder, args, device_id=device_id)
    model.eval()

    done = False
    while not done:
        msg, beam_set = in_queue.get()

        if msg == "done":
            del embs
            done = True
            break

        new_beams = []

        # STEP 1: Explore all candidates in each beam of the beam_set
        for beam in beam_set:
            _, holes, neigh, frontier, visited, graph_idx = beam
            graph = read_graph(args, graph_idx)

            if len(neigh) >= args.max_idiom_size or not frontier:
                continue

            cand_neighs = []

            # EMBED CANDIDATES
            for i, cand_node in enumerate(frontier):
                cand_neigh = graph.subgraph(neigh + [cand_node])
                cand_neigh = featurize_graph(cand_neigh, feat_tokenizer, feat_model, anchor=neigh[0], device_id=device_id)
                cand_neighs.append(cand_neigh)

            cand_batch = Batch.from_data_list(cand_neighs).to(get_device(device_id))
            with torch.no_grad():
                cand_embs = model.encoder(cand_batch)

            # SCORE CANDIDATES (freq)
            for cand_node, cand_emb in zip(frontier, cand_embs):
                score = score_candidate_freq(args, model, embs, cand_emb, device_id=device_id)
                new_neigh = neigh + [cand_node]

                # new frontier = {prev frontier} U {outgoing and incoming neighbors of cand_node} - {visited}
                # note: one can use type='neigh' to add only outgoing neighbors
                new_frontier = list(((set(frontier) | _frontier(graph, cand_node, type="radial")) - visited) - set([cand_node]))

                new_visited = visited | set([cand_node])

                # first, add new holes introduced
                # then, remove hole filled in/by cand_node (incoming/outgoing edge resp)
                new_holes = holes + graph.nodes[cand_node]["span"].count("#") - 1

                # overcome bugs in graph construction
                # TODO: remove once sast is fixed
                if new_holes < 0 or new_holes > args.max_holes:
                    continue

                new_beams.append((score, new_holes, new_neigh, new_frontier, new_visited, graph_idx))

        # STEP 2: Sort new beams by freq_score/#holes
        new_beams = list(sorted(new_beams, key=lambda x: x[0] / x[1] if x[1] > 0 else x[0], reverse=True))

        # print("===== [debugger] new beams =====")
        # for beam in new_beams:
        #     print("freq: ", beam[0])
        #     print("holes: ", beam[1])
        #     print("nodes: ", [graph.nodes[n]['span'] for n in beam[2]])
        #     print("frontier: ", [graph.nodes[n]['span'] for n in beam[3]])
        #     print()
        # print("================================")

        # STEP 3: filter top-k beams
        new_beams = new_beams[: args.n_beams]
        out_queue.put(("complete", new_beams))


######### MAIN ############


@perftimer
def search(args, prog_indices, beam_sets):
    mine_summary = defaultdict(lambda: defaultdict(int))
    size = 1

    if not beam_sets:
        print("Oops, BEAM SETS ARE EMPTY!")
        return mine_summary

    num_gpus = torch.cuda.device_count()
    workers_list = []
    in_queues, out_queues = [], []
    
    for gpu in range(num_gpus):
        in_queue, out_queue = mp.Queue(), mp.Queue()
        workers = start_workers_grow(prog_indices, in_queue, out_queue, args, gpu)
        workers_list.append(workers)
        in_queues.append(in_queue)
        out_queues.append(out_queue)

    while len(beam_sets) != 0:
        for i, beam_set in enumerate(beam_sets):
            gpu = i % num_gpus
            in_queues[gpu].put(("beam_set", beam_set))

        # idioms for generation i
        idiommine_gen = defaultdict(list)
        new_beam_sets = []

        # beam search over this generation
        results_count = 0
        pbar = tqdm(total=len(beam_sets), desc=f"[search {size}]")
        while results_count < len(beam_sets):
            for gpu in range(num_gpus):
                while not out_queues[gpu].empty():
                    msg, new_beams = out_queues[gpu].get()
                    results_count += 1
                    
                    # candidates from only top-scoring beams in the beam set
                    for new_beam in new_beams[:1]:
                        score, holes, neigh, _, _, graph_idx = new_beam
                        graph = read_graph(args, graph_idx)

                        neigh_g = graph.subgraph(neigh).copy()
                        neigh_g.remove_edges_from(nx.selfloop_edges(neigh_g))

                        for v in neigh_g.nodes:
                            neigh_g.nodes[v]["anchor"] = 1 if v == neigh[0] else 0

                        neigh_g_hash = wl_hash(neigh_g)
                        idiommine_gen[neigh_g_hash].append((neigh_g, score, holes))
                        mine_summary[len(neigh_g)][neigh_g_hash] += 1

                    if len(new_beams) > 0:
                        new_beam_sets.append(new_beams)

                    pbar.update(1)
        pbar.close()

        # save generation
        beam_sets = new_beam_sets
        size += 1
        # _print_mine_logs(mine_summary)

        if size >= args.min_idiom_size and size <= args.max_idiom_size:
            _save_idiom_generation(args, idiommine_gen)

    for in_queue in in_queues:
        for _ in range(num_gpus):
            in_queue.put(("done", None))

    for workers in workers_list:
        for worker in workers:
            worker.join()

    return mine_summary


def main(args):
    if args.mode == "g" and args.seed is None:
        parser.error("graph mode requires --seed to begin search.")

    # init search space = sample K programs
    _, prog_indices = sample_programs(args.emb_dir, k=args.prog_samples, seed=4)

    # init search space
    if args.mode == "g":
        beam_sets = init_search_g(args, prog_indices, seed=args.seed)
    else:
        beam_sets = init_search_m(args, prog_indices)

    # search for idioms; saves idioms gradually
    mine_summary = search(args, prog_indices, beam_sets)
    _write_mine_logs(mine_summary, f"{args.result_dir}/mine_summary.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    # data config
    args.prog_dir = f"../data/{args.dataset}/source/"
    args.source_dir = f"../data/{args.dataset}/graphs/"
    args.emb_dir = f"../data/{args.dataset}/emb/"
    args.result_dir = f"./results/{args.seed}/" if args.mode == "g" else "./results/"
    args.idiom_g_dir = f"{args.result_dir}/idioms/graphs/"
    args.idiom_p_dir = f"{args.result_dir}/idioms/progs/"

    # model config
    args.test = True
    args.model_path = f"../representation/ckpt/model.pt"

    if not osp.exists(args.idiom_g_dir):
        os.makedirs(args.idiom_g_dir)

    if not osp.exists(args.idiom_p_dir):
        os.makedirs(args.idiom_p_dir)

    torch.multiprocessing.set_start_method("spawn")
    main(args)
