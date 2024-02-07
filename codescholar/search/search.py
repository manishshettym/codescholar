import os
import redis
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
from math import log
from collections import defaultdict

import torch
import networkx as nx
import scipy.stats as stats
from multiprocessing import Pool

from codescholar.sast.visualizer import render_sast
from codescholar.sast.sast_utils import sast_to_prog
from codescholar.representation import config
from codescholar.search import search_config
from codescholar.search.grow import grow
from codescholar.search.elastic_search import grep_programs
from codescholar.search.init_search import init_search_q, init_search_m, init_search_mq
from codescholar.utils.search_utils import (
    ping_elasticsearch,
    ping_elasticindex,
    wl_hash,
    save_idiom,
    _print_mine_logs,
    _write_mine_logs,
    read_graph
)
from codescholar.utils.search_utils import load_embeddings_batched_redis
from codescholar.utils.graph_utils import nx_to_program_graph
from codescholar.utils.perf import perftimer
from codescholar.constants import DATA_DIR

redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

######### IDIOM STORE ############


def _save_idiom_generation(args, idiommine_gen) -> bool:
    """save the current generation of idioms to disk.
    and return if the search should continue.
    """
    idiom_clusters = idiommine_gen.items()
    idiom_clusters = list(sorted(idiom_clusters, key=lambda x: len(x[1]), reverse=True))
    cluster_id, total_nhoods, total_idioms = 1, 0, 0

    for _, idioms in idiom_clusters:
        idioms = list(sorted(idioms, key=lambda x: x[1], reverse=True))

        for idiom, nhoods, holes in idioms:
            size_id, nhood_count = len(idiom), int(nhoods)

            if args.mode == "mq":
                if nx.number_connected_components(nx.to_undirected(idiom)) != 1:
                    continue

                if nhood_count < args.min_nhoods:
                    continue

            file = "idiom_{}_{}_{}_{}".format(size_id, cluster_id, nhood_count, holes)

            path = f"{args.idiom_g_dir}{file}.png"
            sast = nx_to_program_graph(idiom)

            # NOTE @manishs: when growing graphs in all directions
            # the root can get misplaced. Find root = node with no incoming edges!
            root = [n for n in sast.all_nodes() if sast.incoming_neighbors(n) == []][0]
            sast.root_id = root.id

            if args.render:
                render_sast(sast, path, spans=True, relpos=True)

            path = f"{args.idiom_p_dir}{file}.py"
            prog = sast_to_prog(sast).replace("#", "_")

            if args.mode == "mq":
                prog = "\n".join([line for line in prog.split("\n") if line.strip() != "_"])

            save_idiom(path, prog)

            # update counts
            total_nhoods += nhood_count
            total_idioms += 1

        cluster_id += 1

    # metrics
    reusability = total_nhoods / total_idioms if total_idioms > 0 else 0
    lreusability = log(reusability + 1 if reusability <= 0 else reusability)

    diversity = len(idiom_clusters)
    ldiversity = log(diversity + 1 if diversity <= 0 else diversity)

    if args.stop_at_equilibrium and ldiversity >= lreusability:
        return False
    else:
        return True

######### MAIN ############

# @perftimer
def search(args, prog_indices, beam_sets):
    mine_summary = defaultdict(lambda: defaultdict(int))
    size = 1

    if not beam_sets:
        print("Oops, BEAM SETS ARE EMPTY!")
        return mine_summary

    num_gpus = torch.cuda.device_count()

    # create a pool of workers for each GPU
    pools = [Pool(args.n_workers) for _ in range(num_gpus)]

    continue_search = True
    while continue_search and len(beam_sets) != 0:
        results = []
        for i, beam_set in enumerate(beam_sets):
            gpu = i % num_gpus
            # use apply_async to submit the grow task to the pool
            result = pools[gpu].apply_async(grow, (args, prog_indices, beam_set, gpu))
            results.append(result)

        # idioms for generation i
        idiommine_gen = defaultdict(list)
        new_beam_sets = []

        # beam search over this generation
        pbar = tqdm(total=len(beam_sets), desc=f"[search {size}]")
        for result in results:
            new_beams = result.get()  # Wait for the result and get it
            pbar.update(1)

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

        pbar.close()

        # save generation
        beam_sets = new_beam_sets
        size += 1
        # _print_mine_logs(mine_summary)

        if size >= args.min_idiom_size and size <= args.max_idiom_size:
            continue_search = _save_idiom_generation(args, idiommine_gen)

    # Terminate all pools
    for pool in pools:
        pool.terminate()
        pool.join()

    return mine_summary


def main(args):
    if (args.mode == "q" or args.mode == "mq") and args.seed is None:
        parser.error("query modes require --seed to begin search.")

    if not ping_elasticsearch():
        raise ConnectionError("Elasticsearch not running on localhost:9200! Please start Elasticsearch and try again.")

    if not ping_elasticindex():
        raise ValueError("Elasticsearch index `python_files` not found! Please run `elastic_search.py` to create the index.")

    # sample and constrain the search space
    if args.mode == "mq":
        prog_indices = set()

        for i, seed in enumerate(args.seed.split(";")):
            if i == 0:
                prog_indices = set(grep_programs(args, seed))
            else:
                prog_indices = prog_indices & set(grep_programs(args, seed))

        prog_indices = list(prog_indices)[: args.prog_samples]
    else:
        prog_indices = grep_programs(args, args.seed)[: args.prog_samples]

    # STEP 1: initialize search space
    if args.mode == "q":
        beam_sets = init_search_q(args, prog_indices, seed=args.seed)
    elif args.mode == "mq":
        beam_sets = init_search_mq(args, prog_indices, seeds=args.seed.split(";"))
    elif args.mode == "m":
        prog_indices = grep_programs(args, args.seed)[: args.prog_samples]
        beam_sets = init_search_m(args, prog_indices)
    else:
        raise ValueError(f"Invalid search mode {args.mode}!")

    # load all embeddings of prog_indices to redis
    load_embeddings_batched_redis(args, prog_indices)

    # STEP 2: search for idioms; saves idioms gradually
    mine_summary = search(args, prog_indices, beam_sets)
    _write_mine_logs(mine_summary, f"{args.result_dir}/mine_summary.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    # data config
    args.prog_dir = f"{DATA_DIR}/{args.dataset}/source/"
    args.source_dir = f"{DATA_DIR}/{args.dataset}/graphs/"
    args.emb_dir = f"{DATA_DIR}/{args.dataset}/emb/"
    args.result_dir = f"./results/{args.seed}/" if (args.mode == "q" or args.mode == "mq") else "./results/"
    args.idiom_g_dir = f"{args.result_dir}/idioms/graphs/"
    args.idiom_p_dir = f"{args.result_dir}/idioms/progs/"

    # model config
    args.test = True
    args.model_path = f"../representation/ckpt/model.pt"
    args.batch_size = 512

    if args.render and not osp.exists(args.idiom_g_dir):
        os.makedirs(args.idiom_g_dir)

    if not osp.exists(args.idiom_p_dir):
        os.makedirs(args.idiom_p_dir)

    torch.multiprocessing.set_start_method("spawn")
    main(args)
