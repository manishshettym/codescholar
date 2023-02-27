import os
import os.path as osp
import argparse
from typing import List
import networkx as nx
import black
from collections import defaultdict

from codescholar.representation import models, config
from codescholar.search import search_config
from codescholar.utils.search_utils import sample_prog_embs, graphs_from_embs
from codescholar.utils.train_utils import build_model
from codescholar.search.agents import GreedySearch
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


def main():
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    args.source_dir = f"../data/{args.dataset}/graphs/"
    args.emb_dir = f"./tmp/{args.dataset}/emb/"
    args.idiom_g_dir = f"./results/idioms/graphs/"
    args.idiom_p_dir = f"./results/idioms/progs/"

    if not osp.exists(args.idiom_g_dir):
        os.makedirs(args.idiom_g_dir)
    
    if not osp.exists(args.idiom_p_dir):
        os.makedirs(args.idiom_p_dir)

    # init search space = sample K programs
    embs, emb_paths, _ = sample_prog_embs(args.emb_dir, k=args.prog_samples, seed=4)
    dataset: List[nx.Digraph] = graphs_from_embs(args.source_dir, emb_paths)

    # build subgraph embedding model
    model = build_model(models.SubgraphEmbedder, args)
    model.share_memory()

    # build greedy beam search agent
    # hyperparams: idiom size, n_trials
    # TODO: parallelize the search procedure
    agent = GreedySearch(
        min_idiom_size=args.min_idiom_size,
        max_idiom_size=args.max_idiom_size,
        model=model,
        dataset=dataset,
        embs=embs,
        n_beams=args.n_beams,
        rank=args.rank)

    out_graphs = agent.search(n_trials=args.n_trials)
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
    main()
