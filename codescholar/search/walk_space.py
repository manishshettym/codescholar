import os
import os.path as osp
import argparse
from typing import List
import networkx as nx
from collections import defaultdict

import torch
import matplotlib.pyplot as plt

from codescholar.representation import models, config
from codescholar.search import search_config
from codescholar.utils.search_utils import sample_prog_embs, graphs_from_embs
from codescholar.utils.train_utils import build_model
from codescholar.search.agents import GreedySearch
from codescholar.utils.graph_utils import nx_to_program_graph
from codescholar.sast.visualizer import render_sast


def main():
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    args.source_dir = f"../representation/tmp/{args.dataset}/train/graphs/"
    args.emb_dir = f"./tmp/{args.dataset}/emb/"

    args.plots_dir = "./plots/"
    args.idiom_dir = f"./results/idioms/"

    if not osp.exists(args.idiom_dir):
        os.makedirs(args.idiom_dir)
    
    if not osp.exists(args.plots_dir):
        os.makedirs(args.plots_dir)

    embs, emb_paths, _ = sample_prog_embs(
        args.emb_dir, k=500, seed=4)

    dataset: List[nx.Digraph] = graphs_from_embs(args.source_dir, emb_paths)

    model = build_model(models.SubgraphEmbedder, args)
    model.share_memory()

    agent = GreedySearch(
        min_pattern_size=args.min_pattern_size,
        max_pattern_size=args.max_pattern_size,
        model=model,
        dataset=dataset,
        embs=embs,
        n_beams=1,
        analyze=True,
        out_batch_size=20)

    out_graphs = agent.search(n_trials=10)
    count_by_size = defaultdict(int)

    for idiom in out_graphs:
        if args.node_anchored:
            colors = ["red"] + ["blue"] * (len(idiom) - 1)
            nx.draw(idiom, node_color=colors, with_labels=True)
        else:
            nx.draw(idiom)

        pat_len = len(idiom)
        pat_count = count_by_size[len(idiom)]
        file = "idiom_{}_{}".format(pat_len, pat_count)
        print(f"Saving {file}")
        
        path = f"{args.plots_dir}{file}.png"
        plt.savefig(path)
        plt.close()

        path = f"{args.idiom_dir}{file}.png"
        sast = nx_to_program_graph(idiom)
        render_sast(sast, path, spans=True)

        count_by_size[len(idiom)] += 1


if __name__ == "__main__":
    main()
