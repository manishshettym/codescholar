import json
import argparse
import os
import os.path as osp

import torch

from codescholar.sast.visualizer import render_sast
from codescholar.representation import config
from codescholar.search.search import main as search_main
from codescholar.search import search_config

def main(args):
    # load the benchmarks
    # with open('benchmarks.json') as f:
    #     benchmarks = json.load(f)

    # for lib, apis in benchmarks.items():
    #     for api in apis:

    #         # set seed argument for search
    #         args.seed = '{}.{}'.format(lib, api)

    #         # call search_main
    #         search_main(args)
    
    search_main(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()
    
    # data config
    args.prog_dir = f"../data/{args.dataset}/source/"
    args.source_dir = f"../data/{args.dataset}/graphs/"
    args.emb_dir = f"../search/tmp/{args.dataset}/emb/"
    args.idiom_g_dir = f"./results/idioms/graphs/"
    args.idiom_p_dir = f"./results/idioms/progs/"

    # model config
    args.test = True
    args.model_path = f"../representation/ckpt/model.pt"

    if not osp.exists(args.idiom_g_dir):
        os.makedirs(args.idiom_g_dir)
    
    if not osp.exists(args.idiom_p_dir):
        os.makedirs(args.idiom_p_dir)
    
    torch.multiprocessing.set_start_method('spawn')
    main(args)