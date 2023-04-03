import json
import argparse
import os
import os.path as osp

import torch
from codescholar.representation import config
from codescholar.search.search import main as search_main
from codescholar.search import search_config


def main(args):
    with open('benchmarks.json') as f:
        benchmarks = json.load(f)

    for lib in benchmarks:
        for api in benchmarks[lib]:
            print(f"EVALUATING [{lib}] [{api}]")
            print("=====================================")

            args.mode = 'g'
            args.seed = api
            args.result_dir = f"./results/{lib}/{args.seed}/"
            args.idiom_g_dir = f"{args.result_dir}/idioms/graphs/"
            args.idiom_p_dir = f"{args.result_dir}/idioms/progs/"

            if not osp.exists(args.idiom_g_dir):
                os.makedirs(args.idiom_g_dir)

            if not osp.exists(args.idiom_p_dir):
                os.makedirs(args.idiom_p_dir)

            search_main(args)
            print("=====================================\n\n")


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

    # model config
    args.test = True
    args.model_path = f"../representation/ckpt/model.pt"

    torch.multiprocessing.set_start_method('spawn')
    main(args)
