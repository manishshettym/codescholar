import json
import argparse
import os
import os.path as osp
from datetime import date

import torch
from codescholar.representation import config
from codescholar.search.search import main as search_main
from codescholar.search import search_config
    

def multi_api_eval(args):
    with open("multibench.json") as f:
        benchmarks = json.load(f)
    
    for type in benchmarks:
        for query in benchmarks[type]:
            apis = ";".join(query)
            print(f"EVALUATING [{type}] [{query}]")
            
            args.mode = "mq"
            args.min_nhoods = 1
            args.seed = apis
            args.result_dir = f"./results/{date.today()}/{type}/{args.seed}/"
            args.idiom_g_dir = f"{args.result_dir}/idioms/graphs/"
            args.idiom_p_dir = f"{args.result_dir}/idioms/progs/"
            
            if not osp.exists(args.idiom_g_dir):
                os.makedirs(args.idiom_g_dir)
            
            if not osp.exists(args.idiom_p_dir):
                os.makedirs(args.idiom_p_dir)
            
            search_main(args)
            print("=====================================\n\n")


def single_api_eval(args):
    with open("singlebench.json") as f:
        benchmarks = json.load(f)

    for lib in benchmarks:
        for api in benchmarks[lib]:
            print(f"EVALUATING [{lib}] [{api}]")
            print("=====================================")

            args.mode = "q"
            args.seed = api
            args.result_dir = f"./results/{date.today()}/{lib}_res/{args.seed}/"
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
    parser.add_argument("--benchtype", type=str, default="single", choices=["single", "multi"])
    args = parser.parse_args()

    # data config
    args.prog_dir = f"../data/{args.dataset}/source/"
    args.source_dir = f"../data/{args.dataset}/graphs/"
    args.emb_dir = f"../data/{args.dataset}/emb/"

    # model config
    args.test = True
    args.model_path = f"../representation/ckpt/model.pt"

    torch.multiprocessing.set_start_method("spawn")

    if args.benchtype == "single":
        single_api_eval(args)

    elif args.benchtype == "multi":
        multi_api_eval(args)
