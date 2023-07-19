"""This script runs CodeScholar search for some APIs that are relevant to the NL2Code tasks.
The API for each task is found using GPT-3.5 (see nl2code.py:gpt_find_api for details).
Here, we simply use the GPT-3.5 recommended API as the query to mine idioms from CodeScholar."""

import json
import argparse
import os
import os.path as osp
from datetime import date

import torch
import openai

from codescholar.representation import config
from codescholar.search.search import main as search_main
from codescholar.search import search_config
from codescholar.evaluation.rag.templates import GPT_FIND_API


def gpt_find_api(query):
    """Find the API for the given query using GPT-3.5"""
    prompt = GPT_FIND_API.format(query=query, libs="{pandas, numpy, os, sklearn, matplotlib, torch}")
    prompts = [
        {"role": "system", "content": "You are a helpful API assistant who can provide relevant API names given a programming task."},
        {"role": "user", "content": prompt},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=prompts,
        temperature=0,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        # stop=['"""', "```"],
    )

    return response["choices"][0]["message"]["content"]


def codescholar_mine_examples(queries):
    for lib in queries:
        for api in queries[lib]:
            print(f"EVALUATING [{lib}] [{api}]")
            print("=====================================")

            args.mode = "g"
            args.seed = api
            args.result_dir = f"./data/{date.today()}/{lib}_res/{args.seed}/"
            args.idiom_g_dir = f"{args.result_dir}/idioms/graphs/"
            args.idiom_p_dir = f"{args.result_dir}/idioms/progs/"

            if osp.exists(args.idiom_p_dir):
                print(f"Skipping as idioms already exists")
                continue

            if not osp.exists(args.idiom_g_dir):
                os.makedirs(args.idiom_g_dir)

            if not osp.exists(args.idiom_p_dir):
                os.makedirs(args.idiom_p_dir)

            search_main(args)
            print("=====================================\n\n")


def main(args):
    # Phase 1: get the API for each task
    if args.get_apis:
        dataset = []
        with open("./data/cs_rag.jsonl", "r") as f:
            for l in f:
                d = json.loads(l.strip())
                d["api"] = gpt_find_api(d["intent"])
                print(f"{d['api']}")
                dataset.append(d)

        # write the dataset back to file
        with open("./data/cs_rag.jsonl", "w") as f:
            for d in dataset:
                f.write(json.dumps(d) + "\n")

    queries = {}
    with open("./data/cs_rag.jsonl", "r") as f:
        for l in f:
            d = json.loads(l.strip())
            lib, api = d["library"], d["api"]
            queries[lib] = queries.get(lib, []) + [api]

    # keep only unique apis for each library
    queries = {lib: list(set(apis)) for lib, apis in queries.items()}

    # Phase 2: get the API for each task
    codescholar_mine_examples(queries)

    # Phase 3: build a queriable map of api -> idioms
    # TODO: this is not implemented yet


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    parser.add_argument("--get-apis", action="store_true", help="Get the API for each task using GPT-3.5")
    args = parser.parse_args()

    # search config
    args.dataset = "pnosmt"
    args.min_idiom_size = 2
    args.max_idiom_size = 20
    args.max_init_beam_size = 150

    # data config
    args.prog_dir = f"../../data/{args.dataset}/source/"
    args.source_dir = f"../../data/{args.dataset}/graphs/"
    args.emb_dir = f"../../data/{args.dataset}/emb/"

    # model config
    args.test = True
    args.model_path = f"../../representation/ckpt/model.pt"
    args.batch_size = 512
    args.prog_samples = 100000

    torch.multiprocessing.set_start_method("spawn")
    main(args)
