"""Compute EMD - Earth Mover's Distance - between two distributions:
1. The distribution of the embeddings of a set of python code snippets
2. The distribution of the embeddings of a set of idioms (result of CodeScholar)
"""
import os
import os.path as osp
import argparse
from datetime import date
import random
import json
import numpy as np
import torch
import ot

from codescholar.search.elastic_search import grep_programs
from codescholar.evaluation.emd.utils_codebert import embed_programs_codebert
from codescholar.evaluation.emd.utils_gpt import embed_programs_gpt
from codescholar.evaluation.emd.utils_emd import load_program, load_gpt_idioms, load_cs_idioms, trim_code


def compute_emd(code_embeddings, idiom_embeddings):
    n_code_samples = code_embeddings.shape[0]
    n_idiom_samples = idiom_embeddings.shape[0]

    # Compute pairwise distance matrix
    M = ot.dist(code_embeddings, idiom_embeddings)

    # assume uniform distribution for the two sets of samples
    a = np.ones(n_code_samples) / n_code_samples
    b = np.ones(n_idiom_samples) / n_idiom_samples

    # Compute EMD
    emd = ot.emd2(a, b, M)

    return emd


def main(args):
    random.seed(42)

    if osp.exists(args.emb_cache_file):
        embs = np.load(args.emb_cache_file)
        prog_embeddings = embs["prog_embeddings"]
        cs_idiom_embeddings = embs["cs_idiom_embeddings"]
        gpt_idiom_embeddings = embs["gpt_idiom_embeddings"]

        print(f"Programs: {len(prog_embeddings)}", flush=True)
        print(f"CS idioms: {len(cs_idiom_embeddings)}", flush=True)
        print(f"GPT idioms: {len(gpt_idiom_embeddings)}", flush=True)

    else:
        os.makedirs(osp.dirname(args.emb_cache_file), exist_ok=True)

        # Loa python code snippets with API (max 20k)
        prog_indices = grep_programs(args, api)
        prog_indices = random.sample(prog_indices, min(len(prog_indices), 20000))
        progs = [load_program(f"{args.prog_dir}/example_{i}.py") for i in prog_indices]
        progs = [trim_code(prog, api) for prog in progs]

        # Load all CodeScholar idioms for API
        cs_idioms = load_cs_idioms(args.cs_idioms_dir)

        # Load all GPT idioms for API
        gpt_idioms = load_gpt_idioms(args.gpt_idioms_dir)

        print(f"Programs: {len(progs)}", flush=True)
        print(f"CS idioms: {len(cs_idioms)}", flush=True)
        print(f"GPT idioms: {len(gpt_idioms)}", flush=True)

        if args.model == "codebert":
            prog_embeddings = embed_programs_codebert(args, progs)
            cs_idiom_embeddings = embed_programs_codebert(args, cs_idioms)
            gpt_idiom_embeddings = embed_programs_codebert(args, gpt_idioms)

        elif args.model == "gpt":
            prog_embeddings = embed_programs_gpt(args, progs)
            cs_idiom_embeddings = embed_programs_gpt(args, cs_idioms)
            gpt_idiom_embeddings = embed_programs_gpt(args, gpt_idioms)

        prog_embeddings = np.concatenate(prog_embeddings, axis=0)
        cs_idiom_embeddings = np.concatenate(cs_idiom_embeddings, axis=0)
        gpt_idiom_embeddings = np.concatenate(gpt_idiom_embeddings, axis=0)

        # cache the embeddings for reproducibility/reruns
        np.savez_compressed(
            args.emb_cache_file,
            prog_embeddings=prog_embeddings,
            cs_idiom_embeddings=cs_idiom_embeddings,
            gpt_idiom_embeddings=gpt_idiom_embeddings,
        )

    return (compute_emd(prog_embeddings, cs_idiom_embeddings), compute_emd(prog_embeddings, gpt_idiom_embeddings))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="codebert", choices=["codebert", "gpt"])
    args = parser.parse_args()

    args.dataset = "pnosmt"
    args.batch_size = 256
    args.n_workers = 4
    args.prog_dir = f"../../data/{args.dataset}/source/"
    torch.multiprocessing.set_start_method("spawn")

    with open("../benchmarks.json") as f:
        benchmarks = json.load(f)

    for lib in benchmarks:
        for api in benchmarks[lib]:
            args.cs_idioms_dir = f"../results/2023-07-04/{lib}_res/{api}/idioms/progs"
            args.gpt_idioms_dir = f"../gpt/results/2023-07-03/{lib}_res/{api}/"
            args.emb_cache_file = f"./cache/{args.model}/{lib}/{api}.npz"

            print(f"========== [{lib}: {api}] ==========", flush=True)
            cs_emd, gpt_emd = main(args)
            print(f"CS EMD: {cs_emd}", flush=True)
            print(f"GPT EMD: {gpt_emd}", flush=True)
            print("=====================================\n\n", flush=True)
