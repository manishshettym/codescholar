"""Compute EMD - Earth Mover's Distance - between two distributions:
1. The distribution of the embeddings of a set of python code snippets
2. The distribution of the embeddings of a set of idioms (result of CodeScholar)
"""
import os
import os.path as osp
import argparse
from tqdm import tqdm
from datetime import date
import json
import numpy as np
import torch

import ot
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from codescholar.search.elastic_search import grep_programs
from codescholar.utils.train_utils import get_device


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


def start_workers_genemb(in_queue, out_queue, args, gpu):
    torch.cuda.set_device(gpu)
    workers = []
    for _ in tqdm(range(args.n_workers), desc="[workers]"):
        worker = mp.Process(target=generate_embeddings, args=(args, in_queue, out_queue, gpu))
        worker.start()
        workers.append(worker)

    return workers


def generate_embeddings(args, in_queue, out_queue, device_id=None):
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(get_device(device_id))
    model.eval()

    done = False
    while not done:
        msg, snippet = in_queue.get()

        if msg == "done":
            done = True
            break

        tokenized = tokenizer.encode(snippet, return_tensors="pt", truncation=True)
        tokenized = tokenized.to(get_device(device_id))

        with torch.no_grad():
            embeddings = model(tokenized)[0].cpu().numpy()
            embedding = embeddings.mean(axis=1)
            out_queue.put(("complete", embedding))


def trim_code(snippet, api):
    lines = snippet.split("\n")
    api_line = None
    for i, line in enumerate(lines):
        if api in line:
            api_line = i
            break
    if api_line is None:
        return snippet

    # remove empty lines
    lines = [line for line in lines if line.strip() != ""]

    start_line = max(0, api_line - 2)
    end_line = min(len(lines), api_line + 3)
    return "\n".join(lines[start_line:end_line])


def load_program(path):
    with open(path, "r") as f:
        program = f.read()
    return program


def embed_programs(args, progs):
    num_gpus = torch.cuda.device_count()
    workers_list = []
    in_queues, out_queues = [], []

    for gpu in range(num_gpus):
        in_queue, out_queue = mp.Queue(), mp.Queue()
        workers = start_workers_genemb(in_queue, out_queue, args, gpu)
        workers_list.append(workers)
        in_queues.append(in_queue)
        out_queues.append(out_queue)

    for i, prog in enumerate(progs):
        in_queues[i % num_gpus].put(("prog", prog))

    results_count = 0
    embeddings = []
    pbar = tqdm(total=len(progs), desc=f"[embed]")
    while results_count < len(progs):
        for gpu in range(num_gpus):
            while not out_queues[gpu].empty():
                msg, prog_embedding = out_queues[gpu].get()
                results_count += 1
                pbar.update(1)
                embeddings.append(prog_embedding)
    pbar.close()

    for in_queue in in_queues:
        for _ in range(num_gpus):
            in_queue.put(("done", None))

    for workers in workers_list:
        for worker in workers:
            worker.join()

    return np.array(embeddings)


def main(args):
    # Load all python code snippets that contain the API
    prog_indices = grep_programs(args, api)
    progs = [load_program(f"{args.prog_dir}/example_{i}.py") for i in prog_indices]
    progs = [trim_code(prog, api) for prog in progs]
    prog_embeddings = embed_programs(args, progs)
    prog_embeddings = np.concatenate(prog_embeddings, axis=0)

    # Load all idioms for the API
    idioms = [load_program(osp.join(args.idioms_dir, file)) for file in os.listdir(args.idioms_dir)]
    idiom_embeddings = embed_programs(args, idioms)
    idiom_embeddings = np.concatenate(idiom_embeddings, axis=0)

    return compute_emd(prog_embeddings, idiom_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=date.today())
    parser.add_argument("--dataset", type=str, default="pnosmt")
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
            # args.idioms_dir = f"../results/{args.date}/{lib}_res/{api}/idioms/progs"
            args.idioms_dir = f"../gpt/results/{args.date}/{lib}_res/{api}/"
            emd = main(args)

            print(f"========== [{lib}: {api}] ==========")
            print(f"EMD: {emd}")
            print("=====================================\n\n")
