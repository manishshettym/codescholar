"""Utilities for CodeBERT embeddings."""

from tqdm import tqdm
import numpy as np

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModel

from codescholar.utils.train_utils import get_device


def start_workers_genemb(in_queue, out_queue, args, gpu):
    """start a pool of workers for generating embeddings."""
    torch.cuda.set_device(gpu)
    workers = []
    for _ in tqdm(range(args.n_workers), desc="[workers]"):
        worker = mp.Process(target=generate_embeddings, args=(args, in_queue, out_queue, gpu))
        worker.start()
        workers.append(worker)

    return workers


def generate_embeddings(args, in_queue, out_queue, device_id=None):
    """worker function for generating embeddings using CodeBERT."""
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


def embed_programs_codebert(args, progs):
    """Embed a list of programs using CodeBERT.
    Note: creates a multiprocessing pool of workers.
    """
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
