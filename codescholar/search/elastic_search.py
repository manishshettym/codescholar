"""Elasticsearch indexing and search for Python programs."""
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from tqdm import tqdm
import glob
import os.path as osp
import argparse

import torch.multiprocessing as mp

from codescholar.representation import config
from codescholar.search import search_config
from codescholar.constants import DATA_DIR


def start_workers_bulk_index(in_queue, out_queue, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(target=worker_bulk_index, args=(args, in_queue, out_queue))
        worker.start()
        workers.append(worker)

    return workers


def worker_bulk_index(args, in_queue, out_queue):
    done = False
    while not done:
        msg, idx = in_queue.get()

        if msg == "done":
            done = True
            break

        path = osp.join(args.prog_dir, f"example_{idx}.py")
        with open(path) as file:
            contents = file.read()

        out_queue.put(
            {"_index": "python_files", "_id": idx, "_source": {"content": contents}}
        )


def bulk_index_generator(count, out_queue):
    for _ in tqdm(range(count), desc="Indexing"):
        data = out_queue.get()
        yield data


def index_files(args):
    es = Elasticsearch("http://localhost:9200/")

    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers_bulk_index(in_queue, out_queue, args)
    valid_indices = [
        f.split("_")[-1][:-3] for f in glob.glob(osp.join(args.emb_dir, "*.pt"))
    ]

    for i in valid_indices:
        in_queue.put(("idx", i))

    bulk(es, bulk_index_generator(len(valid_indices), out_queue), chunk_size=1000)

    for _ in range(args.n_workers):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()


def grep_programs(args, keyword: str):
    """Search for programs containing the keyword."""
    es = Elasticsearch("http://localhost:9200/")

    # Perform the search
    res = es.search(
        index="python_files",
        body={"query": {"match": {"content": keyword}}},
        size=10000,
        scroll="1m",
    )

    sid = res["_scroll_id"]
    scroll_size = len(res["hits"]["hits"])

    matching_index = []

    while scroll_size > 0:
        matching_index += [hit["_id"] for hit in res["hits"]["hits"]]

        res = es.scroll(scroll_id=sid, scroll="1m")
        sid = res["_scroll_id"]
        scroll_size = len(res["hits"]["hits"])

    es.clear_scroll(scroll_id=sid)

    return matching_index


if __name__ == "__main__":
    """Usage: python elastic_search.py --dataset <dataset>"""
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    # data config
    args.prog_dir = f"{DATA_DIR}/{args.dataset}/source/"
    args.source_dir = f"{DATA_DIR}/{args.dataset}/graphs/"
    args.emb_dir = f"{DATA_DIR}/{args.dataset}/emb/"

    index_files(args)
