"""[script] for search dataset:
1. create dataset from a directory by breaking down code into methods/functions
2. apply any filters on methods generated to to reduce search space
"""
import os
import json
import os.path as osp
from tqdm import tqdm
import argparse

import glob
import pandas as pd
import numpy as np
import torch.multiprocessing as mp

from codescholar.utils.code_utils import breakdown_code_methods
from codescholar.constants import DATA_DIR

#############################################
######### Create Train/Test Dataset #########
#############################################


def create_train_test_dataset(args, files):
    """create train and test dataset from a list of files

    Args:
        args (argparse.Namespace): arguments
        files (list): list of files to create dataset from
    """
    if not osp.exists(osp.dirname(TRAIN_DIR)):
        os.makedirs(os.path.dirname(TRAIN_DIR))

    if not osp.exists(osp.dirname(TEST_DIR)):
        os.makedirs(os.path.dirname(TEST_DIR))

    idx = 0
    train_count = 0
    test_count = 0
    for file in tqdm(files):
        if idx < train_len:
            c, _ = breakdown_code_methods(
                outdir=TRAIN_DIR, path=file, file_id="example{}".format(idx)
            )
            train_count += c
        else:
            c, _ = breakdown_code_methods(
                outdir=TEST_DIR, path=file, file_id="example{}".format(idx)
            )
            test_count += c

        idx += 1

    print(
        "Train: {} Test: {} Total methods: {}".format(
            train_count, test_count, train_count + test_count
        )
    )


#############################################
########## Create Search Dataset ############
#############################################


def start_workers_breakdown(in_queue, out_queue, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(target=mp_breakdown, args=(args, in_queue, out_queue))
        worker.start()
        workers.append(worker)

    return workers


def mp_breakdown(args, in_queue, out_queue):
    done = False
    while not done:
        msg, file, file_idx = in_queue.get()

        if msg == "done":
            done = True
            break

        meth_count, methods = breakdown_code_methods(
            outdir=args.dest_dir, path=file, file_id="example{}".format(file_idx)
        )

        out_queue.put((file, meth_count, methods))


def create_search_dataset(args, files):
    """create search dataset from a list of files

    Args:
        args (argparse.Namespace): arguments
        files (list): list of files to create dataset from
    """
    if not osp.exists(osp.dirname(DEST_DIR)):
        os.makedirs(os.path.dirname(DEST_DIR))

    args.dest_dir = DEST_DIR
    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers_breakdown(in_queue, out_queue, args)

    count, methods_to_fileid = 0, {}

    for idx, file in enumerate(files):
        in_queue.put(("file", file, idx))

    for _ in tqdm(range(len(files)), desc="Breakdown"):
        file, meth_count, methods = out_queue.get()
        count += meth_count

        for m in methods:
            methods_to_fileid[m] = osp.basename(file)

    for _ in range(args.n_workers):
        in_queue.put(("done", None, None))

    for worker in workers:
        worker.join()

    print("Total number of methods generated: {}".format(count))

    return methods_to_fileid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "search"],
        help="Task for which we are sampling data",
    )
    parser.add_argument(
        "--samples", type=int, default=-1, help="Number of samples to use"
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    SRC_DIR = f"{DATA_DIR}/{args.dataset}/raw"
    files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, "*.py")))]

    if args.samples != -1:
        sampled_files = np.random.choice(files, min(len(files), args.samples))
        train_len = int(0.8 * len(sampled_files))
    else:
        sampled_files = files

    # create methods to train neural subgraph matcher
    if args.task == "train":
        DEST_DIR = f"../representation/tmp/{args.dataset}/"
        TRAIN_DIR = DEST_DIR + "train/raw/"
        TEST_DIR = DEST_DIR + "test/raw/"

        create_train_test_dataset(args, sampled_files)

    # create methods to build search space
    elif args.task == "search":
        DEST_DIR = f"{DATA_DIR}/{args.dataset}/methods/"
        methods_to_fileid = create_search_dataset(args, sampled_files)

        with open(f"{DATA_DIR}/{args.dataset}/mappings/meth_to_fileid.json", "w") as f:
            json.dump(methods_to_fileid, f)
