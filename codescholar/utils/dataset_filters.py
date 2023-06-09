"""script: create a custom dataset from a github snapshot w/ filters (mainly library usage)"""
import os
import os.path as osp
import glob
import shutil
import pandas as pd
from tqdm import tqdm
import argparse
import torch.multiprocessing as mp


def start_workers(in_queue, out_queue, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(target=filter, args=(args, in_queue, out_queue))
        worker.start()
        workers.append(worker)

    return workers


def filter(args, in_queue, out_queue):
    done = False

    keywords = []
    for lib in args.libs:
        keywords += [f"import {lib}", f"from {lib}"]

    while not done:
        msg, file = in_queue.get()

        if msg == "done":
            done = True
            break

        filter_file = True
        with open(file, encoding="utf8", errors="ignore") as fp:
            text = fp.read()

            # select files that import the specified libraries
            if any(usage in text for usage in keywords):
                filter_file = False

        if not filter_file:
            out_queue.put(1)
        else:
            # filter out the file
            os.remove(file)
            out_queue.put(0)


def create_dataset(args, files):
    """create dataset from a list of files

    Args:
        args (argparse.Namespace): arguments
        files (list): list of files to create dataset from
    """

    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers(in_queue, out_queue, args)

    idx = 0

    for file in files:
        in_queue.put(("file", file))

    for _ in tqdm(range(len(files)), desc="Files"):
        idx += out_queue.get()

    for _ in range(args.n_workers):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()

    print("Total number of files retained: {}".format(idx))


def flatten_dataset(args, files):
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)

    mapping = []  # [[fullpath, filename, repo]]

    for file in tqdm(files, desc="Files"):
        if os.path.isfile(file):
            filename = os.path.basename(file)
            repo = os.path.dirname(file)
            mapping.append([file, filename, repo])
            os.rename(file, os.path.join(DEST_DIR, filename))

    mapping_df = pd.DataFrame(mapping, columns=["fullpath", "filename", "repo"])
    mapping_df.to_csv(os.path.join(DEST_DIR, "mapping.csv"), sep=";", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--libs", nargs="+", help="list of libraries to filter")
    parser.add_argument("--n_workers", type=int, default=4, help="number of workers")
    args = parser.parse_args()

    SRC_DIR = "../data/pnosmt/"
    DEST_DIR = "../data/pnosmt_flat/"

    files = [
        file
        for file in glob.glob(SRC_DIR + "/**", recursive=True)
        if os.path.isfile(file) and file.endswith(".py")
    ]
    print("Total number of files originally: {}".format(len(files)))

    create_dataset(args, files)
    flatten_dataset(args, files)
