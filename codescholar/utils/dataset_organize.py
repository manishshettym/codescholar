"""[script] for the search dataset:
1. apply any filters to to reduce search space
2. organize methods into a standard format (example_*.py)
3. update mappings to reflect the new file names
"""
import os
import json
import os.path as osp
from tqdm import tqdm
import argparse

import glob
import torch.multiprocessing as mp

def start_workers_rename(in_queue, out_queue, methods_to_fileid, args):
    workers = []
    for _ in tqdm(range(args.n_workers), desc="Workers"):
        worker = mp.Process(target=mp_rename, 
                    args=(args, methods_to_fileid, in_queue, out_queue))
        worker.start()
        workers.append(worker)

    return workers


def mp_rename(args, methods_to_fileid, in_queue, out_queue):
    done = False    
    while not done:
        msg, methpath, meth_idx = in_queue.get()

        if msg == "done":
            done = True
            break

        method_filename = osp.basename(methpath)
        example_id = f"example_{meth_idx}.py"

        with open(methpath, "r") as f:
            method_content = f.read()
            with open(osp.join(args.dest_dir, example_id), "w") as f:
                f.write(method_content)

        out_queue.put((method_filename, example_id))


def standardize_dataset_files(method_paths, methods_to_fileid):    
    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers_rename(in_queue, out_queue, methods_to_fileid, args)
    
    for idx, methpath in enumerate(method_paths):
        in_queue.put(("method", methpath, idx))
    
    for _ in tqdm(range(len(method_paths)), desc="Rename"):
        method_filename, example_id = out_queue.get()
        methods_to_fileid[example_id] = methods_to_fileid.pop(method_filename)
    
    for _ in range(args.n_workers):
        in_queue.put(("done", None, None))

    for worker in workers:
        worker.join()

    with open(f"../data/{args.dataset}/mappings/example_to_fileid.json", "w") as f:
        json.dump(methods_to_fileid, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers")
    args = parser.parse_args()

    SRC_DIR = f"../data/{args.dataset}/methods"
    method_paths = sorted(glob.glob(osp.join(SRC_DIR, "*.py")))

    args.dest_dir = f"../data/{args.dataset}/source"
    if not osp.exists(args.dest_dir):
        os.makedirs(args.dest_dir)

    # read in methods_to_fileid
    with open(f"../data/{args.dataset}/mappings/meth_to_fileid.json", "r") as f:
        methods_to_fileid = json.load(f)
    
    standardize_dataset_files(method_paths, methods_to_fileid)
