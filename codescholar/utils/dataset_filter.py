"""[script] filter methods in the dataset by keywords and log the selected methods"""
import json
import glob
import os.path as osp
from tqdm import tqdm

import torch.multiprocessing as mp

 
def select_code(method_content):
    for keyword in keywords:
        if keyword in method_content:
            return True
    
    return False

def start_workers(in_queue, out_queue):
    workers = []
    for _ in tqdm(range(NUM_WORKERS), desc="Workers"):
        worker = mp.Process(target=mp_filter, 
                    args=(in_queue, out_queue))
        worker.start()
        workers.append(worker)

    return workers

def mp_filter(in_queue, out_queue):
    done = False    
    while not done:
        msg, methpath = in_queue.get()

        if msg == "done":
            done = True
            break

        with open(methpath, "r") as f:
            method_content = f.read()
            if select_code(method_content):
                out_queue.put(methpath)
            else:
                out_queue.put('filtered')


def filter_dataset(method_paths):    
    in_queue, out_queue = mp.Queue(), mp.Queue()
    workers = start_workers(in_queue, out_queue)
    
    for methpath in method_paths:
        in_queue.put(("method", methpath))

    count = 0
    with open("../data/pnosmt/methods_selected.txt", "w") as f:
        for _ in tqdm(range(len(method_paths)), desc="Filter"):
            methpath = out_queue.get()
            if methpath != 'filtered':
                f.write(methpath + "\n")
                count += 1
    
    print(f"Total number of methods after filtering: {count}")

    for _ in range(NUM_WORKERS):
        in_queue.put(("done", None))

    for worker in workers:
        worker.join()


if __name__ == "__main__":
    NUM_WORKERS = 32
    keywords = []
    
    # choose keywords for a domain-specific dataset
    # NOTE: here we use 65 APIs from top-6 libraries:
    # {pandas, numpy, os, sklearn, matplotlib, torch}
    with open("../evaluation/benchmarks.json", "r") as f:
        benchmarks = json.load(f)

    for lib in benchmarks:
        for api in benchmarks[lib]:
            keywords.append(api)
            
    SRC_DIR = "../data/pnosmt/methods"
    method_paths = glob.glob(osp.join(SRC_DIR, "*.py"))

    filter_dataset(method_paths)
    