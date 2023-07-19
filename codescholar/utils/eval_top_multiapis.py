import os
import os.path as osp
import re
from collections import Counter
from itertools import combinations
from tqdm import tqdm

import glob
import torch.multiprocessing as mp


def extract_api_calls(code):
    api_calls = []

    # Note: this regex is a quick hack to filter out some interesting APIs
    # However, codescholar's search is not limited to these APIs by any means!!
    pattern = r'\b(pd|df|np|os|json|nn|plt|pandas|sklearn|torch)\.([a-zA-Z_][a-zA-Z0-9_.]+)\('
    matches = re.findall(pattern, code)

    for match in matches:
        api = match[0] + '.' + match[1]
        api_calls.append(api)

    return set(api_calls)


def process_file(file, api_pairs, api_triplets):
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
        apis = extract_api_calls(code)

    pair_combos = list(combinations(apis, 2))
    triplet_combos = list(combinations(apis, 3))
    
    api_pairs.append(pair_combos)
    api_triplets.append(triplet_combos)

# ==================== MAIN ====================

LIBS = ["pandas", "numpy", "os", "sklearn", "matplotlib", "torch"]
SRC_DIR = "../data/pnosmt/raw"
files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, "*.py")))]

manager = mp.Manager()
api_pairs = manager.list()
api_triplets = manager.list()

num_processes = mp.cpu_count()
pool = mp.Pool(num_processes)

for file in tqdm(files):
    pool.apply_async(process_file, args=(file, api_pairs, api_triplets))

pool.close()
pool.join()

# flatten the list of lists
api_pairs = [item for sublist in api_pairs for item in sublist]
api_triplets = [item for sublist in api_triplets for item in sublist]

print("================== TOP API PAIRS ==================")
for pair in Counter(api_pairs).most_common():
    print(f"apis: {pair[0]} | freq: {pair[1]}")
print("====================================================")

print("================ TOP API TRIPLETS ==================")
for triplet in Counter(api_triplets).most_common():
    print(f"apis: {triplet[0]} | freq: {triplet[1]}")
print("====================================================")
