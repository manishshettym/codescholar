import os
import os.path as osp
import re
from collections import Counter
from itertools import combinations
from tqdm import tqdm
import random
import numpy as np

import glob
import torch.multiprocessing as mp
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_api_calls(code):
    api_calls = []

    # Note: this regex is a quick hack to filter out some interesting APIs
    # However, codescholar's search is not limited to these APIs by any means!!
    pattern = r"\b(pd|df|np|os|json|nn|plt|pandas|sklearn|torch)\.([a-zA-Z_][a-zA-Z0-9_.]+)\("
    matches = re.findall(pattern, code)

    for match in matches:
        api = match[0] + "." + match[1]
        api_calls.append(api)

    return set(api_calls)


def process_file(file, api_pairs):
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        code = f.read()
        apis = extract_api_calls(code)

    pair_combos = list(combinations(apis, 2))
    api_pairs.append(pair_combos)


# ==================== MAIN ====================

LIBS = ["pandas", "numpy", "os", "sklearn", "matplotlib", "torch"]
SRC_DIR = "../data/pnosmt/raw"
files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, "*.py")))]

random.seed(42)
files = random.sample(files, 100000)

manager = mp.Manager()
api_pairs = manager.list()

num_processes = mp.cpu_count()
pool = mp.Pool(num_processes)

for file in tqdm(files):
    pool.apply_async(process_file, args=(file, api_pairs, api_triplets))

pool.close()
pool.join()

# flatten the list of lists
api_pairs_flat = [tuple(sorted(item)) for sublist in api_pairs for item in sublist]

pair_docs = []
for prog_pairs in api_pairs:
    doc = []
    for pair in prog_pairs:
        doc.append("-".join(pair))
    pair_docs.append(" ".join(doc))

pair_docs = [doc for doc in pair_docs if doc]
tokenizer = lambda doc: doc.split(" ")
tfidf = TfidfVectorizer(analyzer=tokenizer)
X = tfidf.fit_transform(pair_docs)
rank = np.argsort(np.asarray(X.sum(axis=0)).ravel())[::-1]
score = np.sort(np.asarray(X.sum(axis=0)).ravel())[::-1]
feature_names = np.array(tfidf.get_feature_names_out())
ranked_api_pairs = list(feature_names[rank])


print("================== TOP API PAIRS ==================")
for pair in Counter(api_pairs_flat).most_common(1000):
    try:
        api_pair = "-".join(pair[0])
        s = score[ranked_api_pairs.index(api_pair)]
    except:
        api_pair = "-".join(pair[0][::-1])
        s = score[ranked_api_pairs.index(api_pair)]

    print(f"apis: {pair[0]} | freq: {pair[1]} | tf-idf: {round(s, ndigits=1)}")
print("====================================================")
