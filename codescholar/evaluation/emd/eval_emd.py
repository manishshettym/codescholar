"""Compute EMD - Earth Mover's Distance - between two distributions:
1. The distribution of the embeddings of a set of python code snippets
2. The distribution of the embeddings of a set of idioms (result of CodeScholar)
"""
import os
import os.path as osp
import argparse
from tqdm import tqdm

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from transformers import AutoTokenizer, AutoModel

from codescholar.search.elastic_search import grep_programs
from codescholar.utils.train_utils import get_device

def compute_emd(code_embeddings, idiom_embeddings):
    # Compute the distance matrix between the embeddings of the python code snippets and the idioms
    distance_matrix = cdist(code_embeddings, idiom_embeddings, metric='euclidean')

    # Compute the EMD between the two distributions using the distance matrix
    row_ind, col_ind = linear_sum_assignment(distance_matrix)
    emd = distance_matrix[row_ind, col_ind].sum() / len(row_ind)

    return emd


# Load the CodeBERT model and tokenizer
model_name = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def generate_embeddings(code_snippets, batch_size=32):
    # Tokenize the code snippets
    tokenized = [tokenizer.encode(snippet, return_tensors='pt', truncation=True) for snippet in code_snippets]
    tokenized = [t.reshape(-1) for t in tokenized]
    max_len = max([len(t) for t in tokenized])
    
    # Pad the tokenized snippets
    padded = [torch.cat([t, torch.zeros(max_len - len(t), dtype=torch.int64)]) for t in tokenized]
    padded = torch.stack(padded)
        
    # Generate embeddings in batches
    device = get_device()
    model.to(device)
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(padded), batch_size)):
            batch = padded[i:i+batch_size].to(device)
            batch_embeddings = model(batch)[0].cpu().numpy()
            
            # take the mean of the embeddings of the tokens in each snippet
            batch_embeddings = np.mean(batch_embeddings, axis=1)
            embeddings.append(batch_embeddings)

    embeddings = np.concatenate(embeddings, axis=0)

    return embeddings


def trim_code(snippet, api):
    lines = snippet.split('\n')
    api_line = None
    for i, line in enumerate(lines):
        if api in line:
            api_line = i
            break
    if api_line is None:
        return snippet
    start_line = max(0, api_line - 2)
    end_line = min(len(lines), api_line + 3)
    return '\n'.join(lines[start_line:end_line])


def load_program(path):
    with open(path, "r") as f:
        program = f.read()
    return program


def main(args):
    # Load all python code snippets that contain the API
    prog_indices = grep_programs(args, api)
    progs = [load_program(f"{args.prog_dir}/example_{i}.py") for i in prog_indices]
    progs = [trim_code(prog, api) for prog in progs]
    prog_embeddings = generate_embeddings(progs, batch_size=256)

    # Load all idioms for the API
    idioms = [load_program(osp.join(args.idioms_dir, file)) for file in os.listdir(args.idioms_dir)]
    idiom_embeddings = generate_embeddings(idioms, batch_size=256)
    
    emd = compute_emd(prog_embeddings, idiom_embeddings)
    print(f"EMD: {emd}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    lib = "pandas"
    api = "df.groupby"
    
    args.dataset = "pnosmt"
    args.prog_dir = f"../../data/{args.dataset}/source/"
    args.idioms_dir = f"../results/2023-06-21/{lib}_res/{api}/idioms/progs"
    
    main(args)
    