import os.path as osp
import numpy as np
import glob

import torch


def sample_progs(src_dir, k=10000, seed=24):
    np.random.seed(seed)

    prog_embs = []
    prog_sizes = []
    files = [f for f in sorted(glob.glob(osp.join(src_dir, '*.pt')))]
    random_files = np.random.choice(files, min(len(files), k))

    for file in random_files:
        embs = torch.load(file, map_location=torch.device('cpu'))
        prog_embs.append(embs)
        prog_sizes.append(len(embs))
        
    return torch.cat(prog_embs, dim=0), random_files, prog_sizes
