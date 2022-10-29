import glob
from random import random
import numpy as np

from tqdm import tqdm
import os
import os.path as osp
import shutil

SRC_DIR = "../../../data/pandas/"
DEST_DIR = "../representation/tmp/pandas/raw/"

if not osp.exists(osp.dirname(DEST_DIR)):
    os.makedirs(os.path.dirname(DEST_DIR))

files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, '*.py')))]
random_files = np.random.choice(files, 500)

idx = 0
for file in tqdm(random_files):
    shutil.copy(file, osp.join(DEST_DIR, "example{}.py".format(idx)))
    idx += 1
