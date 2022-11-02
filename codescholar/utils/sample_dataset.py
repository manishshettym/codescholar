import glob
from random import random
import numpy as np

from tqdm import tqdm
import os
import os.path as osp

from codescholar.utils.code_utils import breakdown_code_methods

SRC_DIR = "../../../data/pandas/"
DEST_DIR = "../representation/tmp/pandas/raw/"

if not osp.exists(osp.dirname(DEST_DIR)):
    os.makedirs(os.path.dirname(DEST_DIR))

files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, '*.py')))]
random_files = np.random.choice(files, min(len(files), 100))

idx = 0
for file in tqdm(random_files):
    breakdown_code_methods(DEST_DIR, file, file_id="example{}".format(idx))
    idx += 1
