import glob
import numpy as np

from tqdm import tqdm
import os
import os.path as osp

from codescholar.utils.code_utils import breakdown_code_methods

SRC_DIR = "../data/pandas/raw"

# create methods to train neural subgraph matcher
DEST_DIR = "../representation/tmp/pandas/"
TRAIN_DIR = DEST_DIR + "train/raw/"
TEST_DIR = DEST_DIR + "test/raw/"

# create methods to build latent space
ALL_DIR = "../data/pandas/source/"
NO_SPLIT = True

# sampling source files
SAMPLES = -1
files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, '*.py')))]

if SAMPLES != -1:
    random_files = np.random.choice(files, min(len(files), 1000))
    train_len = int(0.8 * len(random_files))
else:
    random_files = files


########### SCRIPT ###########

if not osp.exists(osp.dirname(TRAIN_DIR)):
    os.makedirs(os.path.dirname(TRAIN_DIR))

if not osp.exists(osp.dirname(TEST_DIR)):
    os.makedirs(os.path.dirname(TEST_DIR))

if not osp.exists(osp.dirname(ALL_DIR)):
    os.makedirs(os.path.dirname(ALL_DIR))

idx = 0
for file in tqdm(random_files):
    if NO_SPLIT:
        breakdown_code_methods(
                ALL_DIR, file,
                file_id="example{}".format(idx))
    else:
        if idx < train_len:
            breakdown_code_methods(
                TRAIN_DIR, file,
                file_id="example{}".format(idx))
        else:
            breakdown_code_methods(
                TEST_DIR, file,
                file_id="example{}".format(idx))

    idx += 1
