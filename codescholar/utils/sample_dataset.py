import glob
import numpy as np

from tqdm import tqdm
import os
import os.path as osp

from codescholar.utils.code_utils import breakdown_code_methods

SRC_DIR = "../../../data/pandas/"
DEST_DIR = "../representation/tmp/pandas/"
TRAIN_DIR = DEST_DIR + "train/raw/"
TEST_DIR = DEST_DIR + "test/raw/"

if not osp.exists(osp.dirname(TRAIN_DIR)):
    os.makedirs(os.path.dirname(TRAIN_DIR))

if not osp.exists(osp.dirname(TEST_DIR)):
    os.makedirs(os.path.dirname(TEST_DIR))

files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, '*.py')))]
random_files = np.random.choice(files, min(len(files), 1000))
train_len = int(0.8 * len(random_files))

idx = 0
for file in tqdm(random_files):
    if idx < train_len:
        breakdown_code_methods(
            TRAIN_DIR, file,
            file_id="example{}".format(idx))
    else:
        breakdown_code_methods(
            TEST_DIR, file,
            file_id="example{}".format(idx))

    idx += 1
