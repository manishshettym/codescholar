import glob
import numpy as np
import argparse
    
from tqdm import tqdm
import os
import os.path as osp

from codescholar.utils.code_utils import breakdown_code_methods, is_library_used

def create_train_test_dataset(args, files):
    """ create train and test dataset from a list of files

    Args:
        args (argparse.Namespace): arguments
        files (list): list of files to create dataset from
    """
    if not osp.exists(osp.dirname(TRAIN_DIR)):
        os.makedirs(os.path.dirname(TRAIN_DIR))

    if not osp.exists(osp.dirname(TEST_DIR)):
        os.makedirs(os.path.dirname(TEST_DIR))
    
    idx = 0
    train_count = 0
    test_count = 0
    for file in tqdm(files):
        if idx < train_len:
            train_count += breakdown_code_methods(outdir=TRAIN_DIR, 
                                   path=file, 
                                   file_id="example{}".format(idx))
        else:
            test_count += breakdown_code_methods(outdir=TEST_DIR, 
                                   path=file, 
                                   file_id="example{}".format(idx))

        idx += 1
    
    print("Train: {} Test: {} Total methods: {}".format(train_count, test_count, train_count + test_count))


def create_search_dataset(args, files):
    """ create search dataset from a list of files
    
    Args:
        args (argparse.Namespace): arguments
        files (list): list of files to create dataset from
    """
    if not osp.exists(osp.dirname(DEST_DIR)):
        os.makedirs(os.path.dirname(DEST_DIR))

    idx = 0
    count = 0
    for file in tqdm(files):
        count += breakdown_code_methods(outdir=DEST_DIR, 
                               path=file, 
                               file_id="example{}".format(idx))
        idx += 1

    print("Total number of methods generated: {}".format(count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train',
                        choices=['train', 'search'], help='Task for which we are sampling data')
    parser.add_argument('--samples', type=int, default=-1, help='Number of samples to use')
    args = parser.parse_args()


    SRC_DIR = "../data/pandas/raw"
    files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, '*.py')))]

    if args.samples != -1:
        sampled_files = np.random.choice(files, min(len(files), args.samples))
        train_len = int(0.8 * len(sampled_files))
    else:
        sampled_files = files

    # create methods to train neural subgraph matcher
    if args.task == 'train':
        DEST_DIR = "../representation/tmp/pandas/"
        TRAIN_DIR = DEST_DIR + "train/raw/"
        TEST_DIR = DEST_DIR + "test/raw/"

        create_train_test_dataset(args, sampled_files)

    elif args.task == 'search':
        DEST_DIR = "../data/pandas/source/"
        create_search_dataset(args, sampled_files)
    