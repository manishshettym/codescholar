"""script: sample files from a directory and breakdow code into methods/functions"""
import glob
import pandas as pd
import numpy as np
import argparse

from tqdm import tqdm
import os
import os.path as osp

from codescholar.utils.code_utils import breakdown_code_methods, is_library_used


def create_train_test_dataset(args, files):
    """create train and test dataset from a list of files

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
            c, _ = breakdown_code_methods(
                outdir=TRAIN_DIR, path=file, file_id="example{}".format(idx)
            )
            train_count += c
        else:
            c, _ = breakdown_code_methods(
                outdir=TEST_DIR, path=file, file_id="example{}".format(idx)
            )
            test_count += c

        idx += 1

    print(
        "Train: {} Test: {} Total methods: {}".format(
            train_count, test_count, train_count + test_count
        )
    )


def create_search_dataset(args, files):
    """create search dataset from a list of files

    Args:
        args (argparse.Namespace): arguments
        files (list): list of files to create dataset from
    """
    if not osp.exists(osp.dirname(DEST_DIR)):
        os.makedirs(os.path.dirname(DEST_DIR))

    idx, count = 0, 0
    methods_to_fileid = {}

    for file in tqdm(files):
        c, methods = breakdown_code_methods(
            outdir=DEST_DIR, path=file, file_id="example{}".format(idx)
        )

        for m in methods:
            methods_to_fileid[m] = osp.basename(file)

        count += c
        idx += 1

    src_to_repo_df = pd.read_csv(
        SRC_2_REPO,
        header=0,
        sep=";",
        on_bad_lines="skip",
    )

    method_to_repo = []

    # rename to a standard filename format
    meth_idx = 0
    method_paths = sorted(glob.glob(osp.join(DEST_DIR, "*.py")))
    for idx, mp in enumerate(method_paths):
        method_filename = osp.basename(mp)
        source_fileid = methods_to_fileid[method_filename]
        src_info = src_to_repo_df.loc[src_to_repo_df["fileid"] == source_fileid].values.tolist()[0]
        
        methodid = f"example_{meth_idx}.py"
        method_to_repo.append([methodid] + src_info)
        os.rename(mp, osp.join(DEST_DIR, methodid))
        meth_idx += 1
    
    meth_to_repo_df = pd.DataFrame(method_to_repo, columns=["methodid", "fileid", "file", "repo"])
    meth_to_repo_df.to_csv(METH_2_REPO, sep=";", index=False)
    print("Total number of methods generated: {}".format(count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "search"],
        help="Task for which we are sampling data",
    )
    parser.add_argument(
        "--samples", type=int, default=-1, help="Number of samples to use"
    )
    parser.add_argument("--dataset", type=str, help="Dataset to use")
    args = parser.parse_args()

    SRC_DIR = f"../data/{args.dataset}/raw"
    SRC_2_REPO = f"../data/{args.dataset}/src_to_repo.csv"
    METH_2_REPO = f"../data/{args.dataset}/meth_to_repo.csv"

    files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, "*.py")))]

    if args.samples != -1:
        sampled_files = np.random.choice(files, min(len(files), args.samples))
        train_len = int(0.8 * len(sampled_files))
    else:
        sampled_files = files

    # create methods to train neural subgraph matcher
    if args.task == "train":
        DEST_DIR = f"../representation/tmp/{args.dataset}/"
        TRAIN_DIR = DEST_DIR + "train/raw/"
        TEST_DIR = DEST_DIR + "test/raw/"

        create_train_test_dataset(args, sampled_files)

    # create methods to build search space
    elif args.task == "search":
        DEST_DIR = f"../data/{args.dataset}/source/"
        create_search_dataset(args, sampled_files)
