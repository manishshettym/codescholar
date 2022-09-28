import os
import sys
import git
import json

from tqdm import tqdm
import glob
import shutil
from typing import List


def load_repository_paths(path: str):
    with open(path) as fp:
        repos = json.load(fp)["items"]

    repos = [repo['name'] for repo in repos]

    return repos


def is_library_used(filepath: str, lib: str):
    keywords = [f'import {lib}', f'from {lib}']

    with open(filepath) as fp:
        text = fp.read()

        if any(usage in text for usage in keywords):
            return True

    return False


def get_library_clients(paths: List[str], lib: str):
    data_dir = "../../data/github/"
    clone_loc = f"{data_dir}/{lib}"
    temp_dir = f"{data_dir}/temp"

    # create campaign directory:
    if not os.path.isdir(f'{clone_loc}'):
        os.makedirs(clone_loc)

    for repo_path in tqdm(paths):

        # create scratchpad directory for copying
        os.makedirs(temp_dir)

        # clone repo
        git.Git(clone_loc).clone(f"https://github.com/{repo_path}")

        repo_name = repo_path.split("/")[-1]
        repo_loc = os.path.join(clone_loc, repo_name)
        has_lib_usage = False

        # find lib usages
        for path in glob.glob(f"{repo_loc}/**", recursive=True):

            if os.path.isfile(path) and path.endswith(".py"):

                if is_library_used(path, lib):

                    filename = os.path.basename(path)
                    destination = os.path.join(temp_dir, filename)
                    shutil.move(path, destination)

                    has_lib_usage = True

        # remove repo
        shutil.rmtree(repo_loc)

        # rename or remove scratchpad
        if has_lib_usage:
            os.rename(temp_dir, os.path.join(data_dir, lib, repo_name))
        else:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    github_repo_list = "../../data/github/repositories.json"
    repositories = load_repository_paths(github_repo_list)

    library = "tensorflow"
    get_library_clients(repositories, library)
