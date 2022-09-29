import os
import git
import json

import glob
import shutil
from tqdm import tqdm
from typing import List

from codescholar.utils import multiprocess

MAX_WORKERS = 12


def load_repository_paths(path: str):
    with open(path, 'rb') as fp:
        repos = json.load(fp)["items"]

    repos = sorted(list(set([repo['name'] for repo in repos])))

    return repos


def is_library_used(filepath: str, lib: str):
    keywords = [f'import {lib}', f'from {lib}']

    with open(filepath, encoding="utf8", errors='ignore') as fp:
        text = fp.read()

        if any(usage in text for usage in keywords):
            return True

    return False


def _repo_lib_clients_mp(args):
    repo_path, lib = args
    return repo_lib_clients(repo_path, lib)


def repo_lib_clients(repo_path: str, lib: str):
    data_dir = "../../data/github/"
    clone_loc = f"{data_dir}/{lib}"

    repo_name = repo_path.split("/")[-1]
    repo_loc = os.path.join(clone_loc, repo_name)
    has_lib_usage = False

    # create scratchpad directory for copying
    temp_dir = f"{data_dir}/{repo_name}-temp"
    os.makedirs(temp_dir)

    # clone repo
    try:
        git.Git(clone_loc).clone(f"https://github.com/{repo_path}", depth=1)
    except:
        shutil.rmtree(repo_loc)
        return "uncloned"

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
        return "mined"
    else:
        shutil.rmtree(temp_dir)
        return "skipped"


def get_library_clients(paths: List[str], lib: str):
    data_dir = "../../data/github/"
    clone_loc = f"{data_dir}/{lib}"

    if not os.path.isdir(f'{clone_loc}'):
        os.makedirs(clone_loc)

    lib_clients_iter = multiprocess.run_tasks_in_parallel_iter(
        _repo_lib_clients_mp,
        tasks=[(path, lib) for path in paths],
        use_progress_bar=True,
        use_spawn=True,
        num_workers=MAX_WORKERS)

    count = 0
    for path, result in tqdm(zip(paths, lib_clients_iter)):
        if (result.is_success() and isinstance(result.result, str)):
            print(f"Repo: {path} => [{result.result}]", flush=True)
            if result.result == "mined":
                count += 1

    print("==" * 20 + " [CodeScholar::Github Miner Summary] " + "==" * 20)
    print(f"#Repos: {len(paths)}")
    print(f"Lib: {lib}")
    print(f"#Mined-Repos: {count}")
    print("==" * 60)


if __name__ == "__main__":
    github_repo_list = "../../data/github/repositories.json"
    library = "pandas"

    repositories = load_repository_paths(github_repo_list)
    get_library_clients(repositories, library)
