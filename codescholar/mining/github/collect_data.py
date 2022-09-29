import os
import git
import json
import shutil
from typing import List

import pygments
from pygments.lexers import get_lexer_by_name
from pygments.token import Token

from codescholar.utils import multiprocess

MAX_WORKERS = 12
MAX_FILE_SIZE = 1024 ** 2  # 1 MB
MIN_FILE_TOKENS = 100


def load_repository_paths(path: str):
    with open(path, 'rb') as fp:
        repos = json.load(fp)["items"]

    repos = sorted(list(set([repo['name'] for repo in repos])))

    return repos


def repo_cloner(repo_path: str):
    lexer = get_lexer_by_name('python')
    repo_dir = f"Repos"
    code_dir = f"Code"

    # create repo dir (temp)
    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)

    # create code dir
    if not os.path.exists(code_dir):
        os.makedirs(code_dir)

    repo_name = repo_path.split("/")[-1]
    repo_loc = os.path.join(repo_dir, repo_name)
    code_loc = os.path.join(code_dir, repo_name)

    # skip repos for which python files have been extracted already
    if os.path.exists(code_loc):
        print(f'{repo_name} seen before!')
        return "skipped"
    else:
        os.makedirs(code_loc)

    # clone repo
    try:
        git.Git(repo_dir).clone(f"https://github.com/{repo_path}", depth=1)
    except Exception as e:
        print(f'{repo_name} clone error!')
        shutil.rmtree(repo_loc)
        return "uncloned"

    # filter and move files
    files_found = 0
    for root, _, files in os.walk(repo_loc):
        for file in files:
            if file.endswith(".py"):
                in_path = os.path.join(root, file)

                if not os.path.exists(in_path):
                    continue

                if os.path.getsize(in_path) > MAX_FILE_SIZE:
                    continue

                with open(in_path, errors='ignore') as f_in:
                    text = f_in.read()

                if sum(1 for _ in pygments.lex(text, lexer)) < MIN_FILE_TOKENS:
                    continue

                # create a simplified path for file
                rel_path = root[len(repo_loc)+1:].replace('/', '__')
                out_path = os.path.join(code_loc, rel_path
                                        + ('__' if rel_path else '') + file)

                if not os.path.exists(out_path):
                    try:
                        shutil.copyfile(in_path, out_path)
                    except Exception as e:
                        raise e

                files_found += 1

    # remove repo
    shutil.rmtree(repo_loc)
    print(f'{repo_name} processed; {files_found} #files copied.')

    return "mined"


def clone_repositories(paths: List[str]):
    lib_clients_iter = multiprocess.run_tasks_in_parallel_iter(
        repo_cloner,
        tasks=paths,
        use_progress_bar=False,
        use_spawn=True,
        num_workers=MAX_WORKERS)

    count = 0
    for result in lib_clients_iter:
        if (result.is_success() and isinstance(result.result, str)):
            if result.result == "mined":
                count += 1

    print("==" * 20 + " [CodeScholar::Github Miner Summary] " + "==" * 20)
    print(f"#Repos: {len(paths)}")
    print(f"#Mined-Repos: {count}")
    print("==" * 60)


if __name__ == "__main__":
    github_repo_list = "python-top-repos.json"
    repositories = load_repository_paths(github_repo_list)
    clone_repositories(repositories[:10])
