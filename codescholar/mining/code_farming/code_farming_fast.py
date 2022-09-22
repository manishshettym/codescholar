import os
import ast
import glob
import attrs
import random

from tqdm import tqdm
from typing import Dict, List, Tuple, Set

from codescholar.utils.logs import logger
from codescholar.utils import multiprocess
from codescholar.mining.code_farming.code_farming import (grow_idiom,
                                                          subgraph_matches,
                                                          _mp_subgraph_matches,
                                                          build_dataset_lookup)

MAX_WORKERS = 100


@attrs.define(eq=False, repr=False)
class MinedIdiom:
    idiom: ast.AST
    start_lineno: int
    end_lineno: int


def get_single_nodes(
    dataset: List[ast.AST],
    dataset_lookup: List[Dict[str, List]],
    gamma: float
) -> Set[ast.AST]:
    """Get all unique and frequent ast.stmt nodes in ast.walk
    order (bfs) that are not import or function/class definitions.

    Args:
        dataset (List[ast.AST]): list of program ASTs
        dataset_lookup (List[Dict[str, List]]): maps of type(node):occurences
        gamma (float): min frequency for mined program nodes

    Returns:
        _type_: a set of unique and frequent ast.stmt nodes in the dataset
    """
    stmts = []

    candidates: List[Tuple(ast.AST, Dict[str, List])] = []
    candidate_loc: List[Tuple(int, int)] = []

    for prog in dataset:
        for i in ast.walk(prog):
            if(isinstance(i, ast.stmt)
                and not isinstance(i, (ast.FunctionDef,
                                       ast.AsyncFunctionDef,
                                       ast.AsyncFunctionDef,
                                       ast.ClassDef,
                                       ast.Import,
                                       ast.ImportFrom))):
                
                candidates.append((i, dataset_lookup))
                candidate_loc.append((i.lineno, i.end_lineno))

    subgraph_mp_iter = multiprocess.run_tasks_in_parallel_iter(
        _mp_subgraph_matches,
        tasks=candidates,
        use_progress_bar=False,
        num_workers=MAX_WORKERS)

    for c, loc, result in zip(candidates, candidate_loc, subgraph_mp_iter):

        if (
            result.is_success()
            and isinstance(result.result, int)
            and result.result >= gamma
        ):
            stmts.append(MinedIdiom(c[0], loc[0], loc[1]))
    
    return set(stmts)


def save_idiom(mined_results, candidate_idiom, loc, nodecount, fileid):
    new_idiom = MinedIdiom(candidate_idiom, loc[0], loc[1])
    
    if nodecount not in mined_results:
        mined_results[nodecount] = {}
        mined_results[nodecount][fileid] = [new_idiom]

    elif fileid not in mined_results[nodecount]:
        mined_results[nodecount][fileid] = [new_idiom]

    else:
        mined_results[nodecount][fileid].append(new_idiom)

    return mined_results


def filewise_mine_code(
    dataset: List[ast.AST],
    fix_max_len: bool = False,
    max_len: int = 0
) -> dict:
    dataset, dataset_lookup = build_dataset_lookup(dataset)
    gamma: float = 0.1 * len(dataset)
    node_count: int = 1
    mined_results: dict = {}

    mined_results[1] = {}
    for fileid, prog in enumerate(tqdm(dataset)):
        mined_results[1][fileid] = get_single_nodes([prog],
                                                    dataset_lookup,
                                                    gamma)
    
    print("[Initialized Generation 0]")

    while (node_count in mined_results):
        print(f"[Generation {node_count} w/ [gamma]: {gamma**(1 / node_count)}]")

        for fileid in tqdm(mined_results[node_count].keys()):

            for idiom in mined_results[node_count][fileid]:
                candidates: List[Tuple(ast.AST, Dict[str, List])] = []
                candidates_loc: List[Tuple(int, int)] = []

                # pass 1: create candidate idioms by combining w/ single nodes
                for prog in mined_results[1][fileid]:
                    candidate_idiom = None

                    # don't grow unnatural sequence of operations
                    if prog.end_lineno <= idiom.end_lineno:
                        continue

                    try:
                        candidate_idiom = grow_idiom(idiom.idiom, prog.idiom)
                    except:
                        continue
                    finally:
                        if candidate_idiom is not None:
                            candidates.append((candidate_idiom,
                                               dataset_lookup))
                            candidates_loc.append((idiom.start_lineno,
                                                   prog.end_lineno))

                if MAX_WORKERS > 1:
                    # define a multiprocess worker
                    subgraph_mp_iter = multiprocess.run_tasks_in_parallel_iter(
                        _mp_subgraph_matches,
                        tasks=candidates,
                        use_progress_bar=False,
                        num_workers=MAX_WORKERS)

                    # pass 2: prune candidate idioms based on frequency
                    for (c, _), loc, result in zip(candidates,
                                                   candidates_loc,
                                                   subgraph_mp_iter
                                                   ):
                        if (
                            result.is_success()
                            and isinstance(result.result, int)
                            and result.result >= gamma**(1 / node_count)
                        ):

                            mined_results = save_idiom(mined_results,
                                                       c, loc,
                                                       node_count + 1,
                                                       fileid)
                
                else:
                    for (c, dataset_lookup), loc in zip(candidates,
                                                        candidates_loc):
                        result = subgraph_matches(c, dataset_lookup)

                        if result >= gamma**(1 / node_count):
                            mined_results = save_idiom(mined_results,
                                                       c, loc,
                                                       node_count + 1,
                                                       fileid)
                    else:
                        continue

        node_count += 1

        if((fix_max_len and node_count > max_len)
                or (gamma**(1 / node_count) < 1)):
            break

    return mined_results


if __name__ == "__main__":
    dataset = []
    path = "../../data/Python-master"
    # path = "../../data/examples"

    for filename in sorted(glob.glob(os.path.join(path, '*.py'))):
        with open(os.path.join(path, filename), 'r') as f:
            try:
                dataset.append(ast.parse(f.read()))
            except:
                pass
    
    random.seed(55)
    dataset = random.sample(dataset, k=25)
    
    mined_code = filewise_mine_code(dataset, fix_max_len=True, max_len=4)

    # ******************* CREATE MINING CAMPAIGN SUMMARY *******************

    print("==" * 20 + " [[CodeScholar::Concept Miner Summary]] " + "==" * 20)
    print(f"Dataset: {len(dataset)} progs")
    print(f"# Explorations: {len(mined_code)}")
    print("==" * 60)

    for i, files in mined_code.items():
        print(f"Generation {i} progs:")
        for f, g in files.items():
            for p in g:
                print("Prog")
                print("-" * 10)
                print(ast.unparse(p.idiom))
                print()
