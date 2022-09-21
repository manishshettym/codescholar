import os
import ast
import glob
import random

from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Tuple, Set

from codescholar.utils.mining_utils import (build_node_lookup, build_subgraph)
from codescholar.utils.logs import logger
from codescholar.utils import multiprocess

MAX_WORKERS = 100


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
    print(f"Generation 0 w/ [gamma]: {gamma}")

    candidates: List[Tuple(ast.AST, Dict[str, List])] = []

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
    
    # TODO: REMOVE THIS WHEN RUNNING FULL CAMPAIGN
    # random.seed(55)
    # candidates = random.sample(candidates, k=100)

    subgraph_mp_iter = multiprocess.run_tasks_in_parallel_iter(
        subgraph_matches,
        tasks=candidates,
        use_progress_bar=True,
        num_workers=MAX_WORKERS)

    for c, result in zip(candidates, subgraph_mp_iter):

        if (
            result.is_success()
            and isinstance(result.result, int)
            and result.result >= gamma
        ):
            stmts.append(c[0])
    
    return set(stmts)


def build_dataset_lookup(dataset: List[ast.AST]):
    dataset_lookup = []
    clean_data = []
    for i, datapoint in enumerate(dataset):
        try:
            lookup = build_node_lookup(datapoint)
            dataset_lookup.append(lookup)
            clean_data.append(datapoint)
        except:
            continue
    
    return clean_data, dataset_lookup


def subgraph_matches(args: Tuple[ast.AST, Dict[str, List]]) -> int:
    """Count number of times graph `query` is a subgraph isomorphism
    in any graph in `dataset`.

    Args:
        G (ast.AST): a query python ast
        dataset_lookup (List[]): a list of ast summaries to search
    """
    
    query = args[0]
    dataset_lookup = args[1]
    count = 0
    query_prog = ast.unparse(query)

    for lookup in dataset_lookup:
        try:
            result = build_subgraph(query, deepcopy(lookup))
            result = ast.unparse(result)

            if result == query_prog:
                count += 1

        except Exception:
            pass

    return count


def grow_idiom(idiom, prog):
    
    # print(f"Attempting: {type(idiom).__name__} + {type(prog).__name__}")
    idiom_copy = deepcopy(idiom)
    prog_copy = deepcopy(prog)
    new_idiom = None

    if isinstance(idiom_copy, ast.Module):
        if isinstance(prog_copy, ast.Module):
            new_idiom = ast.Module([idiom_copy.body, prog_copy.body], [])
        else:
            new_idiom = ast.Module([idiom_copy.body, prog_copy], [])
    else:
        if isinstance(prog_copy, ast.Module):
            new_idiom = ast.Module([idiom_copy, prog_copy.body], [])
        else:
            new_idiom = ast.Module([idiom_copy, prog_copy], [])

    # print(f"After merging:\n{ast.unparse(new_idiom)}\n\n\n")
    return new_idiom


def save_idiom(mined_results, candidate_idiom, index):
    if index not in mined_results:
        mined_results[index] = [candidate_idiom]

    else:
        mined_results[index].append(candidate_idiom)

    return mined_results


def generic_mine_code(
    dataset: List[ast.AST],
    fix_max_len: bool = False,
    max_len: int = 0
) -> dict:
    dataset, dataset_lookup = build_dataset_lookup(dataset)
    gamma: float = 0.05 * len(dataset)
    node_count: int = 1
    mined_results: dict = {}

    statements = get_single_nodes(dataset, dataset_lookup, gamma)
    mined_results[node_count] = statements

    while (node_count in mined_results):
        print(f"Generation {node_count} w/ [gamma]: {gamma**(1 / node_count)}")

        for idiom in tqdm(mined_results[node_count]):
            candidates = []

            # pass 1: create candidate idioms by combining graphs
            for prog in statements:
                candidate_idiom = None

                if idiom == prog:
                    continue

                try:
                    candidate_idiom = grow_idiom(idiom, prog)
                except:
                    logger.trace(f'''[Exception] Could not grow:\n I:\
                        {ast.unparse(idiom)}\nP:{ast.unparse(prog)}\n\n''')
                    continue
                finally:
                    if candidate_idiom is not None:
                        candidates.append((candidate_idiom, dataset_lookup))

            # define a multiprocess worker
            subgraph_mp_iter = multiprocess.run_tasks_in_parallel_iter(
                subgraph_matches,
                tasks=candidates,
                use_progress_bar=False,
                num_workers=MAX_WORKERS)

            # pass 2: prune candidate idioms based on frequency
            for (c, _), result in zip(candidates, subgraph_mp_iter):
                if (
                    result.is_success()
                    and isinstance(result.result, int)
                    and result.result >= gamma**(1 / node_count)
                ):

                    # print(f"C:\n{ast.unparse(c)}\n\n")
                    mined_results = save_idiom(mined_results, c,
                                               node_count + 1)

                else:
                    continue

        node_count += 1
        if fix_max_len and node_count > max_len:
            break

    return mined_results


if __name__ == "__main__":
    dataset = []
    # path = "../../data/Python-master"
    path = "../../data/examples"

    for filename in sorted(glob.glob(os.path.join(path, '*.py'))):
        with open(os.path.join(path, filename), 'r') as f:
            try:
                dataset.append(ast.parse(f.read()))
            except:
                pass
    
    mined_code = generic_mine_code(dataset, fix_max_len=True, max_len=3)

    # ******************* CREATE MINING CAMPAIGN SUMMARY *******************

    print("==" * 20 + " [[CodeScholar::Concept Miner Summary]] " + "==" * 20)
    print(f"Dataset: {len(dataset)} progs")
    print(f"# Explorations: {len(mined_code)}")
    print("==" * 60)

    for i, g in mined_code.items():
        print(f"Generation {i}: {len(g)} progs")
        for p in g:
            print(ast.unparse(p))
            print()
