import os
import sys
import ast
import glob
import attrs

from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Tuple, Set

from codescholar.utils import multiprocess
from codescholar.utils.mining_utils import MinedIdiom, pprint_mine, save_idiom
from codescholar.mining.code_farming.subgraphs import (build_dataset_lookup,
                                                       subgraph_matches)

MAX_WORKERS = 2


def _mp_subgraph_matches(args):
    query, dataset_lookup = args
    return subgraph_matches(query, dataset_lookup)


def _mp_code_miner(args):
    mined_results, index, dataset_lookup, gamma = args
    return mine_file(mined_results, index,
                     dataset_lookup, gamma)


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


def grow_idiom(idiom: ast.AST, prog: ast.AST):
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

    return new_idiom


def mine_file(
    mined_results: Dict[int, Dict[int, List]],
    index: Tuple[int, int],
    dataset_lookup: Dict,
    gamma: int
) -> Dict[int, Dict[int, List]]:

    ncount, fileid = index

    for idiom in mined_results[ncount][fileid]:
        candidates: List[Tuple(ast.AST, Dict[str, List])] = []
        candidates_loc: List[Tuple(int, int)] = []

        # pass 1: create candidate idioms by combining w/ single nodes
        for prog in mined_results[1][fileid]:
            candidate_idiom = None

            # don't grow unnatural sequences
            if prog.end <= idiom.end:
                continue

            try:
                candidate_idiom = grow_idiom(idiom.idiom, prog.idiom)
            except:
                continue
            finally:
                if candidate_idiom is not None:
                    candidates.append((candidate_idiom, dataset_lookup))
                    candidates_loc.append((idiom.start, prog.end))

        # pass 2: prune candidate idioms based on frequency
        for (c, dataset_lookup), loc in zip(candidates, candidates_loc):
            result = subgraph_matches(c, dataset_lookup)
            # print(f"C ({result}): {ast.unparse(c)}")

            if result >= gamma**(1 / ncount):
                index = (ncount + 1, fileid)
                mined_results = save_idiom(mined_results,
                                           c, loc, index)
            else:
                continue
    
    return mined_results


def codescholar_codefarmer(
    dataset: List[ast.AST],
    gamma: float,
    fix_max_len: bool = False,
    max_len: int = 0
) -> Dict[int, Dict[int, List]]:

    dataset, dataset_lookup = build_dataset_lookup(dataset)
    gamma = gamma * len(dataset)
    ncount: int = 1

    mined_results: dict = {}
    mined_results[1] = {}

    # ================== INIT SINGLE NODES ==================

    for fileid, prog in enumerate(tqdm(dataset)):
        mined_results[1][fileid] = get_single_nodes([prog],
                                                    dataset_lookup,
                                                    gamma)
    pprint_mine(mined_results, ncount, gamma)

    # ===================== MINING LOOP =====================

    while (ncount in mined_results):
        
        file_ids = mined_results[ncount].keys()

        if MAX_WORKERS > 1:
            codefarmer_tasks = [
                (mined_results, (ncount, fileid), dataset_lookup, gamma)
                for fileid in file_ids
            ]

            miner_mp_iter = multiprocess.run_tasks_in_parallel_iter(
                _mp_code_miner,
                tasks=codefarmer_tasks,
                use_progress_bar=False,
                num_workers=MAX_WORKERS)

            for fileid, result in tqdm(zip(file_ids, miner_mp_iter),
                                       total=len(file_ids)):
                if (result.is_success() and isinstance(result.result, Dict)):
                    mined_results = result.result
        else:
            for fileid in file_ids:
                index = (ncount, fileid)
                mined_results = mine_file(mined_results, index,
                                          dataset_lookup, gamma)

        ncount += 1

        if ncount in mined_results:
            pprint_mine(mined_results, ncount, gamma)

        if((fix_max_len and ncount > max_len)
                or (gamma**(1 / ncount) < 1)):
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
    
    mined_code = codescholar_codefarmer(dataset[:15], gamma=0.1,
                                        fix_max_len=True, max_len=5)

    # ================== CREATE MINING CAMPAIGN SUMMARY ==================

    print("==" * 20 + " [CodeScholar::Concept Miner Summary] " + "==" * 20)
    print(f"Dataset: {len(dataset)} progs")
    print(f"# Explorations: {len(mined_code)}")
    print("==" * 60)
