import os
import sys
import ast
import glob
import attrs

from math import floor
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List, Tuple, Set

from codescholar.utils import multiprocess
from codescholar.utils.mining_utils import MinedIdiom, pprint_mine, save_idioms
from codescholar.mining.code_farming.subgraphs import (build_dataset_lookup,
                                                       subgraph_matches)

MAX_WORKERS = 1


def _mp_subgraph_matches(args):
    """multiprocessing util for subgraph_matches"""
    query, dataset_lookup = args
    return subgraph_matches(query, dataset_lookup)


def _mp_mine_file(args):
    """multiprocessing utility for mine_file
    """
    code_mine, index, dataset_lookup, gamma = args
    return mine_file(code_mine, index,
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
    
    if MAX_WORKERS > 1:
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
    
    else:
        for c, loc in zip(candidates, candidate_loc):
            if (subgraph_matches(c[0], dataset_lookup) >= gamma):
                stmts.append(MinedIdiom(c[0], loc[0], loc[1]))

    return stmts


def grow_idiom(idiom: ast.AST, prog: ast.AST):
    """combine two ASTs -- idiom and prog -- to generate
    a new candidate idiom.
    """
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
    code_mine: Dict[int, Dict[int, List]],
    index: Tuple[int, int],
    dataset_lookup: Dict,
    gamma: int
) -> Dict[int, Dict[int, List]]:
    """Run the mining loop for a single python file and generate
    all idioms in the current generation
    """
    ncount, fileid = index
    mined_idioms = []

    for idiom in code_mine[ncount][fileid]:
        candidates: List[Tuple(ast.AST, Dict[str, List])] = []
        candidates_loc: List[Tuple(int, int)] = []

        # pass 1: create candidate idioms by combining w/ single nodes
        for prog in code_mine[1][fileid]:
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
                mined_idioms.append((c, loc, index))
            else:
                continue
        
    return mined_idioms


def codescholar_codefarmer(
    dataset: List[ast.AST],
    min_freq: float,
    fix_max_len: bool = False,
    max_len: int = 0
) -> Dict[int, Dict[int, List]]:
    """Grow program graphs from single nodes (stmts) to idioms
    as large as function definitions.

    Args:
        dataset (List[ast.AST]): a list of programs converted to ASTs
        min_freq (float): minimum freq threshold to start with
        fix_max_len (bool, optional): should use max depth = max_len.
        max_len (int, optional): max depth for growing programs.

    Returns:
        Dict[int, Dict[int, List]]: #generation -> {#file -> [<prog>]}
    """

    dataset, dataset_lookup = build_dataset_lookup(dataset)
    gamma = floor(min_freq * len(dataset))
    ncount: int = 1

    code_mine: dict = {}
    code_mine[1] = {}

    # ================== INIT SINGLE NODES ==================

    for fileid, prog in enumerate(tqdm(dataset)):
        code_mine[1][fileid] = get_single_nodes([prog],
                                                dataset_lookup,
                                                gamma)
    pprint_mine(code_mine, ncount, gamma)

    # ===================== MINING LOOP =====================

    while (ncount in code_mine):
        
        file_ids = code_mine[ncount].keys()
        mined_idioms = []

        if MAX_WORKERS > 1:
            codefarmer_tasks = [
                (code_mine, (ncount, fileid), dataset_lookup, gamma)
                for fileid in file_ids
            ]

            miner_mp_iter = multiprocess.run_tasks_in_parallel_iter(
                _mp_mine_file,
                tasks=codefarmer_tasks,
                use_progress_bar=False,
                num_workers=MAX_WORKERS)

            for fileid, result in tqdm(zip(file_ids, miner_mp_iter),
                                       total=len(file_ids)):
                if (result.is_success() and isinstance(result.result, List)):
                    mined_idioms += result.result
        else:
            for fileid in file_ids:
                index = (ncount, fileid)
                mined_idioms += mine_file(code_mine, index,
                                          dataset_lookup, gamma)
        
        save_idioms(code_mine, mined_idioms)
        ncount += 1

        if ncount in code_mine:
            pprint_mine(code_mine, ncount, gamma)

        if((fix_max_len and ncount > max_len)
                or (gamma**(1 / ncount) < 1)):
            break

    return code_mine


if __name__ == "__main__":
    dataset = []
    path = "../../data/examples"

    for filename in sorted(glob.glob(os.path.join(path, '*.py'))):
        with open(os.path.join(path, filename), 'r') as f:
            try:
                dataset.append(ast.parse(f.read()))
            except:
                pass
    
    mined_code = codescholar_codefarmer(dataset, min_freq=0.3,
                                        fix_max_len=True, max_len=5)

    # ================== CREATE MINING CAMPAIGN SUMMARY ==================

    print("==" * 20 + " [CodeScholar::Concept Miner Summary] " + "==" * 20)
    print(f"Dataset: {len(dataset)} progs")
    print(f"# Explorations: {len(mined_code)}")
    print("==" * 60)
