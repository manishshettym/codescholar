import os
import ast

from tqdm import tqdm
import glob
from copy import deepcopy
from typing import List

from codescholar.utils.mining_utils import (
    MinedIdiom,
    get_ast_statements,
    build_node_lookup,
    build_subgraph)
from codescholar.utils.logs import logger


def build_dataset_lookup(dataset: List[ast.AST]):
    dataset_lookup = []
    for datapoint in dataset:
        try:
            lookup = build_node_lookup(datapoint)
            dataset_lookup.append(lookup)
        except:
            continue
    
    return dataset_lookup


def subgraph_matches(query, dataset_lookup) -> int:
    """Count number of times graph `query` is a subgraph isomorphism
    in any graph in `dataset`.

    Args:
        G (ast.AST): a query python ast
        dataset_lookup (List[]): a list of ast summaries to search
    """
    count = 0

    for lookup in dataset_lookup:
        try:
            result = build_subgraph(query, lookup)
            result = ast.unparse(result)

            if result == ast.unparse(query):
                count += 1

        except Exception:
            pass

    return count


def grow_idiom(idiom, prog):
    
    # print(f"Attempting: {type(idiom).__name__} + {type(prog).__name__}")
    idiom_copy = deepcopy(idiom)
    body_node = None

    # find last stmt in idiom with a body
    for i in ast.walk(idiom_copy):
        if 'body' in i._fields:
            body_node = i
    
    if body_node:
        try:
            body_node.body.append(prog.body)
        except:
            # print(prog, "has no body attr")
            body_node.body.append(prog)

        return idiom_copy

    return None


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

    gamma: float = 0.1 * len(dataset)
    node_count: int = 1
    mined_results: dict = {}

    statements = get_ast_statements(dataset)
    dataset_lookup = build_dataset_lookup(dataset)
    mined_results[node_count] = statements

    while (node_count in mined_results
            and mined_results[node_count] is not None):
        print(f"Generation {node_count} w/ gamma: {gamma}")

        for idiom in tqdm(mined_results[node_count]):
            for prog in tqdm(statements):

                if idiom == prog:
                    continue
                
                try:
                    candidate_idiom = grow_idiom(idiom, prog)
                except:
                    logger.trace(f"[Exception] Could not grow:\n I:\
                        {ast.unparse(idiom)}\nP:{ast.unparse(prog)}\n\n")
                    continue

                if(candidate_idiom is not None):
                    try:
                        match_count = subgraph_matches(candidate_idiom,
                                                       dataset_lookup)
                    except:
                        logger.trace(f"[Exception] Could not find:\n C:\
                            {ast.unparse(candidate_idiom)}\n\n")
                        continue
                    
                    if match_count > 0:
                        logger.trace(f"Match Count: {match_count}")

                    if match_count >= gamma**(1 / node_count):
                        logger.trace(f"C:\n{ast.unparse(candidate_idiom)}\n\n")

                        mined_results = save_idiom(mined_results,
                                                   candidate_idiom,
                                                   index=node_count + 1)
                else:
                    continue

        node_count += 1
        if fix_max_len:
            if node_count > max_len:
                break
            else:
                continue
        else:
            continue

    return mined_results


if __name__ == "__main__":
    dataset = []
    path = "../../data/Python-master"

    for filename in sorted(glob.glob(os.path.join(path, '*.py'))):
        with open(os.path.join(path, filename), 'r') as f:
            try:
                dataset.append(ast.parse(f.read()))
            except:
                pass
        
    mined_code = generic_mine_code(dataset, fix_max_len=True, max_len=3)

    for i, g in mined_code.items():
        print(f"iteration {i}")
        for p in g:
            print(ast.unparse(p))
            print()
