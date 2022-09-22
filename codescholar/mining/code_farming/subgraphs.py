import ast

from copy import deepcopy
from typing import Dict, List, Tuple, Set

from codescholar.utils.mining_utils import build_subgraph, build_node_lookup


def build_dataset_lookup(dataset: List[ast.AST]):
    dataset_lookup = []
    clean_data = []
    for datapoint in dataset:
        try:
            lookup = build_node_lookup(datapoint)
            dataset_lookup.append(lookup)
            clean_data.append(datapoint)
        except:
            continue
    
    return clean_data, dataset_lookup


def subgraph_matches(query: ast.AST, dataset_lookup: Dict[str, List]) -> int:
    """Count number of times graph `query` is a subgraph isomorphism
    in any graph in `dataset`.

    Args:
        G (ast.AST): a query python ast
        dataset_lookup (List[]): a list of ast summaries to search
    """
    
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
