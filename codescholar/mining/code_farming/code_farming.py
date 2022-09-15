import ast
from typing import List

from codescholar.utils.mining_utils import MinedIdiom, build_node_lookup
from codescholar.utils.logging import logger


def generate_simplified_ast(source_code: str):
    # NOTE: Move this into mining_utils?

    # TODO: Add simplification strategies. For e.g,
    # add a MetaVariable node as a parent for each variable
    # so that the mining algo can decide to chose a symbolic
    # or a concrete value for each node.

    return ast.parse(source_code)


def subgraph_matches(G: ast.AST, dataset: List[ast.AST]) -> int:
    """Find if subgraph G is a subgraph isomorphism
    in each graph H in the dataset.

    Args:
        G (ast.AST): a query python ast
        dataset (List[ast.AST]): a list of asts to search

    Returns:
        int: number of times G is found in dataset
    """

    return 10


def grow_idiom(idiom, prog):
    return idiom
    # pass


def save_idiom(mined_results, candidate_idiom, index):
    if index not in mined_results:
        mined_results[index] = [candidate_idiom]

    else:
        mined_results[index].append(candidate_idiom)

    return mined_results


def generic_mine_code(
    dataset: List,
    fix_max_len: bool = False,
    max_len: int = 0
) -> dict:
    gamma: float = 0.1 * len(dataset)
    node_count: int = 1
    mined_results: dict = {}

    mined_results[node_count] = dataset

    while (mined_results[node_count] is not None):
        # logger.info(f"{node_count}: {mined_results[node_count]}")

        for idiom in mined_results[node_count]:
            for prog in dataset:

                if idiom == prog:
                    continue

                # logger.info(f"I: {idiom} P: {prog}")
                candidate_idiom = grow_idiom(idiom, prog)

                if(
                    subgraph_matches(candidate_idiom)
                    >= gamma**(1 / node_count)
                ):
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

    test_dataset = [
        ast.parse('''with open(file) as fp:
            line = fp.read()'''),
        ast.parse('''inp = line.split()'''),
        ast.parse('''inp = [int(i) for i in line]''')
    ]

    mined_code = generic_mine_code(dataset=test_dataset,
                                   fix_max_len=True,
                                   max_len=3)
