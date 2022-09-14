import ast
from typing import List
import collections
import attrs


@attrs.define(eq=False, repr=False)
class MinedIdiom:
    code: str


def walk_with_ancestors(prog: ast.AST, p: List[str] = []):
    """retrieve all ast nodes of a tree with ancestor hierarchy

    Args:
        tree (ast.AST): python ast to walk
        p (list, optional): list of ancestors of a walked node. Defaults to [].

    Yields:
        ast.AST: each node walked + ancestor
    """
    yield prog, p
    
    for i in prog._fields:
        v = getattr(prog, i)

        if isinstance(v, list):
            for j in v:
                yield from walk_with_ancestors(j, p + [type(prog).__name__])

        elif isinstance(v, ast.AST):
            yield from walk_with_ancestors(v, p + [type(prog).__name__])


def node_attr(node: ast.AST):
    """Get all attributes of a python ast node

    Args:
        node (ast.AST): ast node to analyze
    """
    t = type(node)
    node_type = t.__name__

    child_nodes_lists = [
        i for i in t._fields
        if isinstance(getattr(node, i), (ast.AST, list))
    ]

    child_nodes = [
        a for a in t._fields
        if not isinstance(getattr(node, a), (ast.AST, list))
    ]

    return node_type, child_nodes_lists, child_nodes

  
def build_node_lookup(node: ast.AST):
    """create a lookup for every node type in a python ast mapping
    (type, list(children), children) -> [(node, [ancestors])]

    Args:
        node (ast.AST): _description_
    """
    lookup_table = collections.defaultdict(list)

    for sub_node, ancestors in walk_with_ancestors(node):
        hash = str(node_attr(sub_node))
        lookup_table[hash].append((sub_node, ancestors))
            
    return lookup_table


if __name__ == "__main__":

    s1 = open("../experiments/basic.py").read()
    
    prog = ast.parse(s1)
    lookup_s1 = build_node_lookup(prog)

    for k, v in lookup_s1.items():
        print(k, v)
