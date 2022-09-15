import ast
from curses import resetty
from distutils.command.build import build
import re
from typing import List
import collections
import attrs


@attrs.define(eq=False, repr=False)
class MinedIdiom:
    code: str


def build_subgraph(
    node: ast.AST,
    lookup: dict,
    anc: List[str] = []
) -> ast.AST:

    node_summary = get_node_summary(node)
    ntype = type(node).__name__
    hash = str(node_summary)

    if hash in lookup:
        # find the query node (hash) in database node (lookup)
        node_matches = lookup[hash]

        any_common_ancestral_path = any(
            anc == match_anc[-1 * len(anc):] or not anc
            for _, match_anc in node_matches
        )

        if any_common_ancestral_path:
            subgraphs_at_node = {}

            # loop over children that are lists
            for i in node_summary[2]:
                child = getattr(node, i)

                # if child is also a list
                if isinstance(child, list):
                    subgraphs = []
                    
                    # loop over grandchildren & recurse
                    for j in child:
                        result = build_subgraph(j, lookup, anc + [ntype])

                        if result is not None:
                            subgraphs.append(result)
                    
                    # add all found subgraphs
                    subgraphs_at_node[i] = subgraphs_at_node
                
                # elif child is a node
                elif result := build_subgraph(
                        child, lookup,
                        anc + [ntype]) is not None:
                        
                    # add the subgraph
                    subgraphs_at_node[i] = result
            
            return type(node)(
                **{i : getattr(node, i) for i in node_summary[1]},
                **subgraphs_at_node,
                **{i : getattr(node, i, 0) for i in type(node)._attributes})

    return None
 
      
# verified
def walk_with_ancestors(prog: ast.AST, ancestors: List[str] = []):
    """retrieve all ast nodes of a tree with ancestor hierarchy

    Args:
        tree (ast.AST): python ast to walk
        p (list, optional): list of ancestors of a walked node. Defaults to [].

    Yields:
        ast.AST: each node walked + ancestor
    """
    yield prog, ancestors
    
    for i in prog._fields:
        v = getattr(prog, i)

        if isinstance(v, list):
            for j in v:
                yield from walk_with_ancestors(
                    j, ancestors + [type(prog).__name__])

        elif isinstance(v, ast.AST):
            yield from walk_with_ancestors(
                v, ancestors + [type(prog).__name__])


# verified
def get_node_summary(node: ast.AST):
    """Get all attributes of a ast node:

    Args:
        node (ast.AST): ast node to analyze
    """
    t = type(node)
    node_type = t.__name__

    other_children = [
        a for a in t._fields
        if not isinstance(getattr(node, a), (ast.AST, list))
    ]

    list_children = [
        i for i in t._fields
        if isinstance(getattr(node, i), (ast.AST, list))
    ]

    return node_type, other_children, list_children


# verified
def build_node_lookup(node: ast.AST):
    """create a lookup for every node type in a python ast mapping
    (type, list(children), children) -> [(node, [ancestors])]

    Args:
        node (ast.AST): _description_
    """
    lookup_table = collections.defaultdict(list)

    for child, ancestors in walk_with_ancestors(node):
        hash = str(get_node_summary(child))
        lookup_table[hash].append((child, ancestors))
            
    return lookup_table


if __name__ == "__main__":

    data = open("../experiments/basic_idiom.py").read()
    data_prog = ast.parse(data)
    lookup = build_node_lookup(data_prog)

    query = open("../experiments/basic.py").read()
    query_prog = ast.parse(query)

    result = build_subgraph(query_prog, lookup)
    try:
        result = ast.unparse(result)
        print(result)
    except:
        print("error")
        pass

    # for k, v in lookup_query.items():
    #     print(k, v)
