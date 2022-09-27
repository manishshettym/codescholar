import ast
from typing import List
import collections
import attrs

from codescholar.utils.logs import logger


@attrs.define(eq=False, repr=False)
class MinedIdiom:
    idiom: ast.AST
    start: int
    end: int


def pprint_mine(code_mine, index, gamma):
    """ pretty print a summary of the current generation
    of mined idioms.
    """
    delim = "==" * 20
    thresh = round(gamma**(1 / index), 3)

    print(f"{delim} [CodeScholar::Gen({index}) (\u03BB = {thresh})] {delim}")

    for _, g in code_mine[index].items():
        for p in g:
            print(ast.unparse(p.idiom))
            print("-" * 10 + "\n")


def save_idioms(code_mine, mined_idioms):
    for idiom, loc, (ncount, fileid) in mined_idioms:
        new_idiom = MinedIdiom(idiom, loc[0], loc[1])

        if ncount not in code_mine:
            code_mine[ncount] = {}
            code_mine[ncount][fileid] = [new_idiom]

        elif fileid not in code_mine[ncount]:
            code_mine[ncount][fileid] = [new_idiom]

        else:
            code_mine[ncount][fileid].append(new_idiom)

    return code_mine


def find_common_ancestors(node_matches, anc):
    """check if any of the matched nodes have same
    ancestors.
    """
    # TODO: This method is the comparator for subgraph matching. It performs
    # an approx match by traversing of node type hierarchy. This should
    # be replaced by a better and faster approximation.
    have_common_ancestors = False

    if node_matches is None:
        return have_common_ancestors

    for i, val in enumerate(node_matches):
        _, match_anc = val

        if anc == match_anc[-1 * len(anc):] or not anc:
            have_common_ancestors = True
            break
    
    # remove matched node from lookup
    if have_common_ancestors:
        node_matches.pop(i)

    return have_common_ancestors


def build_subgraph(node: ast.AST, lookup: dict, anc: List[str] = []):
    """_summary_

    Args:
        node (ast.AST): ast node to start with
        lookup_table (dict): map of node_type -> [(node, [path])]
        anc (List[str]): ancestor path from root to node. Defaults to [].

    Returns:
        ast.AST: a subgraph if present in lookup_table
        starting at node.
    """

    node_summary = get_node_summary(node)
    ntype = type(node).__name__
    hash = str(node_summary)

    if hash in lookup:
        
        node_matches = lookup[hash]
        any_common_ancestral_path = find_common_ancestors(node_matches, anc)

        if any_common_ancestral_path:
            subgraphs_at_node = {}
            
            if node_summary[2] != []:

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
                        subgraphs_at_node[i] = subgraphs
                    
                    # elif child is a node
                    elif (result := build_subgraph(
                            child, lookup,
                            anc + [ntype])) is not None:
                            
                        # add the subgraph
                        subgraphs_at_node[i] = result
            
            new_node = type(node)(
                **{i : getattr(node, i) for i in node_summary[1]},
                **subgraphs_at_node,
                **{i : getattr(node, i, 0) for i in type(node)._attributes})

            return new_node

    return None

      
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
