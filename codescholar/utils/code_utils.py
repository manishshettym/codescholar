import ast
import astunparse
import black
import six
import pygraphviz
import networkx as nx


def normalize_code_str(code: str) -> str:
    """
    Returns a formatting-normalized version of the code by running through
    a parser and then a code-generator.
    Args:
        code: A string corresponding to the code to normalize
    Returns:
        (str): The normalized code.
    """
    return normalize_code_ast(ast.parse(code))


def normalize_code_ast(code_ast: ast.AST) -> str:
    """
    Returns a formatting-normalized version of the provided Python AST by
    running through a code-generator.
    Args:
        code_ast: The Python AST to unparse.
    Returns:
        (str): A normalized code string.
    """
    mode = black.FileMode()
    result = black.format_str(astunparse.unparse(code_ast).strip(), mode=mode)
    return result.strip()


def program_graph_to_nx(program_graph, directed=False):
    """Converts a ProgramGraph to a NetworkX graph.
    Args:
        program_graph: A ProgramGraph.
        directed: Whether the graph should be treated as a directed graph.
    Returns:
        A NetworkX graph that can be analyzed by the networkx module.
    """
    # Create a graphviz representation
    graphviz_repr = program_graph_to_graphviz(program_graph)

    # translate to networkx
    if directed:
        return nx.DiGraph(graphviz_repr)
    else:
        return nx.Graph(graphviz_repr)


# note: this method is adapted from the python_graphs library
def program_graph_to_graphviz(graph):
    """Creates a graphviz representation of a ProgramGraph.

    Args:
    graph: A ProgramGraph object to translate.

    Returns:
    A pygraphviz object representing the ProgramGraph.
    """
    
    g = pygraphviz.AGraph(strict=False, directed=True)

    for _, node in graph.nodes.items():
        node_attrs = {}
        if node.ast_type:
            node_attrs['ast_type'] = six.ensure_str(node.ast_type, 'utf-8')
        if node.ast_value:
            node_attrs['value'] = six.ensure_str(node.ast_value, 'utf-8')
        if node.syntax:
            node_attrs['syntax'] = six.ensure_str(node.syntax, 'utf-8')

        g.add_node(node.id, **node_attrs)

    for edge in graph.edges:
        edge_attrs = {}
        # edge_attrs['label'] = edge.type.name

        g.add_edge(edge.id1, edge.id2, **edge_attrs)

    return g
