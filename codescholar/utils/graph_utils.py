import enum
import six
import json

import pygraphviz
import networkx as nx
from networkx.readwrite import json_graph

# from python_graphs import program_graph_graphviz


class GraphEdgeLabel(enum.Enum):
    """The different kinds of edges that can appear in a program graph."""
    UNSPECIFIED = 0
    CFG_NEXT = 1
    LAST_READ = 2
    LAST_WRITE = 3
    COMPUTED_FROM = 4
    RETURNS_TO = 5
    FORMAL_ARG_NAME = 6
    FIELD = 7
    SYNTAX = 8
    NEXT_SYNTAX = 9
    LAST_LEXICAL_USE = 10
    CALLS = 11


class GraphNodeLabel(enum.Enum):
    """The different kinds of nodes that can appear in a program graph"""
    Other = 0
    Module = 1
    Interactive = 2
    Expression = 3
    FunctionDef = 4
    ClassDef = 5
    Return = 6
    Delete = 7
    Assign = 8
    AugAssign = 9
    Print = 10
    For = 11
    While = 12
    If = 13
    With = 14
    Raise = 15
    TryExcept = 16
    TryFinally = 17
    Assert = 18
    Import = 19
    ImportFrom = 20
    Exec = 21
    Global = 22
    Expr = 23
    Pass = 24
    Break = 25
    Continue = 26
    attributes = 27
    BoolOp = 28
    BinOp = 29
    UnaryOp = 30
    Lambda = 31
    IfExp = 32
    Dict = 33
    Set = 34
    ListComp = 35
    SetComp = 36
    DictComp = 37
    GeneratorExp = 38
    Yield = 39
    Compare = 40
    Call = 41
    Repr = 42
    Num = 43
    Str = 44
    Attribute = 45
    Subscript = 46
    Name = 47
    List = 48
    Tuple = 49
    Load = 50
    Store = 51
    Del = 52
    AugLoad = 53
    AugStore = 54
    Param = 55
    Ellipsis = 56
    Slice = 57
    ExtSlice = 58
    Index = 59
    And = 60
    Or = 61
    Add = 62
    Sub = 63
    Mult = 64
    Div = 65
    Mod = 66
    Pow = 67
    LShift = 68
    RShift = 69
    BitOr = 70
    BitXor = 71
    BitAnd = 72
    FloorDiv = 73
    Invert = 74
    Not = 75
    UAdd = 76
    USub = 77
    Eq = 78
    NotEq = 79
    Lt = 80
    LtE = 81
    Gt = 82
    GtE = 83
    Is = 84
    IsNot = 85
    In = 86
    NotIn = 87
    comprehension = 88
    ExceptHandler = 89
    arguments = 90
    keyword = 91
    alias = 92


def program_graph_to_nx(program_graph, directed=False):
    """Converts a ProgramGraph to a NetworkX graph.
    Args:
        program_graph: A ProgramGraph.
        directed: Whether the graph should be treated as a directed graph.
    Returns:
        A NetworkX graph that can be analyzed by the networkx module.
    """
    # Debug: render the program graph
    # program_graph_graphviz.render(program_graph,
    #   path="../representation/plots/temp.png")

    # create custom graphviz representation
    graphviz_repr = program_graph_to_graphviz(program_graph)

    # translate graphviz to networkx
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
        assert node.ast_type
        assert node.span

        node_attrs = {'ast_type': 'Other'}
        node_attrs['ast_type'] = six.ensure_str(node.ast_type, 'utf-8')
        node_attrs['span'] = six.ensure_str(node.span, 'utf-8')
        g.add_node(node.id, **node_attrs)

    for edge in graph.edges:
        edge_attrs = {}
        edge_attrs['flow_type'] = six.ensure_str(edge.type.name, 'utf-8')
        g.add_edge(edge.id1, edge.id2, **edge_attrs)

    return g


def save_as_json(data, path):
    graph = json_graph.adjacency_data(data)
    with open(path, 'w') as f:
        json.dump(graph, f)
