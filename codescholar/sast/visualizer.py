"""Graphviz visualizations of SAST."""

import pygraphviz
from python_graphs import program_graph_dataclasses as pb


def to_graphviz(graph, spans=False):
    """Creates a graphviz representation of a SAST.

    Args:
        graph: A ProgramGraph object to visualize.
    Returns:
        A pygraphviz object representing the ProgramGraph.
    """
    g = pygraphviz.AGraph(strict=False, directed=True)
    for _, node in graph.nodes.items():
        node_attrs = {}
        if node.ast_type:
            node_attrs['label'] = str(node.ast_type)
            if spans:
                node_attrs['label'] = str(node.span)
        else:
            node_attrs['shape'] = 'point'

        node_type_colors = {}
        if node.node_type in node_type_colors:
            node_attrs['color'] = node_type_colors[node.node_type]
            node_attrs['colorscheme'] = 'svg'

        g.add_node(node.id, **node_attrs)

    for edge in graph.edges:
        edge_attrs = {}
        edge_attrs['label'] = edge.type.name
        edge_colors = {
            pb.EdgeType.LAST_READ: 'red',
            pb.EdgeType.LAST_WRITE: 'blue',
            pb.EdgeType.COMPUTED_FROM: 'green'
        }
        if edge.type in edge_colors:
            edge_attrs['color'] = edge_colors[edge.type]
            edge_attrs['colorscheme'] = 'svg'
        g.add_edge(edge.id1, edge.id2, **edge_attrs)

    return g


def render_sast(graph, path='/tmp/graph.png', spans=False):
    g = to_graphviz(graph, spans)
    g.draw(path, prog='dot')
