"""Graphviz visualizations of Program Graphs."""
import re
import pygraphviz
from python_graphs.program_graph import ProgramGraph
from python_graphs import program_graph_dataclasses as pb


def to_graphviz(graph: ProgramGraph, spans=False, relpos=False):
    """Creates a grapvhviz representation of a ProgramGraph.

    Args:
        graph: A ProgramGraph object to visualize.
    Returns:
        A pygraphviz object representing the ProgramGraph.
    """
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

    g = pygraphviz.AGraph(strict=False, directed=True)
    for _, node in graph.nodes.items():
        has_child = len([c for c in graph.children(node)])
        node_attrs = {}

        if node.ast_type:
            node_attrs['label'] = str(node.ast_type)
            if spans:
                node_attrs['label'] = str(node.span)
            if relpos:
                relpos_str = str(node.relpos).translate(SUB)
                node_attrs['label'] += relpos_str
            
            # remove formatting for the render
            node_attrs['label'] = re.sub('\s+', ' ', node_attrs['label'])

            if has_child:
                node_attrs['shape'] = 'box'

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


def render_sast(graph: ProgramGraph, path='/tmp/graph.png', spans=False, relpos=False):
    g = to_graphviz(graph, spans, relpos)
    g.draw(path, prog='dot')
