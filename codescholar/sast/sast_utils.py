import gast as ast
from python_graphs.program_graph import ProgramGraph
from python_graphs import program_graph_dataclasses as pb


class CodeSpan(ast.NodeTransformer):
    def __init__(self, source):
        self.source = source
        self.lines = source.split('\n')
    
    def _get_char_index(self, lineno, col_offset):
        line_index = lineno - 1
        line_start = sum(len(line) + 1 for line in self.lines[:line_index])
        return line_start + col_offset

    def _add_span(self, node):
        try:
            lineno = node.lineno
            end_lineno = node.end_lineno
            col_offset = node.col_offset
            end_col_offset = node.end_col_offset

            span_start = self._get_char_index(lineno, col_offset)
            span_end = self._get_char_index(end_lineno, end_col_offset)
            node.range = (span_start, span_end)
        except AttributeError:
            node.range = (0, 0)
        
        return node
    
    def visit(self, node):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        """Called if no explicit visitor function exists for a node."""
        self._add_span(node)

        for key, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        self._add_span(item)
                        self.visit(item)

            elif isinstance(value, ast.AST):
                self._add_span(value)
                self.visit(value)
        
        return node


def remove_node(sast: ProgramGraph, id):
    '''remove a node from the program graph'''
    edges_to_pop = []
    for edge in sast.edges:
        if edge.id1 == id or edge.id2 == id:
            edges_to_pop.append(edge)
    
    for edge in edges_to_pop:
        sast.remove_edge(edge)

    sast.nodes.pop(id)


def collapse_nodes(sast: ProgramGraph):
    '''collapse noisy nodes from the program graph to create
    a simplified AST'''
    nodes_to_pop = []
    for node in sast.all_nodes():
        parent = sast.parent(node)
        children = [c for c in sast.children(node)]

        if (not isinstance(node.ast_node, ast.Module)
                and node.ast_node.range == (0, 0)):
            for child in children:
                sast.add_new_edge(parent, child, pb.EdgeType.FIELD)
            
            nodes_to_pop.append(node.id)

        elif len(children) == 1:
            child = children[0]
            if node.ast_node.range == child.ast_node.range:
                sast.add_new_edge(parent, child, pb.EdgeType.FIELD)
                nodes_to_pop.append(node.id)

    for i in nodes_to_pop:
        remove_node(sast, i)

    return sast


def label_nodes(sast: ProgramGraph, source: str):
    '''label nodes for the simplified AST'''
    for node in sast.all_nodes():
        if isinstance(node.ast_node, ast.Module):
            setattr(node, 'span', '#')
            continue
            
        children = [c for c in sast.children(node)]
        children = sorted(children, key=lambda node: node.ast_node.range[0])

        l, r = node.ast_node.range
        span = source[l: r]
        offset = l
        
        for c in children:
            c_l, c_r = c.ast_node.range
            c_len = c_r - c_l
            c_l -= offset
            c_r = c_l + c_len

            span = span[:c_l] + '#' + span[c_r:]
            offset += (c_r - c_l) - 1
        
        setattr(node, 'span', span)

    return sast
