import re
import gast as ast
import io
import tokenize
from collections import defaultdict
import textwrap

from python_graphs.program_graph import ProgramGraph
from python_graphs import program_graph_dataclasses as pb


class DropDecorators(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        new_node = node
        new_node.decorator_list = []

        ast.copy_location(new_node, node)
        ast.fix_missing_locations(new_node)
        self.generic_visit(node)

        return new_node


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

            assert span_start >= 0 and span_start <= len(self.source)
            assert span_end >= 0 and span_end <= len(self.source)

        except (AttributeError, AssertionError, TypeError) as e:
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
    
    # remove the edges
    for edge in edges_to_pop:
        sast.edges.remove(edge)
        n1, n2 = edge.id1, edge.id2

        if n1 in sast.child_map:
            try:
                sast.child_map[n1].remove(n2)
            except:
                pass

        if n2 in sast.parent_map:
            del sast.parent_map[n2]

        if n1 in sast.neighbors_map:
            try:
                sast.neighbors_map[n1].remove((edge, n2))
            except:
                pass
        if n2 in sast.neighbors_map:
            try:
                sast.neighbors_map[n2].remove((edge, edge.id1))
            except:
                pass

    # pop the node
    sast.nodes.pop(id)


def filter_non_ast(sast: ProgramGraph):
    nodes_to_pop = []
    for node in sast.all_nodes():
        if node.ast_node is None:
            nodes_to_pop.append(node.id)
    
    for i in nodes_to_pop:
        remove_node(sast, i)
    
    return sast


def collapse_nodes(sast: ProgramGraph):
    '''collapse noisy nodes from the program graph to create
    a simplified AST'''

    sast = filter_non_ast(sast)

    nodes_to_pop = []
    for node in sast.all_nodes():
        parent = sast.parent(node)
        children = [c for c in sast.children(node)]

        # if range of the node is 0:0
        if (not isinstance(node.ast_node, ast.Module)
                and node.ast_node.range == (0, 0)):
            for child in children:
                sast.add_new_edge(parent, child, pb.EdgeType.FIELD)
            
            nodes_to_pop.append(node.id)
        
        # if empty function
        elif isinstance(node.ast_node, ast.Module):
            child = children[0]
            if child.ast_node.range == (0, 0):
                return None
        
        # if only 1 child w/ same range; e.g. Expr-->Call
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

        if not hasattr(node, 'relpos'):
            setattr(node, 'relpos', 0)

        if isinstance(node.ast_node, ast.Module):
            setattr(node, 'span', '#')
            continue
            
        children = [c for c in sast.children(node)]
        children = sorted(children, key=lambda node: node.ast_node.range[0])

        l, r = node.ast_node.range
        span = source[l: r]
        offset = l
        
        for c_id, c in enumerate(children):
            setattr(c, 'relpos', c_id)
            c_l, c_r = c.ast_node.range
            c_len = c_r - c_l
            c_l -= offset
            c_r = c_l + c_len

            span = span[:c_l] + '#' + span[c_r:]
            offset += (c_r - c_l) - 1
        
        span = re.sub('\s+', ' ', span)
        setattr(node, 'span', span)

    return sast


def add_indent_to_block(prog_str: str):
    return textwrap.indent(prog_str, 1 * '\t')


def kth_substr_idx(s: str, sub: str, k):
    where = [m.start() for m in re.finditer(sub, s)]

    if len(where) < k - 1:
        return -1
    else:
        return where[k - 1]


# NOTE @manishs: migth be incomplete and hacky. Need to
# clean it up in the future and simplify transpilation between
# source and sast.
def replace_nonterminals(node, child_spans):
    '''replace nonterminals in a node's span'''

    ins = 0
    dels = 0

    module_flag = False
    if isinstance(node.ast_node, ast.Module):
        module_flag = True

    child_spans = sorted(child_spans, key=lambda x: x[1])
    new_span = node.span if not module_flag else ""

    if ' else' in new_span:
        new_span = new_span.replace(' else', '\nelse')

    for span, span_idx in child_spans:
        elif_flag = span.startswith('elif')
        else_flag = 'else' in span
        span_idx = span_idx - dels + ins

        if module_flag:
            new_span += span + "\n"
        else:

            col_loc = None
            loc = kth_substr_idx(new_span, '#', k=span_idx + 1)

            if ':' in new_span:
                col_loc = new_span.index(':')
            
            # update the child_span
            if elif_flag:
                span = f"\n{span}"
                if else_flag:
                    span = span.replace(' else', '\nelse')
            
            elif col_loc and loc > col_loc:
                span = f"\n{add_indent_to_block(span)}"

            new_span = new_span[:loc] + new_span[loc:].replace('#', span, 1)
            
            dels += 1
            ins += len([m for m in re.finditer('#', span)])

    node.span = new_span
    return node


def sast_to_prog(sast: ProgramGraph):
    '''perform an dfs traversal and regenerate prog'''

    def dfs_util(sast: ProgramGraph, node, visited):
        # print(f"VISITING NODE {node.span}")
        visited[node.id] = True
        span_pos = []
        
        for child in sast.children(node):
            # print(f"- Child {child.span}")
            if not visited[child.id]:
                span, pos = dfs_util(sast, child, visited)
                span_pos.append((span, pos))
            else:
                span_pos.append((child.span, child.relpos))
        
        # print(f"before: {node.span}")
        # print(f"replacements: {span_pos}")
        node = replace_nonterminals(node, span_pos)
        # print(f"after: {node.span}\n")
        
        return node.span, node.relpos

    visited = defaultdict()
    for node in sast.all_nodes():
        visited[node.id] = False

    for node in sast.all_nodes():
        if not visited[node.id]:
            dfs_util(sast, node, visited)
    
    return sast.root.span


def remove_comments_and_docstrings(source: str):
    '''Remove comments and docstrings from a python file'''
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0

    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))

        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING:
            if prev_toktype != tokenize.INDENT:
                if prev_toktype != tokenize.NEWLINE:
                    if start_col > 0:
                        out += token_string
        else:
            out += token_string

        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line

    out = '\n'.join(line for line in out.splitlines() if line.strip())

    return out
