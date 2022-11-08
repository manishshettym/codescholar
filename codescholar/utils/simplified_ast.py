from python_graphs.program_graph import ProgramGraph
from python_graphs.program_graph import *
from python_graphs import program_utils, control_flow
from codescholar.utils.code_utils import CodeSpan
import gast as ast


AST_NODE_FILTER = (ast.Load, ast.Store)


def get_simplified_ast(source, program, dfg=True, cfg=True):
    """Constructs a simplified program graph to represent the given function"""
    program_node = program_utils.program_to_ast(program)
    program_node = CodeSpan(source).visit(program)

    program_graph = ProgramGraph()
    control_flow_graph = control_flow.get_control_flow_graph(program_node)

    # Add AST_NODE program graph nodes corresponding to Instructions in CFG
    if cfg:
        for control_flow_node in control_flow_graph.get_control_flow_nodes():
            program_graph.add_node_from_instruction(
                control_flow_node.instruction)

    # Add AST_NODE program graph nodes corresponding to AST nodes.
    for ast_node in ast.walk(program_node):
        if isinstance(ast_node, AST_NODE_FILTER):
            continue
        if not program_graph.contains_ast_node(ast_node):
            pg_node = make_node_from_ast_node(ast_node)
            program_graph.add_node(pg_node)

    root = program_graph.get_node_by_ast_node(program_node)
    program_graph.root_id = root.id

    # Add AST edges (FIELD). Also AST_VALUE nodes.
    for ast_node in ast.walk(program_node):
        for field_name, value in ast.iter_fields(ast_node):
            if value is None or isinstance(value, AST_NODE_FILTER):
                continue

            elif isinstance(value, list):
                for index, item in enumerate(value):
                    list_field_name = make_list_field_name(field_name, index)
                    if isinstance(item, ast.AST):
                        program_graph.add_new_edge(
                            ast_node, item,
                            pb.EdgeType.FIELD,
                            list_field_name)
                    else:
                        item_node = make_node_from_ast_value(item)
                        program_graph.add_node(item_node)
                        program_graph.add_new_edge(
                            ast_node, item_node,
                            pb.EdgeType.FIELD,
                            list_field_name)
            elif isinstance(value, ast.AST):
                program_graph.add_new_edge(
                    ast_node, value, pb.EdgeType.FIELD, field_name)

    # Perform data flow analysis.
    if dfg:
        analysis = data_flow.LastAccessAnalysis()
        for node in control_flow_graph.get_enter_control_flow_nodes():
            analysis.visit(node)

    if cfg:
        # Add control flow edges (CFG_NEXT).
        for control_flow_node in control_flow_graph.get_control_flow_nodes():
            instruction = control_flow_node.instruction
            for next_control_flow_node in control_flow_node.next:
                next_instruction = next_control_flow_node.instruction
                program_graph.add_new_edge(
                    instruction.node, next_instruction.node,
                    edge_type=pb.EdgeType.CFG_NEXT)

    # Add data flow edges (LAST_READ and LAST_WRITE).
    if dfg:
        for control_flow_node in control_flow_graph.get_control_flow_nodes():
            # Start with the most recent accesses before this instruction.
            last_accesses = control_flow_node.get_label(
                'last_access_in'
            ).copy()

            for access in control_flow_node.instruction.accesses:
                # Extract the node and identifiers for the current access.
                pg_node = program_graph.get_node_by_access(access)
                access_name = instruction_module.access_name(access)
                read_identifier = instruction_module.access_identifier(
                    access_name, 'read')
                write_identifier = instruction_module.access_identifier(
                    access_name, 'write')

                # Find previous reads.
                for read in last_accesses.get(read_identifier, []):
                    read_pg_node = program_graph.get_node_by_access(read)
                    program_graph.add_new_edge(
                        pg_node, read_pg_node, edge_type=pb.EdgeType.LAST_READ)
        
                # Find previous writes.
                for write in last_accesses.get(write_identifier, []):
                    write_pg_node = program_graph.get_node_by_access(write)
                    program_graph.add_new_edge(
                        pg_node, write_pg_node,
                        edge_type=pb.EdgeType.LAST_WRITE)

                # Update the state to refer to this access as the most recent.
                if instruction_module.access_is_read(access):
                    last_accesses[read_identifier] = [access]
                elif instruction_module.access_is_write(access):
                    last_accesses[write_identifier] = [access]

    # Add COMPUTED_FROM edges.
    for node in ast.walk(program_node):
        if isinstance(node, ast.Assign):
            for value_node in ast.walk(node.value):
                if isinstance(value_node, ast.Name):
                    for target in node.targets:
                        program_graph.add_new_edge(
                            target, value_node,
                            edge_type=pb.EdgeType.COMPUTED_FROM)
    
    return program_graph
