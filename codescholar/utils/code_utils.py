import os.path as osp
import ast
import ast
import black
import astunparse


def create_dummy_function(body) -> ast.FunctionDef:
    """
    Create a dummy ast.FunctionDef object with
    name = "main" and body = body
    """
    func_args = ast.arguments(
        posonlyargs=[], args=[], vararg=None,
        kwonlyargs=[], kw_defaults=[],
        kwarg=None, defaults=[])
    
    func = ast.FunctionDef(
        name="main",
        args=func_args,
        body=body,
        decorator_list=[],
        returns=None,
        type_comment=None)

    return func


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


def breakdown_code_methods(outdir: str, path: str, file_id: str):
    """Breakdown a python file into methods.
    Save the methods into seperate files.

    Args:
        path (str): path to the .py file
    """
    example_id = 0
    code = None
    with open(path, 'r') as fp:
        source = fp.read()
        code = ast.parse(source, mode='exec')

    # process methods
    classes = [n for n in code.body if isinstance(n, ast.ClassDef)]

    if len(classes) > 0:
        for class_ in classes:
            methods = [
                n for n in class_.body
                if isinstance(n, ast.FunctionDef)]
            
            for meth in methods:
                example_name = "{}_{}.py".format(file_id, example_id)
                with open(osp.join(outdir, example_name), 'w') as fp:
                    fp.write(astunparse.unparse(meth))
                
                example_id += 1

    # process functions
    functions = [n for n in code.body if isinstance(n, ast.FunctionDef)]

    if len(functions) > 0:
        for func in functions:
            example_name = "{}_{}.py".format(file_id, example_id)
            with open(osp.join(outdir, example_name), 'w') as fp:
                fp.write(astunparse.unparse(func))
            
            example_id += 1
    
    # drop all class and function defs
    code = ASTMethodDropper().visit(code)

    # wrap the rest in a FunctionDef
    if ast.unparse(code).strip() != "":
        main_code = create_dummy_function(code.body)
        example_name = "{}_{}.py".format(file_id, example_id)

        with open(osp.join(outdir, example_name), 'w') as fp:
            fp.write(astunparse.unparse(main_code))


class ASTMethodDropper(ast.NodeTransformer):

    def visit_Import(self, node: ast.Import):
        return None
    
    def visit_ImportFrom(self, node: ast.ImportFrom):
        return None
    
    def visit_ClassDef(self, node: ast.ClassDef):
        super().generic_visit(node)
        return None
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        super().generic_visit(node)
        return None
