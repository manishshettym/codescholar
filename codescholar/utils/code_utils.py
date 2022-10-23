import ast
import black
import astunparse


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
