import ast
from operator import eq
import attrs


@attrs.define(eq=False, repr=False)
class MinedIdiom:
    code: str
