import unittest
import textwrap
import ast

from codescholar.mining.code_farming.code_farming import codescholar_codefarmer
from codescholar.utils.code_utils import normalize_code_str, normalize_code_ast


class CodeFarmingTests(unittest.TestCase):
    def test_code_farming(self):
        p1 = textwrap.dedent(
            """
            a = ["foo", "bar", "zoo"]
            for i in a:
                for j in i:
                    print(j)
            """
        )

        dataset = [ast.parse(p1)]
        mined_code = codescholar_codefarmer(dataset, min_freq=0.3,
                                            fix_max_len=True, max_len=3)

        self.assertEqual(len(mined_code), 2)

        self.assertNotEqual(mined_code[1], None)

        self.assertNotEqual(mined_code[2], None)
        
        result0 = set(normalize_code_ast(code.idiom)
                      for code in mined_code[1][0])
        result1 = set(normalize_code_ast(code.idiom)
                      for code in mined_code[2][0])
                
        target_gen0 = [
            textwrap.dedent("""a = ['foo', 'bar', 'zoo']"""),
            textwrap.dedent("""print(j)"""),
            textwrap.dedent(
                """
                for i in a:
                    for j in i:
                        print(j)
                """
            ),
            textwrap.dedent(
                """
                for j in i:
                    print(j)
                """
            )
        ]
        
        target_gen1 = [
            textwrap.dedent(
                """
                a = ['foo', 'bar', 'zoo']
                for i in a:
                    for j in i:
                        print(j)
                """
            ),
            textwrap.dedent(
                """
                a = ['foo', 'bar', 'zoo']
                for j in i:
                    print(j)
                """
            ),
            textwrap.dedent(
                """
                a = ['foo', 'bar', 'zoo']
                print(j)
                """
            )
        ]
        
        target_gen0 = set(normalize_code_str(target) for target in target_gen0)
        target_gen1 = set(normalize_code_str(target) for target in target_gen1)

        self.assertEqual(target_gen0, result0)
        self.assertEqual(target_gen1, result1)
