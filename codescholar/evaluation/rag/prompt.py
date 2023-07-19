"""Create prompt for different settings: 
 - NL-to-Code generation 
 - More test case generation
"""

import random, string
from typing import Dict, List, Union

from codescholar.evaluation.rag.templates import GPT_NL2CODE, GPT_NL2CODE_FEWSHOT, GPT_NL2CODE_TASK, GPT_NL2CODE_API


def remove_indent(test: str) -> str:
    """Remove the first level of indentation inside unit tests."""
    lines = test.split("\n")
    lines = [l[4:] for l in lines]
    return "\n".join(lines)


def add_indent(test: str) -> str:
    """add 1-level of indentation to be wrapped into the check function."""
    lines = test.split("\n")
    lines = [(" " * 4 + l) for l in lines]
    return "\n".join(lines)


def create_entry_point(
    intent: str,
    first_nwords: int = 4,
    stop_words: List[str] = ["in", "of", "a", "to", "and", "for", "with", "that"],
    delimiters: List[str] = ["`", "'"],
    verbose: bool = False,
) -> str:
    """Heuristically assign (meaningful) function name from the rewritten-intent."""
    words = [w.lower() for w in intent.split()[:first_nwords]]
    if verbose:
        print(f"[words 0] {words}")

    for idx, word in enumerate(words):
        try:
            word_num = float(word)
            break
        except:
            continue
    else:
        idx += 1
    words = words[:idx]
    if verbose:
        print(f"[words 1] {words}")

    for idx, word in enumerate(words):
        if any([word == sw for sw in stop_words]) and idx > 1:
            break
    else:
        idx += 1
    words = words[:idx]
    if verbose:
        print(f"[words 2] {words}")

    for idx, word in enumerate(words):
        if any([word.startswith(de) for de in delimiters]):
            break
    else:
        idx += 1
    words = words[:idx]
    if verbose:
        print(f"[words 3] {words}")

    words = ["".join([c for c in word if (not c in string.punctuation)]) for word in words]
    words = [word for word in words if word.strip()]
    if verbose:
        print(f"[words 4] {words}")
    if len(words) < 2:
        words = ["f"] + words[:first_nwords]

    if words[0].startswith("¿"):
        words[0] = words[0][1:]

    return "_".join(words)


def get_default_entry_point(task_id: Union[int, str]) -> str:
    return f"f_{task_id}"


def replace_entry_point(function_head: str, alternative_name: str) -> str:
    arguments = function_head[function_head.index("(") :]
    return f"def {alternative_name}{arguments}"


def replace_function_name_test(test_case: str, function_name: str) -> str:
    return test_case.replace("candidate", function_name)


def get_test_body(
    function_name: str,
    tests: List[str],
    num_tests: int = 0,
    padding: bool = False,
    clean_indent: bool = True,
) -> str:
    if num_tests == 0:
        return ""

    if len(tests) >= num_tests:
        selected_tests = random.sample(tests, num_tests)
    else:
        if padding:
            selected_tests = []
            for _ in range(num_tests):
                selected_tests.extend(random.sample(tests, 1))
        else:
            selected_tests = tests

    if clean_indent:
        selected_tests = [remove_indent(t) for t in selected_tests]

    selected_tests = [replace_function_name_test(t, function_name) for t in selected_tests]
    return "".join(selected_tests)


def get_entry_point(sample: Dict, method: str) -> str:
    if method == "id":
        return get_default_entry_point(sample["task_id"])
    elif method == "constant":
        return "function"
    elif method == "intent":
        return create_entry_point(sample["intent"])
    else:
        raise ValueError("FuncName Method [{method}] Not Supported!")


def create_prompt_nl2code(
    template: str,
    sample: Dict,
    num_tests: int = 0,
    function_name: str = "id",
) -> str:
    function_head, function_prefix = [p for p in sample["prompt"].split("\n")]

    assert sample["intent"] is not None, f"NL intent is None for sample {sample['task_id']}"

    if function_name == "id":
        function_name = get_default_entry_point(task_id=sample["task_id"])
    elif function_name == "constant":
        function_name = "function"
        function_head = replace_entry_point(function_head, function_name)
    elif function_name == "intent":
        function_name = create_entry_point(intent=sample["intent"])
        function_head = replace_entry_point(function_head, function_name)
    else:
        raise ValueError(f"Method [{function_name}] Not Supported!")

    test_str = get_test_body(function_name, sample["test"], num_tests, padding=False, clean_indent=False)
    docstr = f'{sample["intent"].strip()}\n{test_str}'
    code_body = function_prefix.replace("\t", " " * 4)

    solution = "\n".join([function_head, code_body])
    prompt = template.format(query=docstr, solution=solution)

    return prompt


def create_prompt_example(template: str, example: Dict, num_tests: int = 0, function_name: str = "id") -> str:
    prompt = create_prompt_nl2code(template, example, num_tests, function_name)
    return f'{prompt}{example["canonical_solution"]}{example["suffix"]}\n"""'


# BASELINE
def create_baseline_prompt(
    sample: Dict,
    examples: List[Dict],
    num_tests: int = 0,
    function_name: str = "id",
) -> str:
    assert sample["intent"] is not None, f"NL intent is None for sample {sample['task_id']}"
    prompt = create_prompt_nl2code(template=GPT_NL2CODE_TASK, sample=sample, num_tests=num_tests, function_name=function_name)

    if len(examples) == 0:
        return GPT_NL2CODE.format(prompt=prompt)
    else:
        examples = "\n\n".join(
            [create_prompt_example(template=GPT_NL2CODE_TASK, example=ex, num_tests=0, function_name=function_name) for ex in examples]
        )
        return GPT_NL2CODE_FEWSHOT.format(examples=examples, prompt=prompt)


# APIDISC
def create_apidisc_prompt(
    sample: Dict,
    examples: List[Dict],
    num_tests: int = 0,
    function_name: str = "id",
) -> str:
    assert sample["intent"] is not None, f"NL intent is None for sample {sample['task_id']}"
    prompt = create_prompt_nl2code(template=GPT_NL2CODE_TASK, sample=sample, num_tests=num_tests, function_name=function_name)

    return GPT_NL2CODE_API.format(api=sample["api"], prompt=prompt)


def create_apischolar_prompt(
    sample: Dict,
    examples: List[Dict],
    num_tests: int = 0,
    function_name: str = "id",
) -> str:
    pass
