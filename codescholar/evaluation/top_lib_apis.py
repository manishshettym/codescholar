import os.path as osp
from collections import Counter, defaultdict

import glob
import regex as re
from tqdm import tqdm
import ast

def extract_libraries_and_aliases(source):
    libraries = {}

    try:
        tree = ast.parse(source)
    except:
        return {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                libraries[alias.name] = alias.asname or alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            if node.module in libraries:
                libraries[node.module] = node.module
            else:
                libraries[node.module] = node.names[0].asname or node.names[0].name

    return libraries


def recurse_attr_extract(node):
    if isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"

    elif isinstance(node.value, ast.Attribute):
        chained_api = recurse_attr_extract(node.value)
        if chained_api:
            return chained_api + f".{node.attr}"

    return None

def extract_function_calls(node):
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name):
            yield f"{node.value.id}.{node.attr}"

        # chained attribute. e.g., np.random.randint
        elif isinstance(node.value, ast.Attribute):
            chained_api = recurse_attr_extract(node.value) 
            if chained_api:
                yield chained_api + f".{node.attr}"
            
    for child in ast.iter_child_nodes(node):
        yield from extract_function_calls(child)


def extract_library_calls(source, library_name, library_aliases):
    tree = ast.parse(source)
    # print(ast.dump(tree, indent=4))
    for call in extract_function_calls(tree):
        if call.startswith(library_name + "."):
            yield call
        elif call.split(".")[0] in library_aliases:
            yield call


LIBS = ["pandas", "numpy", "os", "sklearn", "matplotlib", "torch"]
SRC_DIR = "../data/pandas/raw"
files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, '*.py')))]

lib_apis = defaultdict(list)

for file in tqdm(files):
    with open(file ,'r', encoding='utf-8', errors='ignore') as fp:
        source = fp.read()
    
    libs_and_alias = extract_libraries_and_aliases(source)

    for lib in LIBS:
        if lib in libs_and_alias:
            apis = list(extract_library_calls(source, lib, libs_and_alias[lib]))
            apis = [".".join(api.split(".")[1:]) for api in apis]
            lib_apis[lib] += apis

for lib in LIBS:
    print(lib)
    print("=========")
    for api, f in Counter(lib_apis[lib]).most_common():
        print(api, f)
    print("=========\n\n")

