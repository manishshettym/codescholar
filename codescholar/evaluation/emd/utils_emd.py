import os
import os.path as osp
import random

from codescholar.search.elastic_search import grep_programs


def multiapi_trim_code(snippet, apis):
    lines = snippet.split("\n")
    api_lines = []

    for api in apis:
        for i, line in enumerate(lines):
            if api in line:
                api_lines.append(i)
                break
    
    if api_lines == []:
        return snippet

    start_line = max(0, min(api_lines) - 2)
    end_line = min(len(lines), max(api_lines) + 3)
    return "\n".join(lines[start_line:end_line])


def trim_code(snippet, api):
    lines = snippet.split("\n")
    api_line = None
    for i, line in enumerate(lines):
        if api in line:
            api_line = i
            break
    if api_line is None:
        return snippet

    # remove empty lines
    lines = [line for line in lines if line.strip() != ""]

    start_line = max(0, api_line - 2)
    end_line = min(len(lines), api_line + 3)
    return "\n".join(lines[start_line:end_line])


def load_program(path):
    with open(path, "r") as f:
        program = f.read()
    return program


def load_api_progs(args):
    random.seed(42)
    
    if args.benchtype == "single":
        prog_indices = grep_programs(args, args.query)

    elif args.benchtype == "multi":
        assert type(args.query) == list, "Multi-api query must be a list of APIs"

        prog_indices = set()
        for i, seed in enumerate(args.query):
            if i == 0:
                prog_indices = set(grep_programs(args, seed))
            else:
                prog_indices = prog_indices & set(grep_programs(args, seed))
        
        prog_indices = list(prog_indices)
        
    prog_indices = random.sample(prog_indices, min(len(prog_indices), 20000))
    progs = [load_program(f"{args.prog_dir}/example_{i}.py") for i in prog_indices]
    
    if args.benchtype == "single":
        progs = [trim_code(prog, args.query) for prog in progs]

    elif args.benchtype == "multi":
        progs = [multiapi_trim_code(prog, args.query) for prog in progs]
    
    return progs


def load_gpt_idioms(dir_path):
    programs = []
    for file in os.listdir(dir_path):
        programs.append(load_program(osp.join(dir_path, file)))
    return programs


def load_cs_idioms(dir_path):
    programs = []
    for file in os.listdir(dir_path):
        _, size, cluster, nhood_count, hole = file.split("_")
        hole = hole.split(".")[0]

        # if int(hole) == 0 and int(nhood_count) > 0:
        programs.append(load_program(osp.join(dir_path, file)))

    return programs
