import os
import os.path as osp


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


def load_prog_dir(dir_path):
    programs = []
    for file in os.listdir(dir_path):
        programs.append(load_program(osp.join(dir_path, file)))
    return programs
