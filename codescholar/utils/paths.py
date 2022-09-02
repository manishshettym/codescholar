import os
import pathlib


def get_home_dir_path() -> str:
    return str(pathlib.Path.home().absolute())

def get_logging_dir_path() -> str:
    return os.path.join(get_home_dir_path(), ".codescholar", "logs")