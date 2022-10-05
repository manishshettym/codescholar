def is_library_used(filepath: str, lib: str):
    keywords = [f'import {lib}', f'from {lib}']

    with open(filepath, encoding="utf8", errors='ignore') as fp:
        text = fp.read()

        if any(usage in text for usage in keywords):
            return True

    return False
