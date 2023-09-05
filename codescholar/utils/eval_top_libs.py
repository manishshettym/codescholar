import os.path as osp
from collections import Counter

import glob
import regex as re
from tqdm import tqdm


SRC_DIR = "../data/pandas/raw"

files = [f for f in sorted(glob.glob(osp.join(SRC_DIR, "*.py")))]
import_regex = r"\s*(?:from|import)\s+(\w+(?:\s*,\s*\w+)*)"
libraries = []

for file in tqdm(files):
    with open(file, "r", encoding="utf-8", errors="ignore") as fp:
        source = fp.read()
        matches = re.findall(import_regex, source)
        for match in matches:
            if "," in match:
                libraries += match.split()
            else:
                libraries.append(match)

counts = Counter(libraries).most_common(100)
for lib, f in counts:
    print(lib, f)
