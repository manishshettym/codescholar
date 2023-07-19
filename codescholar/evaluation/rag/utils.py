"""Dataset utilities. 
"""

import json
import torch
import random
from typing import Dict, List


def select_fewshot_examples(
    sample: Dict,
    candidates: List[Dict],
    num_examples: int = 1,
    method: str = "random",
) -> List[Dict]:
    """Select example as prefix to the prompt of the current sample."""
    if method == "random":
        num_examples = min(num_examples, len(candidates))
        return random.sample(candidates, num_examples)
