# Author: Manish Shetty
# Description: Run evaluation pipeline for CodeScholar Idiom Search
# [dataset] pnosmt  (dataset containing pandas, numpy, os, matplotlib, and torch code)
# [min_idiom_size] 2  (minimum size of idiom mined)
# [max_idiom_size] 20  (maximum size of idiom mined)
# [max_init_beams] 150  (maximum number of beams to initialize the search with)

# This runs idiom search on a set of benchmark single APIs queries as described in singlebench.json
# python evaluate.py --dataset pnosmt --min_idiom_size 2 --max_idiom_size 20 --max_init_beams 150

# This runs idiom search on a set of benchmark multi APIs queries as described in multibench.json
python evaluate.py --benchtype multi --dataset pnosmt --min_idiom_size 2 --max_idiom_size 20 --max_init_beams 150
