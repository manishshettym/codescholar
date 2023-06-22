"""A flask backend for the CodeScholar app."""

import json
import os
import os.path as osp
import sys
import argparse

import flask
import torch

from codescholar.representation import config
from codescholar.search import search_config
from codescholar.search.search import main as search_main

api_cache_dir = {
    "df.groupby": "../evaluation/results/2023-06-21/pandas_res/",
    "pd.concat": "../evaluation/results/2023-06-21/pandas_res/",
    "np.dot": "../evaluation/results/2023-06-21/numpy_res/",
    "np.mean": "../evaluation/results/2023-06-21/numpy_res/",
}

app = flask.Flask(__name__)

@app.route('/')
def index():
    return "CodeScholar is running!"

@app.route('/search', methods=['POST'])
def search():
    api = flask.request.json['api']
    try:
        api_cache = osp.join(api_cache_dir[api], api, "idioms", "progs")
    except:
        return "Searching results for API: {}".format(api)

    ideal_size = 10 # assume you can query this for each api
    
    resp = {}

    # check if results exist in cache
    if osp.exists(api_cache):
        count = 0
        
        for file in os.listdir(api_cache):
            _, size, cluster, nhood_count, hole = file.split("_")
            hole = hole.split(".")[0]

            if int(hole) == 0 and int(size) == ideal_size and int(nhood_count) > 0:
                with open(osp.join(api_cache, file), "r") as f:
                    resp.update({count: {
                                    "idiom": f.read(), 
                                    "size": size, 
                                    "cluster": cluster, 
                                    "freq": nhood_count, 
                                    }
                                })
                count += 1
        return flask.jsonify(resp)
    else:
        return "Searching results for API: {}".format(api)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    config.init_optimizer_configs(parser)
    config.init_encoder_configs(parser)
    search_config.init_search_configs(parser)
    args = parser.parse_args()

    # data config
    args.dataset = "pnosmt"
    args.prog_dir = f"../data/{args.dataset}/source/"
    args.source_dir = f"../data/{args.dataset}/graphs/"
    args.emb_dir = f"../data/{args.dataset}/emb/"

    # model config
    args.test = True
    args.model_path = f"../representation/ckpt/model.pt"

    torch.multiprocessing.set_start_method("spawn")
    app.run(host='0.0.0.0', debug=True, port=3003)