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
from codescholar.apps.utils import clean_idiom, write_idiom, get_result_from_dir, get_plot_metrics

api_cache_dir = {
    "df.groupby": "../evaluation/results/2023-06-21/pandas_res/",
    "pd.concat": "../evaluation/results/2023-06-21/pandas_res/",
    "np.dot": "../evaluation/results/2023-06-21/numpy_res/",
    "np.mean": "../evaluation/results/2023-06-21/numpy_res/",
}

################ Flask App ################

app = flask.Flask(__name__)


@app.route("/")
def index():
    return "CodeScholar is running!"


@app.route("/search", methods=["POST"])
def search():
    api = flask.request.json["api"]
    size = flask.request.json["size"]

    try:
        api_cache = osp.join(api_cache_dir[api], api, "idioms", "progs")
    except:
        return "Searching results for API: {}".format(api)

    if osp.exists(api_cache):
        resp = get_result_from_dir(api, api_cache, size)
        return flask.jsonify(resp)
    else:
        return "Searching results for API: {}".format(api)


@app.route("/clean", methods=["POST"])
def clean():
    api = flask.request.json["api"]
    idiom = flask.request.json["idiom"]
    resp = {"idiom": clean_idiom(api, idiom)}
    return flask.jsonify(resp)


@app.route("/write", methods=["POST"])
def write():
    api = flask.request.json["api"]
    idiom = flask.request.json["idiom"]
    resp = {"idiom": write_idiom(api, idiom)}
    return flask.jsonify(resp)


@app.route("/plot", methods=["POST"])
def plot():
    api = flask.request.json["api"]
    try:
        api_cache = osp.join(api_cache_dir[api], api, "idioms", "progs")
    except:
        return None

    if osp.exists(api_cache):
        sizes, clusters, freq = get_plot_metrics(api_cache)
        resp = {"sizes": sizes, "clusters": clusters, "freq": freq}
        return flask.jsonify(resp)
    else:
        return None


if __name__ == "__main__":
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
    app.run(host="0.0.0.0", debug=True, port=3003)
