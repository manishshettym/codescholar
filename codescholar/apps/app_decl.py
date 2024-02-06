"""A flask backend for the CodeScholar app."""

import json
import os
import os.path as osp
import sys
import argparse
import flask
import torch

from codescholar.search.search import main as search_main
from codescholar.apps.utils import find_api, clean_idiom, write_idiom, get_result_from_dir, get_plot_metrics
from codescholar.apps.async_utils import get_celery_app_instance
from codescholar.representation import config
from codescholar.search import search_config
from codescholar.constants import DATA_DIR

api_cache_dir = "./cache/"

################ Flask App ################

scholarapp = flask.Flask(__name__)
celery = get_celery_app_instance(scholarapp)


@celery.task(name="search_task")
def search_task(args_dict):
    try:
        torch.multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    args = argparse.Namespace(**args_dict)
    search_main(args)


@scholarapp.route("/")
def index():
    return "CodeScholar is running!"


@scholarapp.route("/findapi", methods=["POST"])
def findapi():
    query = flask.request.json["query"]
    resp = {"api": find_api(query)}
    return flask.jsonify(resp)


@scholarapp.route("/search", methods=["POST"])
def search():
    api = flask.request.json["api"]
    size = flask.request.json["size"]
    api_cache = osp.join(api_cache_dir, api, "idioms", "progs")

    if osp.exists(api_cache):
        resp = get_result_from_dir(api, api_cache, size)
        return flask.jsonify(resp)
    else:
        parser = argparse.ArgumentParser()
        config.init_optimizer_configs(parser)
        config.init_encoder_configs(parser)
        search_config.init_search_configs(parser)
        args = parser.parse_args()

        # data config
        args.dataset = "pnosmt"
        args.prog_dir = {DATA_DIR}/{args.dataset}/source/"
        args.source_dir = {DATA_DIR}/{args.dataset}/graphs/"
        args.emb_dir = {DATA_DIR}/{args.dataset}/emb/"

        # model config
        args.test = True
        args.model_path = f"../representation/ckpt/model.pt"

        # search config
        args.mode = "q"
        args.seed = api
        args.min_idiom_size = 2
        args.max_idiom_size = 20
        args.max_init_beams = 150
        args.result_dir = f"{api_cache_dir}/{args.seed}/"
        args.idiom_g_dir = f"{args.result_dir}/idioms/graphs/"
        args.idiom_p_dir = f"{args.result_dir}/idioms/progs/"

        if not osp.exists(args.idiom_g_dir):
            os.makedirs(args.idiom_g_dir)

        if not osp.exists(args.idiom_p_dir):
            os.makedirs(args.idiom_p_dir)

        # start a celery task to search for idioms in the background
        args_dict = vars(args)
        search_task.delay(args_dict)
        # search_main(args)

        return flask.jsonify({"status": "CodeScholar is now growing idioms for this API. Please try again in ~2 mins."})


@scholarapp.route("/clean", methods=["POST"])
def clean():
    api = flask.request.json["api"]
    idiom = flask.request.json["idiom"]
    resp = {"idiom": clean_idiom(api, idiom)}
    return flask.jsonify(resp)


@scholarapp.route("/write", methods=["POST"])
def write():
    api = flask.request.json["api"]
    idiom = flask.request.json["idiom"]
    resp = {"idiom": write_idiom(api, idiom)}
    return flask.jsonify(resp)


@scholarapp.route("/plot", methods=["POST"])
def plot():
    api = flask.request.json["api"]
    api_cache = osp.join(api_cache_dir, api, "idioms", "progs")

    if osp.exists(api_cache):
        sizes, clusters, freq = get_plot_metrics(api_cache)
        resp = {"sizes": sizes, "clusters": clusters, "freq": freq}
        return flask.jsonify(resp)
    else:
        return flask.jsonify({})
