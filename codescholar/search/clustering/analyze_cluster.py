import os.path as osp
import pandas as pd

import torch


def main():
    data = pd.read_pickle("results/clusters.pkl")
    cluster_id = 4

    cluster_data = data[data.clusterId == cluster_id]["graphs"].values[0]

    for file, sub_idx in cluster_data:
        graph_path = "data_" + file.split("_")[-1]
        graph_path = osp.join("./tmp/pandas/processed", graph_path)
        print(graph_path)

        neighs = torch.load(graph_path, map_location=torch.device("cpu"))
        graph = neighs[sub_idx]
        print(graph.span)


if __name__ == "__main__":
    main()
