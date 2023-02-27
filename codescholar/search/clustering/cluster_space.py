import glob
import os.path as osp
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, OPTICS

from codescholar.utils.search_utils import sample_prog_embs


def write_results(paths, sizes, labels):
    prog_labels = []
    count = 0
    
    for i, size in enumerate(sizes):
        temp = labels[count : count + size]
        assert len(temp) == size

        prog_labels.append(temp)
        count += size
    
    cluster_d = defaultdict(list)
    for path, labels in zip(paths, prog_labels):
        for i, l in enumerate(labels):
            cluster_d[l].append((path, i))
    
    res = pd.DataFrame({"path": paths, "size": sizes, "label": prog_labels})
    res.to_excel("results/prog_labels.xlsx", index=False)

    res = pd.DataFrame({
        "clusterId": list(cluster_d.keys()),
        "graphs": list(cluster_d.values()),
    })

    res.reset_index(inplace=True, drop=True)
    res.to_excel("results/clusters.xlsx", index=False)
    res.to_pickle("results/clusters.pkl")


def main():
    SRC_DIR = "./tmp/pandas/emb"
    n_workers = 4
    embs, paths, sizes = sample_prog_embs(SRC_DIR, k=500, seed=4)

    # concatenate to get [#neighs x emb-dim]
    embs = torch.cat(embs, dim=0)
    
    dbscan = DBSCAN(
        eps=0.1, min_samples=5,
        metric='euclidean', n_jobs=n_workers).fit(embs)
    
    labels = dbscan.labels_

    # write results to csv
    write_results(paths, sizes, labels)

    # plotting
    plot_embs_idx = [i for i, label in enumerate(labels) if label != -1]
    plot_embs_labels = [label for label in labels if label != -1]
    plot_embs = embs[plot_embs_idx, :]
    
    plt.scatter(
        plot_embs[:, 0], plot_embs[:, 1],
        c=plot_embs_labels)

    plt.savefig("plots/neighborhood.png")
    plt.close()

    # metrics
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    label_counts = dict(Counter(labels))
    plt.bar(label_counts.keys(), label_counts.values())
    plt.savefig("plots/labels.png")
    plt.close()

    print("Total points: {}".format(len(embs)))
    print("Estimated number of clusters: {}".format(n_clusters_))
    print("Estimated noise points: {}".format(n_noise_))
    print("Silhouette Coefficient: {:0.3f}".format(
        silhouette_score(plot_embs, plot_embs_labels)))
    print("Distribution:", label_counts)


if __name__ == "__main__":
    main()
