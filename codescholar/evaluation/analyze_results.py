import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date

DATE = date.today()


def plot_rde(df, api):
    fig, ax = plt.subplots()
    x = df["size"].unique()

    # add a line plot of the number of clusters per size
    y1 = df.groupby("size")["cluster"].nunique()
    ax.plot(x, y1, label="Diversity", marker="o", markersize=3, color="blue")

    # add a line plot of the average number of neighborhoods per size
    y2 = df.groupby("size")["freq"].mean()
    ax.plot(x, y2, label="Reusability", marker="o", markersize=3, color="red")

    ax.set_yscale("log")
    ax.set_xlabel("Size (Expressivity)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("Reusability and Diversity by Expressivity")
    ax.set_xticks(x)
    ax.legend()

    # save the figure to png
    fig.savefig(f"results/{DATE}/{lib}_res/{api}/{api}.rde.png")


with open("smol-benchmark.json") as f:
    benchmarks = json.load(f)

with pd.ExcelWriter(f"results/{DATE}/{DATE}.results.xlsx") as writer:
    for lib in benchmarks:
        for api in benchmarks[lib]:
            programs = []
            sizes = []
            clusters = []
            neighborhoods = []
            holes = []
            plens = []

            try:
                for file in os.listdir(f"results/{DATE}/{lib}_res/{api}/idioms/progs"):
                    _, size, cluster, nhood_count, hole = file.split("_")
                    hole = hole.split(".")[0]

                    sizes.append(int(size))
                    clusters.append(int(cluster))
                    neighborhoods.append(int(nhood_count))
                    holes.append(int(hole))

                    with open(f"results/{DATE}/{lib}_res/{api}/idioms/progs/" + file, "r") as f:
                        program = f.read()
                        programs.append(program)

                    plens.append(len(program))
            except FileNotFoundError:
                pass

            df = pd.DataFrame(
                {"size": sizes, "cluster": clusters, "freq": neighborhoods, "hole": holes, "plen": plens, "program": programs}
            )

            # sort by metrics
            df = df.sort_values(by=["size", "cluster", "freq", "hole", "plen"], ascending=[True, True, False, True, True])

            # save as excel
            df.to_excel(writer, sheet_name=f"{api}", index=False)

            # plot rde
            plot_rde(df, api)
