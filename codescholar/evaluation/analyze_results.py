import os
import json
import pandas as pd
import matplotlib.pyplot as plt

DATE = "2023-05-23"

with open('benchmarks.json') as f:
    benchmarks = json.load(f)

with pd.ExcelWriter(f"results/{DATE}/{DATE}.results.xlsx") as writer:
    for lib in benchmarks:
        for api in benchmarks[lib]:
            programs = []
            sizes = []
            freqs = []
            scores = []
            holes = []
            plens = []

            try:
                for file in os.listdir(f"results/{DATE}/{lib}_res/{api}/idioms/progs"):
                    _, size, id, freq, score, hole = file.split("_")
                    hole = hole.split(".")[0]

                    sizes.append(int(size))
                    scores.append(int(score))
                    freqs.append(int(freq))
                    holes.append(int(hole))

                    # read the program
                    with open(f"results/{DATE}/{lib}_res/{api}/idioms/progs/" + file, "r") as f:
                        program = f.read()
                        programs.append(program)
                    
                    plens.append(len(program))
            except FileNotFoundError:
                pass


            df = pd.DataFrame({
                "size": sizes,
                "freq": freqs,
                "score": scores,
                "hole": holes,
                "plen": plens,
                "program": programs
            })

            # sort by metrics
            df = df.sort_values(by=["size", "freq", "score", "hole", "plen"], ascending=[True, False, False, True, True])

            # save as excel
            df.to_excel(writer, sheet_name=f"{api}", index=False)
