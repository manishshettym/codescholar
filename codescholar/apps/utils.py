import os
import os.path as osp
import openai
import json
import networkx as nx
import torch
from networkx.readwrite import json_graph

from codescholar.utils.graph_utils import nx_to_sast
from codescholar.sast.sast_utils import sast_to_prog
from codescholar.utils.search_utils import read_prog, read_graph

from codescholar.constants import DATA_DIR

openai.api_key = os.getenv("OPENAI_API_KEY")


gpt_prompt_find = """
Return an appropriate python API for the following query.
Only return the API name, with no explanations. For example:
Query: How do I find the mean of a numpy array?
"\"\"
np.mean
"\"\"

Query: {query}
\"\"\"
"""

gpt_prompt_explain = """
The following is an example of a idiomatic usage example for {api}.
The idiomatic usage is annotated in the code example.
\"\"\"
{idiom}
\"\"\"
Write a 4-6 line explanation of how the API is used in the code example.
Write an explanation that is specific to this code example.
Explain the setting not just the usage. Use markdown syntax to format the explanation
and highlight the important parts. Return only the explanation with no code.
\"\"\"
"""


def find_api(query):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=gpt_prompt_find.format(query=query),
        temperature=0,
        max_tokens=250,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['"""', "```"],
    )
    return response.choices[0].text


def explain_idiom(api, idiom):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=gpt_prompt_explain.format(api=api, idiom=idiom),
        temperature=0,
        max_tokens=250,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['"""', "```"],
    )
    return response.choices[0].text


def get_result_from_dir(api, api_cache, select_size):
    results, count = {}, 0
    for file in os.listdir(api_cache):
        _, size, cluster, nhood_count, hole = file.split("_")
        hole = hole.split(".")[0]

        if int(size) == select_size and int(nhood_count) > 0:
            with open(osp.join(api_cache, file), "r") as f:

                data = json.load(f)
                idx = data["index"]
                prog_path = f"{DATA_DIR}/pnosmt/source/example_{idx}.py"
                graph_path = f"{DATA_DIR}/pnosmt/graphs/data_{idx}.pt"

                with open(prog_path, "r") as f:
                    prog = f.read()
                    p_graph = torch.load(graph_path, map_location=torch.device("cpu"))

                # highlight idiom in prog
                i_graph = json_graph.node_link_graph(data["graph"])
                i_subg = p_graph.subgraph(i_graph).copy()
                i_subg.remove_edges_from(nx.selfloop_edges(i_subg))
                for v in p_graph.nodes:
                    p_graph.nodes[v]["is_idiom"] = 1 if v in i_subg.nodes else 0

                # convert to sast and extract highlighted code
                idiom_prog = sast_to_prog(nx_to_sast(p_graph), mark_idiom=True)

                results.update(
                    {
                        count: {
                            "idiom": idiom_prog,
                            "size": size,
                            "cluster": cluster,
                            "freq": nhood_count,
                        }
                    }
                )
            count += 1

    return results


def get_plot_metrics(api_cache):
    sizes, clusters, freq = [], [], []
    for file in os.listdir(api_cache):
        _, size, cluster, nhood_count, _ = file.split("_")
        sizes.append(int(size))
        clusters.append(int(cluster))
        freq.append(int(nhood_count))

    return sizes, clusters, freq
