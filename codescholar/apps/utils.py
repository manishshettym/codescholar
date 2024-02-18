import os
import os.path as osp
import openai
import json
from networkx.readwrite import json_graph

from codescholar.utils.graph_utils import nx_to_sast
from codescholar.sast.sast_utils import sast_to_prog
from codescholar.utils.search_utils import read_prog

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

gpt_prompt_clean = """
The following is an example of a idiomatic usage example for {api}
\"\"\"
{idiom}
\"\"\"
Rename the variables and constants to make it more generic. Return only the code with no explanations.
\"\"\"
"""

gpt_prompt_write = """
The following is an example of a idiomatic usage example for {api}
\"\"\"
{idiom}
\"\"\"
Write a 2-4 line snippet of code using exactly the above given example. Return only the code with no explanations.
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


def write_idiom(api, idiom):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=gpt_prompt_write.format(api=api, idiom=idiom),
        temperature=0,
        max_tokens=250,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=['"""', "```"],
    )
    return response.choices[0].text


def clean_idiom(api, idiom):
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=gpt_prompt_clean.format(api=api, idiom=idiom),
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
                with open(prog_path, "r") as f:
                    prog = f.read()

                graph = json_graph.node_link_graph(data["graph"])
                sast = nx_to_sast(graph)
                idiom = sast_to_prog(sast).replace("#", "_")

                results.update(
                    {
                        count: {
                            "idiom": idiom,
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
