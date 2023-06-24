import os
import os.path as osp
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


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


def write_idiom(api, idiom):
    response = openai.Completion.create(
        model="text-davinci-003",
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
        model="text-davinci-003",
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

        if int(hole) == 0 and int(size) == select_size and int(nhood_count) > 0:
            with open(osp.join(api_cache, file), "r") as f:
                results.update({count: {
                                "idiom": f.read(), 
                                "size": size, 
                                "cluster": cluster, 
                                "freq": nhood_count, 
                                }
                            })
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