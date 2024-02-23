import os
import re
import math
import json
import pandas as pd
import numpy as np
from collections import namedtuple
import altair as alt
import time
import requests
import streamlit as st
import streamlit.components.v1 as components


def highlight_code_with_html(idiom_code):
    # Replace <mark> tags with placeholders
    placeholder_start = "##START_HIGHLIGHT##"
    placeholder_end = "##END_HIGHLIGHT##"
    highlighted_code = idiom_code.replace("<mark>", placeholder_start).replace(
        "</mark>", placeholder_end
    )

    code_html = f"""
    <html>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/default.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <script>
        document.addEventListener('DOMContentLoaded', (event) => {{
            document.querySelectorAll('pre code').forEach((block) => {{
                hljs.highlightElement(block);
                // Replace placeholders with <span> elements for highlighting after syntax highlighting
                const regexStart = new RegExp('{placeholder_start}', 'g');
                const regexEnd = new RegExp('{placeholder_end}', 'g');
                block.innerHTML = block.innerHTML.replace(regexStart, "<span class='custom-highlight'>");
                block.innerHTML = block.innerHTML.replace(regexEnd, "</span>");
            }});
        }});
        </script>
    </head>
    <body>
        <style>
        .code-container {{
            background-color: #f6f8fa; /* Code block background color */
            border-radius: 5px;
            padding: 10px;
            font-family: monospace;
            overflow: auto; /* Allow scrolling */
        }}
        pre {{
            margin: 0;
        }}
        code {{
            white-space: pre;
        }}
        .custom-highlight {{
            background-color: rgba(255, 255, 0, 0.5); /* Custom highlight color */
            color: black!important;
        }}
        </style>
        <div class="code-container">
            <pre><code class="python">{highlighted_code}</code></pre>
        </div>
    </body>
    </html>
    """
    return code_html


# deployment
# root = "/app/codescholar/"
# endpoint = st.secrets["ENDPOINT"]

# local
root = "../../"
endpoint = "localhost:3003"

st.image(f"{root}/codescholar.png")

if "input_type" not in st.session_state:
    st.session_state["input_type"] = "dropdown"

if "last_query" not in st.session_state:
    st.session_state["last_query"] = None

if "dropdown_api" not in st.session_state:
    st.session_state["dropdown_api"] = None

if "search_api" not in st.session_state:
    st.session_state["search_api"] = None


def dropdown_selected():
    st.session_state["input_type"] = "dropdown"


def search_selected():
    st.session_state["input_type"] = "search"


def check_search_status(api):
    response = requests.get(f"http://{endpoint}/search_status", params={"api": api})
    return response.json()


def get_idioms(api, size):
    response = requests.post(
        f"http://{endpoint}/search", json={"api": api, "size": size}
    )
    return response.json()


API = None

"""
Find the best code idioms for your task using **CodeScholar**! CodeScholar not only finds the best code idioms for your APIs, 
but also provides you with the many other features such as: (1) *provenance*, (2) *idiomaticity*, and (3) *diversity*.
Further it also provides *LLM plugins* to automatically clean the idiom or generate some simple runnable code using it!
"""

api_dirs = [d for d in os.listdir("cache") if os.path.isdir(f"./cache/{d}")]
api_options = set()
for api in api_dirs:
    api_progs = os.path.join("cache", api, "idioms", "progs")
    if os.path.exists(api_progs) and len(os.listdir(api_progs)) > 0:
        api_options.add(api)

st.session_state.dropdown_api = st.selectbox(
    "Go ahead, try it out! Select a popular API:",
    api_options,
    on_change=dropdown_selected,
)

with st.expander("Can't find your API? 🧐"):
    query = st.text_input(
        "Just type your API/query below and we'll find some usage examples for you!",
        value="",
        key="search",
        on_change=search_selected,
    )

    if query:
        if st.session_state.last_query != query:
            st.session_state.last_query = query
            with st.spinner("Finding your API 🗺️..."):
                response = requests.post(
                    f"http://{endpoint}/findapi", json={"query": query}
                )

            st.session_state.search_api = response.json()["api"]
            st.write(
                "We found the following API for your query: **{}**".format(
                    st.session_state.search_api
                )
            )
        else:
            st.write(
                "We found the following API for your query: **{}**".format(
                    st.session_state.search_api
                )
            )


if (
    st.session_state.input_type == "dropdown"
    and st.session_state.dropdown_api is not None
):
    API = st.session_state.dropdown_api
elif (
    st.session_state.input_type == "search" and st.session_state.search_api is not None
):
    API = st.session_state.search_api

size = st.slider("Choose the size of your idiom:", 4, 20, 4)

status_placeholder = st.empty()

with st.spinner("Growing your idioms 🌱..."):
    idioms = get_idioms(API, size)
    status = check_search_status(API)["status"]
    i = 0
    while status != "ready":
        status_placeholder.info(f"[{i}] {API} API is new to CodeScholar! {status}")
        time.sleep(5)
        status = check_search_status(API)["status"]
        i += 1

    status_placeholder.empty()
    idioms = get_idioms(API, size)


if len(idioms) == 0:
    st.error("No idioms found for API: {} 🫙".format(API))
else:
    idioms = [
        v
        for k, v in sorted(
            idioms.items(), key=lambda item: int(item[1]["freq"]), reverse=True
        )
    ]
    tabs = st.tabs(["Idiom {}".format(i + 1) for i in range(len(idioms))])

    for idiom, tab in zip(idioms, tabs):
        with tab:
            st.write("🎓: Found this idiom in {} programs!".format(idiom["freq"]))
            highlighted_idiom_html = highlight_code_with_html(idiom["idiom"])
            maxh = max(100, min(30 * idiom["idiom"].count("\n"), 500))
            components.html(highlighted_idiom_html, height=maxh, scrolling=True)

            colbut1, colbut2 = st.columns([0.25, 0.8])
            with colbut1:
                but1 = st.button("Clean this idiom?", key=f"clean_{tabs.index(tab)}")
            with colbut2:
                but2 = st.button(
                    "Code with this idiom?", key=f"write_{tabs.index(tab)}"
                )

            if but1:
                with st.spinner("Cleaning your idioms 🧹..."):
                    response = requests.post(
                        f"http://{endpoint}/clean",
                        json={"api": API, "idiom": idiom["idiom"]},
                    )
                    # st.error("This feature is not available yet! 🫙")
                st.code(response.json()["idiom"], language="python")
            if but2:
                with st.spinner("Writing some code 👩🏻‍💻..."):
                    response = requests.post(
                        f"http://{endpoint}/write",
                        json={"api": API, "idiom": idiom["idiom"]},
                    )
                    # st.error("This feature is not available yet! 🫙")
                st.code(response.json()["idiom"], language="python")

    st.divider()

    """
    ##### CodeScholar Suggestions
    """
    with st.spinner("Analyzing your idioms 📊..."):
        response = requests.post(f"http://{endpoint}/plot", json={"api": API})

    metrics = response.json()
    metrics_df = pd.DataFrame(
        {
            "size": metrics["sizes"],
            "cluster": metrics["clusters"],
            "freq": metrics["freq"],
        }
    )

    x = metrics_df["size"].unique()
    y1 = np.log(metrics_df.groupby("size")["cluster"].nunique())
    y2 = np.log(metrics_df.groupby("size")["freq"].mean())

    try:
        ideal_size = np.where(y1 > y2)[0][0] + 3
    except IndexError:
        pass
    else:
        chart_data = pd.DataFrame(
            {"Size (Expressivity)": sorted(x), "Diversity": y1, "Reusability": y2}
        )
        chart_data = chart_data[chart_data["Reusability"] > 0]

        col1, col2 = st.columns([2, 1])
        col1.line_chart(
            chart_data, x="Size (Expressivity)", y=["Diversity", "Reusability"]
        )
        col2.write("Ideal Idiom Size is **{} (± 1)** for {}".format(ideal_size, API))
        col2.write(
            "At this size, reusability, diversity, and expressivity are at an equilibrium!!"
        )
