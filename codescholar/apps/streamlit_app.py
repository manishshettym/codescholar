from collections import namedtuple
import altair as alt
import math
import json
import pandas as pd
import numpy as np

import requests
import streamlit as st

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


API = None

"""
Find the best code idioms for your task using **CodeScholar**! CodeScholar not only finds the best code idioms for your APIs, 
but also provides you with the many other features such as: (1) *provenance*, (2) *idiomaticity*, and (3) *diversity*.
Further it also provides *LLM plugins* to automatically clean the idiom or generate some simple runnable code using it!
"""

with open(f"{root}/codescholar/apps/app_bench.json") as f:
    benchmarks = json.load(f)

api_options = set()
for lib in benchmarks:
    for api in benchmarks[lib]:
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

with st.spinner("Growing your idioms 🌱..."):
    response = requests.post(
        f"http://{endpoint}/search", json={"api": API, "size": size}
    )
    idioms = response.json()

    if "status" in idioms:
        st.info("{} API is new to CodeScholar! {}".format(API, idioms["status"]))
    elif len(idioms) == 0:
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
                st.code(idiom["idiom"], language="python")

                colbut1, colbut2 = st.columns([0.25, 0.8])
                with colbut1:
                    but1 = st.button(
                        "Clean this idiom?", key=f"clean_{tabs.index(tab)}"
                    )
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

            col1, col2 = st.columns([2, 1])
            col1.line_chart(
                chart_data, x="Size (Expressivity)", y=["Diversity", "Reusability"]
            )
            col2.write(
                "Ideal Idiom Size is **{} (± 1)** for {}".format(ideal_size, API)
            )
            col2.write(
                "At this size, reusability, diversity, and expressivity are at an equilibrium!!"
            )
