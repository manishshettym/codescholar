from collections import namedtuple
import altair as alt
import math
import json
import pandas as pd
import numpy as np

import requests
import streamlit as st

# deployment
root = "/app/codescholar/"
endpoint = st.secrets["ENDPOINT"]

# local
# root = "../../"
# endpoint = "localhost:3003"

st.image(f"{root}/codescholar.png")

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

option = st.selectbox("Go ahead, try it out! Select an API from the dropdown menu below:", api_options)

size = st.slider("Idiom Size", 4, 20, 4)

with st.spinner("Growing your idioms 🌱..."):
    response = requests.post(f"http://{endpoint}/search", json={"api": option, "size": size})

idioms = response.json()

if len(idioms) == 0:
    st.error("No idioms found for API: {} 🫙".format(option))
else:
    idioms = [v for k, v in sorted(idioms.items(), key=lambda item: int(item[1]["freq"]), reverse=True)]
    tabs = st.tabs(["Idiom {}".format(i + 1) for i in range(len(idioms))])

    for idiom, tab in zip(idioms, tabs):
        with tab:
            st.write("🎓: Found this idiom in {} programs!".format(idiom["freq"]))
            st.code(idiom["idiom"], language="python")

            colbut1, colbut2 = st.columns([0.25, 0.8])
            with colbut1:
                but1 = st.button("Clean this idiom?", key=f"clean_{tabs.index(tab)}")
            with colbut2:
                but2 = st.button("Code with this idiom?", key=f"write_{tabs.index(tab)}")

            if but1:
                with st.spinner("Cleaning your idioms 🧹..."):
                    # response = requests.post(f"http://{endpoint}/clean", json={"api": option, "idiom": idiom["idiom"]})
                    st.error("This feature is not available yet! 🫙")
                # st.code(response.json()["idiom"], language="python")
            if but2:
                with st.spinner("Writing some code 👩🏻‍💻..."):
                    # response = requests.post(f"http://{endpoint}/write", json={"api": option, "idiom": idiom["idiom"]})
                    st.error("This feature is not available yet! 🫙")
                # st.code(response.json()["idiom"], language="python")

    st.divider()

    """
    ##### CodeScholar Suggestions
    """
    with st.spinner("Analyzing your idioms 📊..."):
        response = requests.post(f"http://{endpoint}/plot", json={"api": option})

    metrics = response.json()
    metrics_df = pd.DataFrame({"size": metrics["sizes"], "cluster": metrics["clusters"], "freq": metrics["freq"]})

    x = metrics_df["size"].unique()
    y1 = np.log(metrics_df.groupby("size")["cluster"].nunique())
    y2 = np.log(metrics_df.groupby("size")["freq"].mean())
    ideal_size = np.where(y1 > y2)[0][0] + 3

    chart_data = pd.DataFrame({"Size (Expressivity)": sorted(x), "Diversity": y1, "Reusability": y2})

    col1, col2 = st.columns([2, 1])
    col1.line_chart(chart_data, x="Size (Expressivity)", y=["Diversity", "Reusability"])
    col2.write("Ideal Idiom Size is **{} (± 1)** for {}".format(ideal_size, option))
    col2.write("At this size, reusability, diversity, and expressivity are at an equilibrium!!")
