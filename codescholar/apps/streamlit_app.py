from collections import namedtuple
import altair as alt
import math
import json
import pandas as pd

import requests
import streamlit as st

st.image("/app/codescholar/codescholar.png")

"""
Find the best code idioms for your task using **CodeScholar**! CodeScholar not only finds the best code idioms for your task, 
but also provides you with the *provenance* -- links to the real-world code snippets that use the idiom!

Go ahead and try it out! Select an API from the dropdown menu below and see what idioms CodeScholar finds for you!
"""


with open("/app/codescholar/codescholar/apps/app_bench.json") as f:
    benchmarks = json.load(f)

api_options = set()
for lib in benchmarks:
    for api in benchmarks[lib]:
        api_options.add(api)

option = st.selectbox(
    'Which API do you want to search for?',
    api_options)

# send the selected API to the backend as a post request
# the backend will return a json of code idioms
with st.spinner('Growing your idioms 🌱...'):
    endpoint = st.secrets["ENDPOINT"]
    response = requests.post(f"http://{endpoint}/search", json={"api": option})

st.write("Here are the top 10 code idioms for " + option + "!")

# parse the json response from the backend
idioms = response.json()

# sort the idioms by frequency
idioms = {k: v for k, v in sorted(idioms.items(), key=lambda item: item[1]["freq"], reverse=True)}

# print each idiom in a separate tab using st.tabs
tabs = st.tabs(["Idiom {}".format(i + 1) for i in range(len(idioms))])

for i, tab in enumerate(tabs):
    with tab:
        st.write("🎓: Found this idiom in {} programs!".format(idioms[str(i)]["freq"]))
        st.code(idioms[str(i)]["idiom"], language="python")
