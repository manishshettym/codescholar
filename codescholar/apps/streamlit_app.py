from collections import namedtuple
import altair as alt
import math
import json
import pandas as pd
import streamlit as st

"""
<img align="center" src="https://github.com/tart-proj/codescholar/blob/ms/evaluation/codescholar.png"/>

Find the best code idioms for your task using CodeScholar! CodeScholar not only 
finds the best code idioms for your task, but also provides you with the **provenance** -- links to the
real-world code snippets that use the idiom!

Go ahead and try it out! Select an API from the dropdown menu below and see what idioms CodeScholar finds for you!
"""


with open("app_bench.json") as f:
    benchmarks = json.load(f)

api_options = set()
for lib in benchmarks:
    for api in benchmarks[lib]:
        api_options.add(api)

option = st.selectbox(
    'Which API do you want to search for?',
    api_options)

# with st.echo(code_location='below'):
#     total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#     num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

#     Point = namedtuple('Point', 'x y')
#     data = []

#     points_per_turn = total_points / num_turns

#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))

#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q'))