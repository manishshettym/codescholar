# CodeScholar x LLMs: Evaluation of RAGs for NL2Code

Here, we evaluate CodeScholar as a retriever for retrieval-augmented generation (RAG) with SOTA large-language models
such as GPT3.5 for the NL2Code task. 

Note: the approach described does not implement an end-to-end system. However, it is meant to be an evaluation of an actual and feasible
pipeline of CodeScholar x LLMs for the NL2Code task.

## Dataset
For the NL2Code task, there are several benchmark datasets. However, here we are interested in the use of APIs in code.

## Pipeline
1. First we retrieve relevant APIs for a given task. For this, we simply query the LLM with the task description. [`LLM`]
2. We then use CodeScholar to retrieve relevant usage examples for the API of interest. [`CodeScholar`]
3. Lastly, we use the LLM to generate code from the task description and the retrieved usage examples. [`LLM`]

Note: to simplify evaluation, we preprocess and store the retrieved APIs (1) and usage examples (2) for each task in a database. We then simply query this database for the relevant APIs and usage examples for a given task. This mimics a real-world scenario where we would have a large database of APIs and usage examples for various APIs (all mined by CodeScholar).

## Relevant Files
1. API Discovery + CodeScholar Search: build_retriever.py
3. LLM Code Generation: eval_nl2code.py
