# Reproducing Codescholar Evaluation Results

## Table of Contents
- [Single API Search Evaluation](#single-api-search-evaluation)
  - [Dataset](#dataset)
  - [`CodeScholar` Results](#codescholar-results)
  - [`GPT-3.5` Results](#gpt-35-results)
- [Multi API Search Evaluation](#multi-api-search-evaluation)
    - [Dataset](#dataset-1)
    - [`CodeScholar` Results](#codescholar-results-1)
    - [`GPT-3.5` Results](#gpt-35-results-1)
- [Retrieval-Augmented Generation Evaluation](#retrieval-augmented-generation-evaluation)
    - [Dataset](#dataset-2)
    - [Built the retriever](#built-the-retriever)
    - [Evaluate NL2Code generation](#evaluate-nl2code-generation)




## Single API Search Evaluation

### Dataset
The benchmark dataset is available at [singlebench.json](./singlebench.json). It covers ~60 APIs across 6 popular python libraries: `{pandas,
numpy, os, sklearn, matplotlib, and torch}`

### `CodeScholar` Results
```bash
python evaluate.py --benchtype single --dataset pnosmt --min_idiom_size 2 --max_idiom_size 20 --max_init_beams 150
```

### `GPT-3.5` Results
```bash
cd gpt
python eval_singlbench.py
```



## Multi API Search Evaluation

### Dataset
The benchmark dataset is available at [multibench.json](./multibench.json). It covers 25 multi-API queries split into
three categories. `single-library-pairs` are 15 queries that contain two APIs from the same library.
`mixed-library-pairs` are 5 queries that contain two APIs from different libraries. 
Finally, `triplets` are 5 queries that contain three APIs.

### `CodeScholar` Results
```bash
python evaluate.py --benchtype multi --dataset pnosmt --min_idiom_size 2 --max_idiom_size 30 --max_init_beams 150
```

### `GPT-3.5` Results
```bash
cd gpt
python eval_multibench.py
```



## Retrieval-Augmented Generation Evaluation
Details about the RAG evaluation are available at [codescholar/evaluation/rag/README.md](./codescholar/evaluation/rag/README.md). However to reproduce
the results:

### Dataset
The benchmark dataset is available at [cs_rag.jsonl](./ragbench.json). It is a subset of the [ODEX](https://code-eval.github.io/) dataset for NL2Code
generation task. It contains 85 problems/tasks covering APIsi in `{pandas, numpy, os, sklearn, matplotlib, and torch}`.


### Built the retriever
```bash
cd rag
python build_retriever.py --dataset pnosmt --get-apis --mine-examples --build-index
```
This creates an API to Idioms mapping at `./data/api2idioms.json` for each problem/task in the benchmark dataset.


### Evaluate NL2Code generation
```bash
# baseline: just gpt-3.5
python eval_nl2code.py --experiment baseline

# apidisc: gpt-3.5 + api discovery (relevant API added to prompt)
python eval_nl2code.py --experiment apidisc

# apischolar: gpt-3.5 + api discovery + codescholar (relevant API added to prompt + relevant idioms added to prompt)
python eval_nl2code.py --experiment apischolar
```
