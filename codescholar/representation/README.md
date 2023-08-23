# CodeScholar Subgraph Representation

This module of CodeScholar implements the subgraph representation learning component. Here, a large set of python programs (treated as graphs) are mapped to an embedding space using a neural model. In this embedding space subgraphs of graphs are forced to be to the lower left of the parent graph.

# Training Procedure

## Dataset Preparation
To train the subgraph representation model, we first prepare the datasets.
One can use any corpus of python programs, for example a dataset mined from GitHub. We provide some utilities to mine GitHub repos in `codescholar/mining/collect_data.py`.

Assuming you have a dataset of python files in `codescholar/data/<dataset_name>/raw`, you can use the following command to prepare the dataset for training:

```bash
cd utils
python dataset_create.py --dataset <dataset_name> --samples <num_samples>
```
This will create a train and test split (80:20 default) of your dataset in `codescholar/representation/tmp/<dataset_name>/train` and `codescholar/representation/tmp/<dataset_name>/test` respectively. Each file in your source dataset is broken into smaller methods/functions and saved as a separate file in the train and test directories.

## Training
```bash
cd codescholar/representation
python learn.py --dataset <dataset_name>
```
There are several other configurations that can be passed to the training script. Please refer to `codescholar/representation/config.py` for more details.

## Testing
```bash
python learn.py --dataset <dataset_name> --test
```