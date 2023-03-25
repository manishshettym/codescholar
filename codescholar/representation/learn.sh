#!/bin/bash

gendata=false;
samples=3000;


# generate data if needed
if $gendata -eq true
then
    # cd into the util directory
    cd ../utils

    # run the sample_dataset script
    python sample_dataset.py --task train --samples $samples

    # cd back into the representation directory
    cd ../representation
fi

# run the training script
# configs are in config.py
python learn.py

# run the evaluation script
python learn.py --test

