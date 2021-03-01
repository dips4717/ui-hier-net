#!/bin/bash
epochs=(5 10 15)
for ep in ${epochs[@]}
do
    for name in VAE, AE
    do
        for split_set in test, train, valid
        do
            python test.py --model_epoch $ep --split $split_set --exp_name $name
        done
    done
done
