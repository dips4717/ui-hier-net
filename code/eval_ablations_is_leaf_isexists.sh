#!/bin/bash
leaf_threshold=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
exists_threshold=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
for lthres in ${leaf_threshold[@]}
do
for ethres in ${exists_threshold[@]}
    do
        for expname in rico_hier_AE_SemWt1_nnemb_nonGNN2, rico_hier_exp_AE_sem_wt_1_nnemb
        do
            python reconstruct_hier.py --exp_name $expname --split 'test' --device 'cuda:3' --is_leaf_thres $lthres --is_exists_thres $ethres
        done
    done
done
