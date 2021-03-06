#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 12:39:27 2021
Reconstruction and visualization RICO Hier 
Modified from prj-layout by Zhenping
@author: dipu
"""
from rico import *
from utils import *
from moka import *
from datasets import *
from scipy.optimize import linear_sum_assignment

import os
import time
import sys
import shutil
import random
from time import strftime
import argparse
import numpy as np
import torch
import torch.utils.data
from config_rico import add_eval_args
#from data import PartNetDataset, Tree
from rico import Hierarchy
from datasets import RicoFlatDataset, RicoHierDataset
import utils
import time 


def vis_fn(uxid, o, o_gt, z, edit_distance, IoU):
    return dict(
        id = uxid,
        pred = o.to_string(render='html', labeled=o.is_labeled),
        pred_layout = o.plot(),
        gt = o_gt.to_string(render='html', labeled=o.is_labeled),
        gt_layout = o_gt.plot(),
        IoU = IoU,
        edit_distance = edit_distance
    )


def max_linear_assignment(boxes_pred, boxes_gt, reward_fn=batch_iou):
    N, K = len(boxes_pred), len(boxes_gt)

    M = -reward_fn(
        np.tile(boxes_pred.reshape([N, 1, 4]), [1, K, 1]).reshape([N*K, 4]),
        np.tile(boxes_gt  .reshape([1, K, 4]), [N, 1, 1]).reshape([N*K, 4])
    ).reshape([N, K])

    pred_inds, gt_inds = linear_sum_assignment(M)
    assert len(pred_inds) == len(gt_inds) == min(N, K)

    reward = -M[pred_inds, gt_inds].sum()

    pred_inds = list(pred_inds) + list(set(range(max(N, K))) - set(pred_inds))
    gt_inds = list(gt_inds) + list(set(range(max(N, K))) - set(gt_inds))

    indices = np.zeros([max(N, K)], dtype=int)
    for i in range(max(N, K)):
        indices[pred_inds[i]] = gt_inds[i]

    return reward, indices


def visualize(P, opt, expname, dataset, encoder, decoder, savedir, web_dir, show=True, refresh=False):

    device = torch.device(conf.device)
    
    with torch.no_grad():
        objects = []
        stats = Statistics()
        
        # I = list(range(len(dataset)))
        # I = sorted(random.sample(I, min(100, len(I))))

        for i, (uxid, o_gt) in enumerate(tqdm(dataset)):
            if i>200 and 'train' in opt.test_dataset:
                break
            
            o_gt = o_gt.to(device)
            # print(o_gt)
            root_code = encoder.encode_structure(obj=o_gt)
            
            if not conf.non_variational:
                z, obj_kldiv_loss = torch.chunk(root_code, 2, 1)
                obj_kldiv_loss = -obj_kldiv_loss.sum() 
            else:
                z = root_code
            
            #losses = decoder.structure_recon_loss(z=z, gt_tree=o_gt)    
            o = decoder.decode_structure(z=z, max_depth=conf.max_tree_depth)
            # print(o)
            # print('\n\===============================\n\n')
            
            o_flat, o_gt_flat  = o.copy() , o_gt.copy()
            o_flat.flatten()
            o_gt_flat.flatten()

            o_flat.root.children = sorted(o_flat.root.children, key=lambda x: x.box[0])
            #o.root.children = sorted(o.root.children, key=lambda x: int(x.label != 'Toolbar'))

            # IoU
            b_pred = np.array([x.box for x in o_flat.root.children])
            b_gt = np.array([x.box for x in o_gt_flat.root.children])
            IoU, I = max_linear_assignment(b_pred, b_gt, batch_iou)
            K = min(len(b_pred), len(b_gt))
            IoU = IoU / K if K > 0 else 0
            stats.add('IoU', IoU)
            
            # edit distance
            edit_distance = max(len(b_pred), len(b_gt)) - K
            for pred_ind, gt_ind in enumerate(I):
                if pred_ind >= len(b_pred) or gt_ind >= len(b_gt): continue
                c_pred = o_flat.root.children[pred_ind].label
                c_gt = o_gt_flat.root.children[gt_ind].label
                if c_pred != c_gt: edit_distance += 1
            stats.add('edit_distance', edit_distance)

            objects.append([uxid, o, o_gt, z, edit_distance, IoU])

        # statistics
        P.print('\n Expname: ', opt.expname)
        P.print('\n isleaf_threshold: ', opt.is_leaf_thres)
        P.print('\n isexist_threshold: ', opt.is_exists_thres)
        P.print(stats.to_string(verbose=True))

        # HTML visualize
        if len(objects) < 200:
            samples = objects
        else:
            samples = random.sample(objects, min(200, len(objects)))
            #samples = sorted(samples, key=lambda x: x[-1])
    
        
        split = 'train' if 'train' in opt.test_dataset else 'val'
        


        if opt.model_epoch is None:
            html = HTML(f'/recon_{split}@{expname}_isleaf_{conf.is_leaf_thres}_isexists_{conf.is_exists_thres}', expname, base_url=web_dir, inverted=True, overwrite=True, refresh=int(refresh))
        else:
            html = HTML(f'/recon_{split}@{expname}_epoch_{opt.model_epoch}_isleaf_{conf.is_leaf_thres}_isexists_{conf.is_exists_thres}', expname, base_url=web_dir, inverted=True, overwrite=True, refresh=int(refresh))
        html.add_table().add([vis_fn(*_) for _ in tqdm(samples)])
        html.save()

        domain = opt.domain if hasattr(opt, 'domain') else None
        if show: html.show(domain)
        else: P.print(html.url(domain))

        return stats

parser = argparse.ArgumentParser()
parser = add_eval_args(parser)
eval_conf = parser.parse_args() 

# Write here settings for debuging
#eval_conf.exp_name = 'rico_hier_exp_AE_sem_wt_1_nnemb'

print('\nExp_name: {eval_conf.exp_name}')
print('Split: {eval_conf.split}\n')

base_dataset_path = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_gen_data/rico_mtn_50_geq2_mcpn_10_V2/'
eval_conf.test_dataset = base_dataset_path + eval_conf.split + '_uxid.txt'

eval_conf.model_epoch = None
eval_conf.num_gen = 100
if 'mini' in eval_conf.test_dataset:
    eval_conf.web_dir = './www_minidataset'
else:
    eval_conf.web_dir = './www'

# load train config
conf = torch.load(os.path.join(eval_conf.model_path, eval_conf.exp_name, 'conf.pth'))
eval_conf.data_path = conf.data_path

# merge training and evaluation configurations, giving evaluation parameters precendence
conf.__dict__.update(eval_conf.__dict__)

# load object category information
if conf.semantics:
    Hierarchy.set_semantics(conf.semantics)
if conf.extract_hier:
    assert conf.semantics == 'rico_plus'

# load model
models = utils.get_model_module(conf.model_version)

# set up device
device = torch.device(conf.device)
print(f'Using device: {conf.device}')

# check if eval results already exist. If so, delete it. 
# if os.path.exists(os.path.join(conf.result_path, conf.exp_name)):
#     response = input('Eval results for "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
#     if response != 'y':
#         sys.exit()
#     shutil.rmtree(os.path.join(conf.result_path, conf.exp_name))

# create a new directory to store eval results
mkdir_if_missing(os.path.join(conf.result_path, conf.exp_name))
result_dir = os.path.join(conf.result_path, conf.exp_name)

# create models
encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=not conf.non_variational)
decoder = models.RecursiveDecoder(conf)
models = [encoder, decoder]
model_names = ['encoder', 'decoder']

print('\n\n')
#print(f'non_probabilistic: {conf.non_probabilistic}')
print(f'non_variational: {conf.non_variational}')

# load pretrained model
__ = utils.load_checkpoint(
    models=models, model_names=model_names,
    dirname=os.path.join(conf.model_path, conf.exp_name),
    epoch=conf.model_epoch,
    strict=True)


# send to device
for m in models:
    m.to(device)

# set models to evaluation mode
for m in models:
    m.eval()

global P
P = Printer(f'{os.path.join(conf.result_path, conf.exp_name)}/{conf.split}recon_isleaf_{conf.is_leaf_thres}_isexists_{conf.is_exists_thres}.log')


# create dataset and data loader
data_features = ['uxid', 'object']
DatasetClass = globals()[conf.DatasetClass] 
print('Using dataset:', DatasetClass) 

test_dataset = DatasetClass(conf.data_path, conf.test_dataset, ['uxid', 'object'],
                            is_train=False, permute=False, n_permutes=1)

#dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  collate_fn=lambda x: list(zip(*x)))

visualize(P, conf, conf.exp_name, test_dataset, encoder, decoder, result_dir, conf.web_dir, show=False)




