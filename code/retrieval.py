#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:00:19 2021

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
from utils import mkdir_if_missing
from scipy.spatial.distance import cdist


def vis_fn(q_uxid, q_o, r1_id, r1_o, r2_id, r2_o, r3_id, r3_o, r4_id, r4_o, r5_id, r5_o):
    return dict(
        q_id = q_uxid,
        query = q_o.to_string(render='html', labeled=True),
        query_layout = q_o.plot(),
        
        rank1_id = r1_id,
        rank1 = r1_o.to_string(render='html', labeled=r1_o.is_labeled),
        rank1_layout = r1_o.plot(),
        
        rank2_id = r2_id,
        rank2 = r2_o.to_string(render='html', labeled=r2_o.is_labeled),
        rank2_layout = r2_o.plot(),
        
        rank3_id = r3_id,
        rank3 = r3_o.to_string(render='html', labeled=r3_o.is_labeled),
        rank3_layout = r3_o.plot(),
        
        rank4_id = r4_id,
        rank4 = r4_o.to_string(render='html', labeled=r4_o.is_labeled),
        rank4_layout = r4_o.plot(),
        
        rank5_id = r5_id,
        rank5 = r5_o.to_string(render='html', labeled=r5_o.is_labeled),
        rank5_layout = r5_o.plot()     
        
    )

def test_vis_fn(q_uxid, q_o, r1_id, r1_o, r2_id, r2_o, r3_id, r3_o, r4_id, r4_o, r5_id, r5_o):
    aa = [q_uxid, q_o, r1_id, r1_o, r2_id, r2_o, r3_id, r3_o, r4_id, r4_o, r5_id, r5_o]
    return aa

def extract_features(conf, dataset, encoder):
    device = torch.device(conf.device)
    
    with torch.no_grad():
        objects = []
        
        for i, (uxid, o_gt) in enumerate(tqdm(dataset)):
            o_gt = o_gt.to(device)
            root_code = encoder.encode_structure(obj=o_gt)
            
            if not conf.non_variational:
                z, obj_kldiv_loss = torch.chunk(root_code, 2, 1)
            else:
                z = root_code
            z = z.detach().cpu().numpy()
            
            objects.append([uxid, o_gt, z])
      
    return objects
                
            

def main():
    parser = argparse.ArgumentParser()
    parser = add_eval_args(parser)
    eval_conf = parser.parse_args() 
    
    # Write here settings for debuging
    eval_conf.category = 'rico'
    eval_conf.exp_name = 'rico_hier_exp_AE_sem_wt_1_nnemb'
    eval_conf.semantics = 'rico_plus'
    eval_conf.test_dataset = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_gen_data/rico_mtn_50_geq2_mcpn_10_V2/train_uxid.txt'
    eval_conf.model_epoch = None
    eval_conf.num_gen = 100
    eval_conf.web_dir = './www'
    eval_conf.semantic_representation = 'nn_embedding'
    
    
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
    # result_dir = os.path.join(conf.result_path, conf.exp_name)
    # mkdir_if_missing()
    # os.makedirs(os.path.join(conf.result_path, conf.exp_name))
    # result_dir = os.path.join(conf.result_path, conf.exp_name)
    
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
    
    
    # create dataset and data loader
    data_features = ['uxid', 'object']
    DatasetClass = globals()[conf.DatasetClass] 
    print('Using dataset:', DatasetClass) 
    
    test_dataset = DatasetClass(conf.data_path, conf.test_dataset, ['uxid', 'object'],
                                is_train=False, permute=False, n_permutes=1)
    
    #dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,  collate_fn=lambda x: list(zip(*x)))
    
    # visualize(P, conf, conf.exp_name, test_dataset, encoder, decoder, result_dir, conf.web_dir, show=False)
    
    feats_objects =  extract_features(conf, test_dataset, encoder)   
    
    feats = np.concatenate([x[-1] for x in feats_objects])
    uxids = [x[0] for x in feats_objects]
    hiers = [x[1] for x in feats_objects]
    uxid2hier = dict((k,v) for k,v in zip(uxids, hiers)) 
    

    distances = cdist(feats, feats, metric= 'euclidean')
    sort_inds = np.argsort(distances)   
    
    sample_retrievals = []
    for ii in range(100):
        q_uxid = uxids[ii]
        
        ranked_uxids = []
        ranked_hiers = []
        for yy in sort_inds[ii,:5]:
            ranked_uxids.append(uxids[yy])
            ranked_hiers.append(uxid2hier[uxids[yy]])
        
        # ranked_uxids = [uxids[yy] for yy in sort_inds[ii,:5]]
        # ranked_hiers = [uxid2hier[id] for id in ranked_uxids ]
        
        ranked = [None] * (len(ranked_uxids) + len(ranked_hiers))
        ranked[::2] = ranked_uxids
        ranked[1::2] = ranked_hiers
        
        sample_retrievals.append([q_uxid, uxid2hier[q_uxid]] + ranked)
        
    visualize_retrieved_images(conf, sample_retrievals, web_dir = 'www', show=False )
    
    
def visualize_retrieved_images(conf, sample_retrievals, web_dir='www', show=False, refresh=False):
    split = 'train' if 'train' in conf.test_dataset else 'val'
        
    if conf.model_epoch is None:
        html = HTML(f'/retrieval_{split}@{conf.exp_name}', conf.exp_name, base_url=web_dir, inverted=True, overwrite=True, refresh=int(refresh))
    else:
        html = HTML(f'/retrieval_{split}@{conf.exp_name}_epoch_{conf.model_epoch}', conf.expname, base_url=web_dir, inverted=True, overwrite=True, refresh=int(refresh))
    
    html.add_table().add([vis_fn(*_) for _ in tqdm(sample_retrievals)])
    html.save()

    domain = conf.domain if hasattr(conf, 'domain') else None
    if show: html.show(domain)
    #else: P.print(html.url(domain))    
    
if __name__ == '__main__':
    main()