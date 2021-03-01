import sys
sys.path.append('/home/dipu/codes/GraphEncoding-RICO/')

import torch
from torchvision import transforms
import pickle
from scipy.spatial.distance import cdist
import numpy as np
import init_paths
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
import torch.nn.functional as F

from eval_metrics.get_overall_IOU import get_overall_IOU
from eval_metrics.get_overall_Classwise_IOU import get_overall_Classwise_IOU
from eval_metrics.get_overall_pix_acc import get_overall_pix_acc
from eval_metrics.get_overall_ClasswiseIou_ndcg import get_overall_ClasswiseIou_ndcg
from eval_metrics.get_overall_PixAcc_ndcg import get_overall_PixAcc_ndcg 

from rico import *
from utils_structurenet import *
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
from rico import Hierarchy
from datasets import RicoFlatDataset, RicoHierDataset
import utils_structurenet
import time 

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

def visualize_retrieved_images(conf, sample_retrievals, web_dir='www', show=False, refresh=False):
    split = 'train' if 'train' in conf.test_dataset else 'val'
        
    if conf.model_epoch is None:
        html = HTML(f'/ECCV_retrieval@{conf.exp_name}', conf.exp_name, base_url=web_dir, inverted=True, overwrite=True, refresh=int(refresh))
    else:
        html = HTML(f'/ECCV_retrieval@{conf.exp_name}_epoch_{conf.model_epoch}', conf.expname, base_url=web_dir, inverted=True, overwrite=True, refresh=int(refresh))
    
    html.add_table().add([vis_fn(*_) for _ in tqdm(sample_retrievals)])
    html.save()

    domain = conf.domain if hasattr(conf, 'domain') else None
    if show: html.show(domain)

def extract_features(conf, dataset, encoder):
    device = torch.device(conf.device)
    
    with torch.no_grad():
        objects = []
        
        for i, (uxid, o_gt) in enumerate(tqdm(dataset)):
            o_gt = o_gt.to(device)
            o_gt.check_valid_for_dataset(node=o_gt.root, max_child_per_node=10)
            all_nodes = o_gt.dfs()
            
            if not o_gt.is_valid or len(all_nodes) >50:
                print(uxid)
                continue
             
            root_code = encoder.encode_structure(obj=o_gt)
            
            if not conf.non_variational:
                z, obj_kldiv_loss = torch.chunk(root_code, 2, 1)
            else:
                z = root_code
            z = z.detach().cpu().numpy()
            
            objects.append([uxid, o_gt, z])
    print(f'Extracted feature from {len(objects)} UXs')  
    return objects



def main():
    parser = argparse.ArgumentParser()
    parser = add_eval_args(parser)
    eval_conf = parser.parse_args() 
    
    #arges for dedug mode
    # eval_conf.exp_name =  'rico_hier_AE_SemWt1_nnemb_nonGNN2' 
    # eval_conf.device = 'cuda:2'
    
    base_dataset_path = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_gen_data/rico_mtn_50_geq2_mcpn_10_V2/'
    eval_conf.test_dataset = base_dataset_path + eval_conf.split + '_uxid.txt'

    #data_dir = '/mnt/amber/scratch/Dipu/RICO/semantic_annotations/'
    boundingBoxes = getBoundingBoxes_from_info_list(info_file = '/home/dipu/codes/GraphEncoding-RICO/data/rico_box_info_list.pkl') 

 
    # eval_conf.model_epoch = None
    eval_conf.num_gen = 100
    eval_conf.web_dir = './www'

    # load train config
    conf = torch.load(os.path.join(eval_conf.model_path, eval_conf.exp_name, 'conf.pth'))
    if not hasattr(conf, 'semantic_representation'): conf.semantic_representation = 'one_hot'
    if not hasattr(conf, 'intermediate_box_encoding'): conf.intermediate_box_encoding = False
    if not hasattr(conf, 'encode_child_count'): conf.encode_child_count = False
    if not hasattr(conf, 'non_gnn'): conf.non_gnn = False

    eval_conf.data_path = conf.data_path

    print(f'\nExp_name: {eval_conf.exp_name}')
    print(f'Split: {eval_conf.split}')
    print(f'Using device: {conf.device}')
    print(f'non_probabilistic: {conf.non_probabilistic}')
    print(f'non_variational: {conf.non_variational}\n')

    # merge training and evaluation configurations, giving evaluation parameters precendence
    conf.__dict__.update(eval_conf.__dict__)

    # load object category information
    if conf.semantics:
        Hierarchy.set_semantics(conf.semantics)
    if conf.extract_hier:
        assert conf.semantics == 'rico_plus'

    # Log file     
    global P
    if conf.model_epoch is None:
        P = Printer(f'{os.path.join(conf.result_path, conf.exp_name)}/ ECCV_ret_recon_isleaf_{conf.is_leaf_thres}_isexists_{conf.is_exists_thres}.log')
    else:
        P = Printer(f'{os.path.join(conf.result_path, conf.exp_name)}/ ECCV_ret_epoch_{conf.model_epoch}_recon_isleaf_{conf.is_leaf_thres}_isexists_{conf.is_exists_thres}.log')

    # load model
    models = utils_structurenet.get_model_module(conf.model_version)

    # set up device
    device = torch.device(conf.device)

    # create a new directory to store eval results
    mkdir_if_missing(os.path.join(conf.result_path, conf.exp_name))
    result_dir = os.path.join(conf.result_path, conf.exp_name)

    # create models
    encoder = models.RecursiveEncoder(conf, variational=True, probabilistic=not conf.non_variational)
    decoder = models.RecursiveDecoder(conf)
    models = [encoder, decoder]
    model_names = ['encoder', 'decoder']

    # load pretrained model
    __ = utils_structurenet.load_checkpoint(
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
    DatasetClass = globals()[conf.DatasetClass] 
    print('Using dataset:', DatasetClass) 

    query_path = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/eccv_split/query_uxid_revisted.txt'
    gallery_path = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/eccv_split/gallery_uxid_revisted.txt'

    query_dataset = DatasetClass(conf.data_path, query_path, ['uxid', 'object'],
                                is_train=False, permute=False, n_permutes=1)    

    gallery_dataset = DatasetClass(conf.data_path, gallery_path, ['uxid', 'object'],
                                is_train=False, permute=False, n_permutes=1)   

    q_feats_objects =  extract_features(conf, query_dataset, encoder)   
    g_feats_objects =  extract_features(conf, gallery_dataset, encoder) 

    g_feats = np.concatenate([x[-1] for x in g_feats_objects])
    q_feats = np.concatenate([x[-1] for x in q_feats_objects])

    g_uxids = [x[0] for x in g_feats_objects]
    g_hiers = [x[1] for x in g_feats_objects]
    guxid2hier = dict((k,v) for k,v in zip(g_uxids, g_hiers)) 

    q_uxids = [x[0] for x in q_feats_objects]
    q_hiers = [x[1] for x in q_feats_objects]
    quxid2hier = dict((k,v) for k,v in zip(q_uxids, q_hiers)) 
    
    q_fnames = [str(x) for x in q_uxids]
    g_fnames = [str(x) for x in g_uxids]

    distances = cdist(q_feats, g_feats, metric= 'euclidean')
    sort_inds = np.argsort(distances)

    overallMeanClassIou, overallMeanWeightedClassIou, classwiseClassIoU = get_overall_Classwise_IOU(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])
    overallMeanAvgPixAcc, overallMeanWeightedPixAcc, classPixAcc = get_overall_pix_acc(boundingBoxes,sort_inds,g_fnames,q_fnames, topk = [1,5,10])     
    
    per_query_metrics = {'IoU': classwiseClassIoU, 
                'PixAcc': classPixAcc}

    P.print(conf.exp_name)    
    P.print('The overallMeanClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanClassIou]) + '\n')        
    P.print('The overallMeanWeightedClassIou =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedClassIou]) + '\n')
    P.print('The overallMeanAvgPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanAvgPixAcc]) + '\n')
    P.print('The overallMeanWeightedPixAcc =  ' + str([ '{:.3f}'.format(x) for x in overallMeanWeightedPixAcc]) + '\n')

    sample_retrievals = []
    for ii , q_uxid in enumerate(q_uxids):
        
        ranked_uxids = []
        ranked_hiers = []
        for yy in sort_inds[ii,:5]:
            ranked_uxids.append(g_uxids[yy])
            ranked_hiers.append(guxid2hier[g_uxids[yy]])
        
        ranked = [None] * (len(ranked_uxids) + len(ranked_hiers))
        ranked[::2] = ranked_uxids
        ranked[1::2] = ranked_hiers

        sample_retrievals.append([q_uxid, quxid2hier[q_uxid]] + ranked)

    visualize_retrieved_images(conf, sample_retrievals, web_dir = 'www', show=False )
              

 

def getBoundingBoxes_from_info_list(info_file = 'data/rico_box_info_list.pkl'):
    allBoundingBoxes = BoundingBoxes()
    info = pickle.load(open(info_file, 'rb'))
    
    for yy in range(len(info)):
        
        count = info[yy]['nComponent']
        imageName = info[yy]['id']
        for i in range(count):
            box = info[yy]['xywh'][i]
            bb = BoundingBox(
                imageName,
                info[yy]['componentLabel'][i],
                box[0],
                box[1],
                box[2],
                box[3],
                iconClass=info[yy]['iconClass'][i],
                textButtonClass=info[yy]['textButtonClass'][i])
            allBoundingBoxes.addBoundingBox(bb) 
    print('Collected {} bounding boxes from {} images'. format(allBoundingBoxes.count(), len(info) ))         
#    testBoundingBoxes(allBoundingBoxes)
    return allBoundingBoxes
       

if __name__ == '__main__':
    main()


#Extracted feature from 10453 UXs
#Completed computing Classwise IoU: 43/43
#Completed computing Pixel Accuracies: 43/43
#Feb-26-21@02:58:35        rico_hier_AE_SemWt1_nnemb_nonGNN2
#Feb-26-21@02:58:35        The overallMeanClassIou =  ['0.470', '0.403', '0.371']

#Feb-26-21@02:58:35        The overallMeanWeightedClassIou =  ['0.545', '0.490', '0.453']

#Feb-26-21@02:58:35        The overallMeanAvgPixAcc =  ['0.554', '0.481', '0.445']

#Feb-26-21@02:58:35        The overallMeanWeightedPixAcc =  ['0.617', '0.555', '0.509']