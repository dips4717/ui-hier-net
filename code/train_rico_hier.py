#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:32:04 2021

    Modified from train_box.py of structurenet
    This is the main trainer script for rico-hier structure AE/VAE experiments.
    Use scripts/train_rico_hier_ae.sh or scripts/train_rico_hier_vae.sh to run.
@author: dipu
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
from config_rico import add_train_vae_args
from data import PartNetDataset, Tree
from rico import Hierarchy
from datasets import RicoFlatDataset, RicoHierDataset
import utils
import time 
from moka import *

# Use 1-4 CPU threads to train.
# Don't use too many CPU threads, which will slow down the training.
torch.set_num_threads(2)

def train(conf):
    # load network model
    models = utils.get_model_module(conf.model_version)

    # check if training run already exists. If so, delete it.
    if os.path.exists(os.path.join(conf.log_path, conf.exp_name)) or \
       os.path.exists(os.path.join(conf.model_path, conf.exp_name)):
           print(f'A Training run named {conf.exp_name} already exists!!')
           sys.exit()
         # response = input('A training run named "%s" already exists, overwrite? (y/n) ' % (conf.exp_name))
         # if response != 'y':
             # sys.exit()
    if os.path.exists(os.path.join(conf.log_path, conf.exp_name)):
        shutil.rmtree(os.path.join(conf.log_path, conf.exp_name))
    if os.path.exists(os.path.join(conf.model_path, conf.exp_name)):
        shutil.rmtree(os.path.join(conf.model_path, conf.exp_name))

    # create directories for this run
    os.makedirs(os.path.join(conf.model_path, conf.exp_name))
    os.makedirs(os.path.join(conf.log_path, conf.exp_name))

    # file log
    flog = open(os.path.join(conf.log_path, conf.exp_name, 'train.log'), 'w')

    # set training device
    device = torch.device(conf.device)
    print(f'Using device: {conf.device}')
    flog.write(f'Using device: {conf.device}\n')

    # log the object category information
    print(f'Object Category: {conf.category}')
    flog.write(f'Object Category: {conf.category}\n')

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    print("Random Seed: %d" % (conf.seed))
    flog.write(f'Random Seed: {conf.seed}\n')
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.model_path, conf.exp_name, 'conf.pth'))
    print('\n\n Configuration:\n', vars(conf), '\n\n')

    # create models
    encoder = models.RecursiveEncoder(conf, variational= not conf.non_variational, probabilistic=not conf.non_probabilistic)
    decoder = models.RecursiveDecoder(conf)
    models = [encoder, decoder]
    model_names = ['encoder', 'decoder']

    # create optimizers
    encoder_opt = torch.optim.Adam(encoder.parameters(), lr=conf.lr)
    decoder_opt = torch.optim.Adam(decoder.parameters(), lr=conf.lr)
    optimizers = [encoder_opt, decoder_opt]
    optimizer_names = ['encoder', 'decoder']

    # learning rate scheduler
    if conf.scheduler == 'StepLR':
        encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_opt, \
                step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
        decoder_scheduler = torch.optim.lr_scheduler.StepLR(decoder_opt, \
                step_size=conf.lr_decay_every, gamma=conf.lr_decay_by)
    elif conf.scheduler == 'ReduceLROnPlateau':
        encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_opt, mode='min', \
                factor=conf.lr_decay_factor, patience=2)
        decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_opt, mode='min',\
                factor=conf.lr_decay_factor, patience=2)

    # create training and validation datasets and data loaders
    
    
    data_features = ['uxid', 'object']
    # train_dataset = PartNetDataset(conf.data_path, conf.train_dataset, data_features, \
    #        load_geo=conf.load_geo)
    # valdt_dataset = PartNetDataset(conf.data_path, conf.val_dataset, data_features, \
    #        load_geo=conf.load_geo)
    
    DatasetClass = globals()[config.DatasetClass] 
    print('Using dataset:', DatasetClass)   
    train_dataset = DatasetClass(conf.data_path, conf.train_dataset, ['uxid', 'object'],
                                is_train=True, permute=(conf.permutations > 1), n_permutes=(conf.permutations)                                )
    
    valdt_dataset = DatasetClass(conf.data_path, config.val_dataset, ['uxid', 'object'],
                                is_train=False, permute=False, n_permutes=1)
        
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, \
            shuffle=True, collate_fn=lambda x: list(zip(*x)))
    valdt_dataloader = torch.utils.data.DataLoader(valdt_dataset, batch_size=conf.batch_size, \
            shuffle=True, collate_fn=lambda x: list(zip(*x)))

    # create logs
    if not conf.no_console_log:    
        #header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR       BoxLoss   StructLoss   EdgeExists  KLDivLoss   SymLoss    AdjLoss  AnchorLoss  TotalLoss'
        header = '     Time    Epoch     Dataset    Iteration     Progress(%)       LR       BoxLoss   LeafLoss     NodeExists      SemLoss   ChildCountLoss    KLDivLoss     TotalLoss'
    
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'train'))
        valdt_writer = SummaryWriter(os.path.join(conf.log_path, conf.exp_name, 'val'))

    # send parameters to device
    for m in models:
        m.to(device)
    for o in optimizers:
        utils.optimizer_to_device(o, device)

    # start training
    print("Starting training ...... ")
    flog.write('Starting training ......\n')

    start_time = time.time()

    last_checkpoint_step = None
    last_train_console_log_step, last_valdt_console_log_step = None, None
    train_num_batch, valdt_num_batch = len(train_dataloader), len(valdt_dataloader)

    # train for every epoch
    for epoch in range(conf.epochs):
        global train_stats
        global valdt_stats
        
        train_stats = Statistics()
        valdt_stats = Statistics()

        if not conf.no_console_log:
            print(f'training run {conf.exp_name}')
            flog.write(f'training run {conf.exp_name}\n')
            print(header)
            flog.write(header+'\n')

        train_batches = enumerate(train_dataloader, 0)
        valdt_batches = enumerate(valdt_dataloader, 0)

        train_fraction_done, valdt_fraction_done = 0.0, 0.0
        valdt_batch_ind = -1

        # train for every batch
        for train_batch_ind, batch in train_batches:
            #tic_compl_batch = time.time()

            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or \
                    train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # set models to training mode
            for m in models:
                m.train()

            # forward pass (including logging)
            total_loss = forward(
                batch=batch, data_features=data_features, encoder=encoder, decoder=decoder, device=device, conf=conf,
                is_valdt=False, step=train_step, epoch=epoch, batch_ind=train_batch_ind, num_batch=train_num_batch, start_time=start_time,
                log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=train_writer,
                lr=encoder_opt.param_groups[0]['lr'], flog=flog)

            # optimize one step
            encoder_opt.zero_grad()
            decoder_opt.zero_grad()
            total_loss.backward()
            encoder_opt.step()
            decoder_opt.step()

            train_stats.add('total_loss', float(total_loss.item()), len(batch[0]) )
            del total_loss

            # save checkpoint
            with torch.no_grad():
                if last_checkpoint_step is None or \
                        train_step - last_checkpoint_step >= conf.checkpoint_interval:
                    print("Saving checkpoint ...... ", end='', flush=True)
                    flog.write("Saving checkpoint ...... ")
                    utils.save_checkpoint(
                        models=models, model_names=model_names, dirname=os.path.join(conf.model_path, conf.exp_name),
                        epoch=epoch, prepend_epoch=True, optimizers=optimizers, optimizer_names=model_names)
                    print("DONE")
                    flog.write("DONE\n")
                    last_checkpoint_step = train_step

            # validate one batch
            # while valdt_fraction_done <= train_fraction_done and valdt_batch_ind+1 < valdt_num_batch:
            #     valdt_batch_ind, batch = next(valdt_batches)

            #     valdt_fraction_done = (valdt_batch_ind + 1) / valdt_num_batch
            #     valdt_step = (epoch + valdt_fraction_done) * train_num_batch - 1

            #     log_console = not conf.no_console_log and (last_valdt_console_log_step is None or \
            #             valdt_step - last_valdt_console_log_step >= conf.console_log_interval)
            #     if log_console:
            #         last_valdt_console_log_step = valdt_step

            #     # set models to evaluation mode
            #     for m in models:
            #         m.eval()

            #     with torch.no_grad():
            #         # forward pass (including logging)
            #         __ = forward(
            #             batch=batch, data_features=data_features, encoder=encoder, decoder=decoder, device=device, conf=conf,
            #             is_valdt=True, step=valdt_step, epoch=epoch, batch_ind=valdt_batch_ind, num_batch=valdt_num_batch, start_time=start_time,
            #             log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=valdt_writer,
            #             lr=encoder_opt.param_groups[0]['lr'], flog=flog)

        # Validate in every batch 
        with torch.no_grad():
            for valdt_batch_ind , batch in valdt_batches:
                valdt_fraction_done = (valdt_batch_ind + 1) / valdt_num_batch
                valdt_step = epoch * valdt_num_batch + valdt_batch_ind
    
                log_console = not conf.no_console_log and (last_valdt_console_log_step is None or \
                        valdt_step - last_valdt_console_log_step >= conf.console_log_interval)
                if log_console:
                    last_valdt_console_log_step = valdt_step
    
                # set models to training mode
                for m in models:
                    m.eval()
    
                # forward pass (including logging)
    
                total_loss = forward(
                    batch=batch, data_features=data_features, encoder=encoder, decoder=decoder, device=device, conf=conf,
                    is_valdt=True, step=valdt_step, epoch=epoch, batch_ind=valdt_batch_ind, num_batch=valdt_num_batch, start_time=start_time,
                    log_console=log_console, log_tb=not conf.no_tb_log, tb_writer=valdt_writer,
                    lr=encoder_opt.param_groups[0]['lr'], flog=flog)

                valdt_stats.add('total_loss', float(total_loss.item()), len(batch[0]) ) 

                    
        valid_loss = valdt_stats.mean(conf.metric)    
        
        if conf.scheduler == 'StepLR':
            encoder_scheduler.step()
            decoder_scheduler.step()
        elif conf.scheduler == 'ReduceLROnPlateau':
            encoder_scheduler.step(valid_loss)
            decoder_scheduler.step(valid_loss)
            #print(f'1 complete batch update, Elsped time: {time.time()-tic_compl_batch:.2f}')
    
    # save the final models
    print("Saving final checkpoint ...... ", end='', flush=True)
    flog.write("Saving final checkpoint ...... ")
    utils.save_checkpoint(
        models=models, model_names=model_names, dirname=os.path.join(conf.model_path, conf.exp_name),
        epoch=epoch, prepend_epoch=False, optimizers=optimizers, optimizer_names=optimizer_names)
    print("DONE")
    flog.write("DONE\n")

    flog.close()

def forward(batch, data_features, encoder, decoder, device, conf,
            is_valdt=False, step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, flog=None):
    
    objects = batch[data_features.index('object')]

    #tic_batch = time.time()

    losses = {
        'box': torch.zeros(1, device=device),
        'leaf': torch.zeros(1, device=device),
        'exists': torch.zeros(1, device=device),
        'semantic': torch.zeros(1, device=device),
        'kldiv': torch.zeros(1, device=device), 
        'child_count': torch.zeros(1, device=device)}

    # 'anchor': torch.zeros(1, device=device),
    # 'edge_exists': torch.zeros(1, device=device),
    # 'sym': torch.zeros(1, device=device),
    # 'adj': torch.zeros(1, device=device)
    # process every data in the batch individually
    

    for obj in objects:
        obj.to(device)
        tic_obj = time.time()

        # encode object to get root code
        root_code = encoder.encode_structure(obj=obj)

        #toc_obj_enc = time.time() - tic_obj
        # print(f'Time elapsed for 1 object encoding = {toc_obj_enc} sec')

        # get kldiv loss
        if not conf.non_variational and not conf.non_probabilistic:
            root_code, obj_kldiv_loss = torch.chunk(root_code, 2, 1)
            obj_kldiv_loss = -obj_kldiv_loss.sum() # negative kldiv, sum over feature dimensions
            losses['kldiv'] = losses['kldiv'] + obj_kldiv_loss

        # decode root code to get reconstruction loss
        obj_losses = decoder.structure_recon_loss(z=root_code, gt_tree=obj)
        # toc_obj_encdec = time.time() - tic_obj 
        # print(f'Time elapsed for 1 object encoding + decoding = {toc_obj_encdec} sec')

        
        for loss_name, loss in obj_losses.items():
            losses[loss_name] = losses[loss_name] + loss
    #print(f'\n\n Completed 1 batch. Elsp time: {time.time() - tic_batch:.2f} \n\n' )
    
    for loss_name in losses.keys():
        losses[loss_name] = losses[loss_name] / len(objects)

    losses['box'] *= conf.loss_weight_box
    losses['leaf'] *= conf.loss_weight_leaf
    losses['exists'] *= conf.loss_weight_exists
    losses['semantic'] *= conf.loss_weight_semantic 
    losses['kldiv'] *= conf.loss_weight_kldiv
    losses['child_count'] *= conf.loss_weight_childcount
    
    # losses['edge_exists'] *= conf.loss_weight_edge_exists
    # losses['anchor'] *= conf.loss_weight_anchor
    # losses['sym'] *= conf.loss_weight_sym
    # losses['adj'] *= conf.loss_weight_adj

    total_loss = 0
    for loss in losses.values():
        total_loss += loss

    with torch.no_grad():
        # log to console
        if log_console:
            print(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{losses['box'].item():>11.2f} '''
                f'''{losses['leaf'].item():>11.2f} '''
                f'''{losses['exists'].item():>11.2f} '''
                f'''{losses['semantic'].item():>11.2f} '''
                f'''{losses['child_count'].item():>11.2f} '''
                f'''{losses['kldiv'].item():>10.2f} '''
                f'''{total_loss.item():>10.2f}''')
            flog.write(
                f'''{strftime("%H:%M:%S", time.gmtime(time.time()-start_time)):>9s} '''
                f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                f'''{'validation' if is_valdt else 'training':^10s} '''
                f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                f'''{100. * (1+batch_ind+num_batch*epoch) / (num_batch*conf.epochs):>9.1f}%      '''
                f'''{lr:>5.2E} '''
                f'''{losses['box'].item():>11.2f} '''
                f'''{losses['leaf'].item():>11.2f} '''
                f'''{losses['exists'].item():>11.2f} '''
                f'''{losses['semantic'].item():>11.2f} '''    
                f'''{losses['child_count'].item():>11.2f} '''            
                f'''{losses['kldiv'].item():>10.2f} '''
                f'''{total_loss.item():>10.2f}\n''')
            flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            tb_writer.add_scalar('loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)
            tb_writer.add_scalar('box_loss', losses['box'].item(), step)
            #tb_writer.add_scalar('anchor_loss', losses['anchor'].item(), step)
            tb_writer.add_scalar('leaf_loss', losses['leaf'].item(), step)
           # tb_writer.add_scalar('exists_loss', losses['exists'].item(), step)
            tb_writer.add_scalar('semantic_loss', losses['semantic'].item(), step)
            tb_writer.add_scalar('child_count_loss', losses['child_count'].item(), step)

           # tb_writer.add_scalar('edge_exists_loss', losses['edge_exists'].item(), step)
            tb_writer.add_scalar('kldiv_loss', losses['kldiv'].item(), step)
           # tb_writer.add_scalar('sym_loss', losses['sym'].item(), step)
           # tb_writer.add_scalar('adj_loss', losses['adj'].item(), step)

    return total_loss

if __name__ == '__main__':
    sys.setrecursionlimit(5000) # this code uses recursion a lot for code simplicity

    parser = ArgumentParser()
    parser = add_train_vae_args(parser)
    config = parser.parse_args()
    
    #Set parameters for debugging:
    # config.exp_name =  'rico_hier_vae' 
    config.category = 'Rico' 
    config.semantics= 'rico_plus'
    config.data_path = '/home/dipu/dipu_ps/codes/UIGeneration/data'
    config.train_dataset = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_gen_data/rico_mtn_50_geq2_mcpn_10_V2/train_uxid.txt'
    config.val_dataset = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_gen_data/rico_mtn_50_geq2_mcpn_10_V2/val_uxid.txt' 
    # config.train_dataset = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_max_node49_dataset.txt'
    # config.val_dataset = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_max_node49_dataset.txt'

    
    config.epochs = 40 
    config.model_version = 'model_rico_hier' 
    config.extract_hier = True
    config.loss_for_box = 'L2'
    #config.exp_name = 'rico_hier_timing'

    # Tree.load_category_info(config.category)
    
    # semantics
    if config.semantics:
        Hierarchy.set_semantics(config.semantics)
    if config.extract_hier:
        assert config.semantics == 'rico_plus'
   
    train(config)