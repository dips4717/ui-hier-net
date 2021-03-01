#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 21:36:28 2021

@author: dipu
"""

from argparse import ArgumentParser

def main():
    parser = ArgumentParser('Test for the bash.sh script for exexuting python programs with for args')
    parser.add_argument('--exp_name', type = str, default = 'noname')
    parser.add_argument('--split', type = str, default = 'test')
    parser.add_argument('--model_epoch', type=int, default=1)
    
    args = parser.parse_args()
    
    print (f'Exp name: {args.exp_name}\t split: {args.split}\t model_epoch: {args.model_epoch}\n')
    
if __name__ == '__main__':
    main()
    
    