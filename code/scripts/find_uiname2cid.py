#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 13:59:37 2021

@author: dipu
"""
import json
import os, sys
import pickle
from collections import defaultdict

sys.path.append('../')
from rico import SemanticHierarchy, SemanticNode 

machine = 'amber' #'amber' # 'mypc' 
base_dir = '/home/dipu/dipu_ps' if machine =='amber' else '/vol/research/projectSpaceDipu'

ux_list_fn = '/home/dipu/dipu_ps/codes/UIGeneration/prj-ux-layout-copy/codes/scripts/rico_mtn_50_geq2_mcpn_10_V2.txt'
sui_path = f'{base_dir}/data/RICO/semantic_annotations/'

with open(ux_list_fn, 'r') as f:
      ux_list = [line.rstrip() for line in f]   
# initialize the Semantics of hierarchy
# SemanticHierarchy.set_semantics(semantics='rico_plus')

#initialize the name2cids
d = defaultdict(list)
w = defaultdict(list)

def parse_children(node):
    for child in node.children:
        d[node.label].append(child.label)
        if child.children !=[] or not child.is_leaf:
            parse_children(child)
    
    
for ind, ux_id in enumerate(ux_list):       
    hier_path = sui_path + ux_id + '.json'
    hier_json = json.load(open(hier_path))
    hier = SemanticHierarchy(hier_json, uxid=ux_id)
    hier.greedy_merge_on_rico_hier()
    if ind%5000 ==0:
        print(f'Completed: {ind}/{len(ux_list)}')
    
    for child in hier.root.children:
        parse_children(child)
    

from collections import Counter

child_freq= defaultdict(dict)
for key in d.keys():
    counter = Counter(d[key])
    child_freq[key] = counter
    del counter
    
save_fn = 'ui2childfreq.pkl'
pickle.dump (child_freq, open(save_fn, 'wb'))



#########################
# Manaul Selection
#########################



child_freq = pickle.load(open(save_fn, 'rb'))
uinames2child = defaultdict(dict)

for key in child_freq.keys():
    child_freq[key] = dict(sorted(child_freq[key].items(), key= lambda x:x[1], reverse=True))

for key in child_freq.keys():
    print (child_freq[key])
    print( child_freq[key].keys())    


# ['Modal', '[Merged-hca]', '[Merged-vca]', 'Toolbar', 'Card', 'Drawer', 'Advertisement', 'Web View', 
#   'List Item', 'Multi-Tab', 'Video', 'Image', 'On/Off Switch', 'Bottom Navigation', 'Date Picker', 
#  'Map View', 'Number Stepper', 'Icon', 'Input', 'Button Bar', 'Checkbox', 'Background Image', 
# 'Slider', 'Radio Button', 'Merged-vca']

rico_labels = ['Toolbar',    'Image',   'Web View',     'Icon',   'Text Button',     'Text', \
                'Multi-Tab',  'Card',     'List Item',       'Advertisement',   'Background Image', \
                'Drawer',     'Input',   'Bottom Navigation',  'Modal',          'Button Bar', \
                'Pager Indicator', 'On/Off Switch',  'Checkbox', 'Map View',  'Radio Button', \
                'Slider',  'Number Stepper',  'Video', 'Date Picker', '[Merged-vca]',  '[Merged-hca]'] 

uinames2child['[Merged-hca]'] = ['Text', 'Image', 'Text Button', 'Icon', 'Card', 'Input', '[Merged-vca]', 'On/Off Switch', 'List Item', 'Pager Indicator', 'Background Image', 'Web View', 'Slider', 'Toolbar', 'Radio Button', 'Advertisement', 'Checkbox', 'Map View', 'Multi-Tab', 'Video', 'Drawer', 'Bottom Navigation', 'Number Stepper', 'Button Bar', 'Date Picker']
uinames2child['[Merged-vca]'] = ['Text', 'List Item', 'Text Button', '[Merged-hca]', 'Image', 'Toolbar', 'Icon', 'Input', 'Advertisement', 'Card', 'Multi-Tab', 'Web View', 'Pager Indicator', 'Background Image', 'Radio Button', 'Slider', 'Map View', 'On/Off Switch', 'Drawer', 'Button Bar', 'Video', 'Checkbox', 'Date Picker', 'Bottom Navigation', 'Number Stepper']
uinames2child['Modal'] = ['Text', 'List Item', '[Merged-hca]', '[Merged-vca]', 'Text Button', 'Image', 'Icon', 'Button Bar', 'Date Picker', 'Input', 'Web View', 'Radio Button'] 
uinames2child['Toolbar'] = ['Icon', 'Text', 'Image', 'Text Button', '[Merged-hca]', 'Input']
uinames2child['Card'] = ['Text', 'Image', '[Merged-hca]', '[Merged-vca]', 'Icon', 'Text Button', 'Input', 'Card', 'List Item']
uinames2child['Drawer'] = ['List Item', '[Merged-vca]', 'Text', 'Text Button', '[Merged-hca]', 'Image', 'Icon']
uinames2child['Advertisement'] =  ['Web View', 'Advertisement']
uinames2child['Web View']  =  ['Web View']   
uinames2child['List Item']  = ['Text', 'Image', 'Icon', '[Merged-vca]', '[Merged-hca]', 'Text Button', 'Card', 'Checkbox', 'List Item', 'Input', 'On/Off Switch', 'Pager Indicator', 'Radio Button', 'Slider']
uinames2child['Multi-Tab'] = ['Text Button', 'Text', 'Icon', 'Image', '[Merged-vca]', '[Merged-hca]', 'Input', 'Radio Button']
uinames2child['Bottom Navigation'] =['Image', 'Icon', '[Merged-vca]', '[Merged-hca]', 'List Item', 'Text Button', 'Text']
uinames2child['Date Picker'] = ['Text Button', 'Number Stepper', '[Merged-hca]', '[Merged-vca]', 'Icon', 'Date Picker', 'Text', 'Image', 'Input']
uinames2child['Map View'] = ['Image', 'Map View', 'Icon', 'Text']
uinames2child['Number Stepper']= ['Input', 'Icon', 'Image', 'Text', 'Text Button']
uinames2child['Button Bar'] = ['Text Button']
uinames2child['Video'] = []
uinames2child['Image'] = []
uinames2child['On/Off Switch'] = []
uinames2child['Icon'] = []
uinames2child['Input'] = []
uinames2child['Checkbox'] = []
uinames2child['Background Image'] = []
uinames2child['Slider'] = []
uinames2child['Radio Button'] = []
uinames2child['Pager Indicator'] = []
uinames2child['Text'] = []
uinames2child['Text Button']= []






            






































        
