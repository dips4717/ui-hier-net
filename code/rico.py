

import argparse
import random
import os, sys
import json, math
import itertools
from glob import glob
from queue import Queue
from copy import deepcopy
from fnmatch import fnmatch
from functools import cmp_to_key
from collections import defaultdict, OrderedDict, Counter

import cv2
import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import termcolor

import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from moka import *

import networkx 
from networkx.algorithms.components.connected import connected_components
"""https://stackoverflow.com/questions/4842613/merge-lists-that-share-common-elements"""



class Palette(object):

    DEFAULT_COLOR = (128, 128, 128)

    def __init__(self, n=-1):
        self.palette = OrderedDict()
        if n > 0:
            self.rainbow = matplotlib.cm.ScalarMappable(
                norm = matplotlib.colors.Normalize(vmin=0, vmax=n - 1),
                cmap = plt.get_cmap('gist_rainbow')
            )
            for label in range(n):
                color = self.rainbow.to_rgba(label)[:3]
                self.palette[str(label)] = np.array(color) * 255

    def get_colors(self):
        return list(self.palette.values())

    def get_labels(self):
        return list(self.palette.keys())

    def __call__(self, label):
        """"BGR color"""
        return self.palette.get(label, self.DEFAULT_COLOR)[::-1]

    def legend(self):
        assert len(self.get_colors()) == len(self.get_labels())
        fig = plt.figure(dpi=80)
        ax = fig.add_subplot(111)
        ax.set_color_cycle([tuple(c/255.0) + (1.0,) for c in self.get_colors()])

        for label in self.get_labels():
            ax.plot(0, 0, lw=10, label=label)

        ax.set_axis_off()
        return ax.legend(fontsize="x-large", loc='upper left')

class RandomPalette(Palette):
    def __init__(self):
        super(RandomPalette, self).__init__()

    def __call__(self, label):
        if label in self.palette:
            return self.palette[label]
        else:
            color = (np.random.rand(3) * 255)
            self.palette[label] = color
            return color


class RainbowPalette(Palette):
    # sorted by # of UIs that the label appears in, top 15
    ALL_LABELS = ['[ROOT]', 'Text', 'Icon', 'Image', 'Text Button', 'Toolbar', 'List Item', 'Web View', 'Advertisement',
              'Input', 'Drawer', 'Background Image', 'Card', 'Multi-Tab', 'Modal', 'Pager Indicator']

    def __init__(self, labels=None):
        super(RainbowPalette, self).__init__()

        if labels is not None:
            self.LABELS = list(OrderedDict.fromkeys(labels))
        else:
            self.LABELS = self.ALL_LABELS

        assert self.LABELS[0] == '[ROOT]'
        self.NUM_COLORS = len(self.LABELS) - 1
        self.LABEL2ID = {label: i for i, label in enumerate(self.LABELS)}

        self.rainbow = matplotlib.cm.ScalarMappable(
            norm = matplotlib.colors.Normalize(vmin=0, vmax=self.NUM_COLORS - 1),
            cmap = plt.get_cmap('gist_rainbow')
        )
        # map for all except for [ROOT]
        for label in self.LABELS[1:]:
            color = self.rainbow.to_rgba(self.LABEL2ID[label] - 1)[:3]
            self.palette[label] = np.array(color) * 255


class RicoPalette(Palette):
    """For semantic labels."""

    def __init__(self, labels):
        self.HEX_PALETTE = {
            'Toolbar': '#52D0DF',
            'Text': '#4B0A88',
            'Image': '#EF5E92',
            'Icon': '#EF5E92',
            'Multi-Tab': '#FEF284',
            'List Item': '#FEE0B6',
            'Web View': '#48A4F0',
            'Advertisement': '#48A4F0',
            'Modal': '#1FFFCE',
            'Button Bar': '#FECCD3',
            'Date Picker': '#CB6398',
            'Icon': '#CCDD52',
            'Video': '#0ECE32',
            'Card': '#D7BCE0',
            'Checkbox': '#FD896C',
            'Input': '#92CAF6',
            'Drawer': '#6835B2',
            'Background Image': '#D10155',
            'Number Stepper': '#AED688',
            'Map View': '#E1BDE5',
            'On/Off Switch': '#54C3F3',
            'Slider': '#CDDBDC',
            'Radio Button': '#FEB75D',
            'Pager Indicator': '#F7BAD0',
            'Bottom Navigation': '#BA65C4',
            'Text Button': '#522E1E',
            '[ROOT]': rgb2hex(self.DEFAULT_COLOR),
            # --------------------------------------------------------------------------------
            'Layout': '#99CC99',
            '[Merged-vca]': '#D3D3D3',
            '[Merged-hca]': '#B0E0E6'
        }
        super(RicoPalette, self).__init__()
        self.palette = {k: np.array(hex2rgb(self.HEX_PALETTE.get(k, rgb2hex(self.DEFAULT_COLOR)))).astype(float) for k in labels}


class AndroidPalette(Palette):
    """For Android view labels."""

    def __init__(self, labels=None):
        self.HEX_PALETTE = {
            'TextView': '#4B0A88',
            'ImageView': '#EF5E92',
            'ListView': '#FEE0B6',
            'RecyclerView': '#FF9999',
            'LinearLayout': '#99CC99',
            'FrameLayout': '#99FFFF',
            'RelativeLayout': '#9999CC',
            'Toolbar': '#52D0DF',
            'Button': '#522E1E',
        }
        super(AndroidPalette, self).__init__()
        self.palette = {k: np.array(hex2rgb(v)).astype(float) for k, v in self.HEX_PALETTE.items() if (labels is None or k in labels)}
        if labels:
            for label in labels:
                for k, v in self.HEX_PALETTE.items():
                    if k in label:
                        self.palette[label] = np.array(hex2rgb(v)).astype(float)


# --------------------------------------------------------------------------------
class Hierarchy(object):

    MAX_WIDTH = 1440
    MAX_HEIGHT = 2560

    def __init__(self, hier, NodeClass, PaletteClass, uxid=None,  flattened=False):
        """
        bounds: [0, 0, 1440, 2560]
        children:
            bounds: [x1, y1, x2, y2]
            componentLabel: str
            children: [...]
        """
        self.NodeClass = NodeClass
        self.PaletteClass = PaletteClass
        self.depth = 0
        self._nodes = []
        self._childnodes= []        
        if flattened:
            self.root = self.build_flat_hier(hier,0,0)  # Directly build the flattened hier
        else:
            self.root = self.build_nodes(hier,0,0)  # First build RICO hier
        self.uxid = uxid
        self.is_valid = True
        
        self._palette = None
        self.is_labeled = False


    def build_nodes(self, hier, child_idx, level):
        """parses json hierarchy"""
        node_id = len(self._nodes)

        is_root = (level == 0)
        is_leaf = ('children' not in hier or len(hier['children']) == 0)

        # node attributes
        node = self.NodeClass(hier) if not isinstance(hier, LayoutNode) else deepcopy(hier)
        # hierarchy structure
        node.node_id = node_id
        node.level = level
        node.is_leaf = is_leaf
        node.is_root = is_root
        node.child_idx = child_idx
        node.children = []

        self.depth = max(self.depth, node.level)
        self._nodes += [node]

        if not is_leaf:
            for idx, child_hier in enumerate(hier['children']):
                child = self.build_nodes(child_hier, idx, level + 1)
                node.children += [child]

        return node
    
    def flatten(self, node):
        for idx, child_node in enumerate(node.children):
            child_node.level = 1 
            if child_node not in self.root.children:
                # node.children.remove(child_node)
                node.children = []
                self.root.children.append(child_node)
            if len(child_node.children)!=0:
                self.flatten(child_node)
   
    def build_flat_hier(self, hier, child_idx, level):
        """parses json hierarchy"""
        node_id = len(self._nodes)
        
        is_root = (level == 0)
        is_leaf = ('children' not in hier or len(hier['children']) == 0)

        # node attributes
        root = self.NodeClass(hier) 
        # hierarchy structure
        root.node_id = node_id
        root.level = level
        root.is_leaf = is_leaf
        root.is_root = is_root
        root.child_idx = child_idx
        self._nodes += [root]
        
        if is_leaf: 
            root.children = [] 
        else:
            root.children = self.recurse(hier)
        return root    
    
    def recurse(self,hier):
        for i, child_hier  in enumerate (hier['children']):
            node = self.NodeClass(child_hier) 
            node.node_id = len(self._nodes)
            node.level = 1
            node.is_root = (node.level == 0)
            node.is_leaf = ('children' not in child_hier or len(child_hier['children']) == 0)
            node.child_idx = i
            node.children = []
            self._nodes += [node]
            self._childnodes +=[node]
            
            if not node.is_leaf:
                self.recurse(child_hier)
  
        return self._childnodes      

    def dfs(self, node=None):
        if node is None: node = self.root
        ret = [node]
        for child in node.children:
            ret += self.dfs(child)
        return ret


    @property
    def nodes(self):
        return self.dfs()


    def copy(self):
        return deepcopy(self)
    
    def copy_(self):
        return self.copy()
    

    # --------------------------------------------------------------------------------
    """For StructureNet"""

    SEM2CHILDREN_PATTERN = {
        'Toolbar': '*',
        'Text Button': 'Image | Text | Icon | Aligned* | Repeat*',
        'List Item': 'Image | Text | Icon | Text Button | Checkbox | Input | Aligned* | Repeat*',
        'Aligned*': '*',
        'Repeat*': '*'
    }
    
    rico_labels = ['Toolbar',    'Image',   'Web View',     'Icon',   'Text Button',     'Text', \
                'Multi-Tab',  'Card',     'List Item',       'Advertisement',   'Background Image', \
                'Drawer',     'Input',   'Bottom Navigation',  'Modal',          'Button Bar', \
                'Pager Indicator', 'On/Off Switch',  'Checkbox', 'Map View',  'Radio Button', \
                'Slider',  'Number Stepper',  'Video', 'Date Picker'] 
        
    toolbar_labels = ['Image', 'Text','Text Button', 'Icon', 'Toolbar']   
    additional_labels = ['[Merged-vca]',  '[Merged-hca]']
    
    @classmethod
    def set_semantics(cls, semantics='rico'):
        if semantics=='rico':
            cls.SEMANTICS = cls.rico_labels
        elif semantics=='toolbar':
            cls.SEMANTICS = cls.toolbar_labels
        elif semantics=='rico_plus':
            cls.SEMANTICS = cls.rico_labels + cls.additional_labels
        else:
            raise(f'Unknown semantics type: {semantics}')
        
        cls.SEMANTIC2ID = {sem: i for i, sem in enumerate(cls.SEMANTICS)}
        cls.SEM2ID = cls.SEMANTIC2ID
        cls.NUM_SEMANTICS = len(cls.SEMANTICS)
        cls.ID2SEM = {value:key for key, value in Hierarchy.SEM2ID.items()}
       
        # Need to work on this
        cls.SEM2CIDS = dict()
        cls.SEM2CIDS['[ROOT]'] = list(range(len(cls.SEMANTICS)))  # NEed to revise this with some prior knowledge and stats

        for sem, children in cls.SEM2CHILDREN_PATTERN.items():
            for k in cls.SEMANTICS:
                if not fnmatch(k, sem): continue
                cls.SEM2CIDS[k] = []

                for c in children.split(' | '):
                    cls.SEM2CIDS[k] += [cls.SEMANTIC2ID[_] for _ in cls.SEMANTICS if fnmatch(_, c)]

        for sem in cls.SEMANTICS:
            if sem not in cls.SEM2CIDS:
                cls.SEM2CIDS[sem] = []

        cls.SEM2CHILDREN = {sem: [cls.SEMANTICS[ci] for ci in cids] for sem, cids in cls.SEM2CIDS.items()}
        
        #Get the name of non leaf UI components
        cls.non_leaf_sem_names = ['Toolbar', 'Drawer', 'Bottom Navigation' , 'Modal', 'Button Bar', 'Date Picker', '[Merged-vca]',  '[Merged-hca]'  ]
        
    

    def to(self, device):
        self.root = self.root.to(device)
        return self
    # --------------------------------------------------------------------------------


    def compact_(self, node=None, strict=False, keep_ids=None):
        """Only keep nodes with more than one child & leaf nodes.
           If `strict`, then only collapse when parent and child have the same size.
        """
        if node is None: node = self.root
        if node.is_leaf: return node

        vis_children = [c for c in node.children if c.visible]

        if (len(vis_children) == 1) and \
           (keep_ids is None or node.node_id not in keep_ids) and \
           (not strict or np.all(vis_children[0].box == node.box)):
            # discard this node and replace with its child
            new_node = vis_children[0]
            new_node.level = node.level
            new_node.child_idx = node.child_idx
            new_node.is_root = node.is_root
            node.is_root = False
            if new_node.is_root: self.root = new_node
            return self.compact_(new_node, strict, keep_ids)

        for i, child in enumerate(node.children):
            child.level = node.level + 1
            node.children[i] = self.compact_(child, strict, keep_ids)
        return node


    def keep_(self, keep_ids, node=None):
        """Returns None IFF all nodes in the subtree of `node` are not in `keep_ids`.
        NOTE(zhzhou): `self.root` is assumed to be always kept."""
        if node is None: node = self.root
        if node.is_leaf:
            return node if (node.node_id in keep_ids) else None

        keep_children = []

        for i, child in enumerate(node.children):
            child = self.keep_(keep_ids, child)

            if child:
                child.child_idx = len(keep_children)
                keep_children += [child]

        node.children = keep_children

        if len(keep_children) == 0:
            if node.node_id in keep_ids:
                node.is_leaf = True
            else:
                return None

        return node


    def clean_(self, target_hier):
        self.compact_(strict=True)

        nodes = self.nodes
        keep_ids = set()

        for x in target_hier.nodes:
            matched = [y for y in nodes if (y.visible and np.all(y.box == x.box))]
            if len(matched) != 1:
                raise RicoError(f'Cannot find matching node for {x}, matched: {matched}')

            y = matched[0]
            y.label = x.label
            keep_ids.add(y.node_id)

        self.keep_(keep_ids)
        self.compact_(strict=False, keep_ids=keep_ids)
        self.palette = target_hier.palette



    def layout_(self, target_hier, node=None):
        """For any node with more than one children, grow a Layout node;
           For any node whose label is not in `target_hier`, convert it into a Layout Node.
        Determine Layout type according to the align/repeat relationship among children:
        1) LinearLayout
        2) Not LinearLayout but can be aligned
        3) Not LinearLayout and cannot be aligned
            y1
        x1  +-------+
            |       |
            |       |
            +-------+  x2
                    y2
        """
        def __linear_layout(node):
            boxes = [child.box for child in node.children]
            labels = [child.label for child in node.children]

            # check overlap
            is_H = True
            H_order = np.argsort([b[0] for b in boxes])
            for i in range(len(boxes) - 1):
                if boxes[H_order[i]][2] > boxes[H_order[i + 1]][0]:
                    is_H = False; break

            is_V = True
            V_order = np.argsort([b[1] for b in boxes])
            for i in range(len(boxes) - 1):
                if boxes[V_order[i]][3] > boxes[V_order[i + 1]][1]:
                    is_V = False; break

            # check align
            # TODO(zhzhou): ordering? check repeat group in align?
            # TODO(zhzhou): should repeat enforce that all subtrees are matching?
            EPS = 3
            is_HT = True; is_HB = True; is_HC = True
            is_VL = True; is_VR = True; is_VC = True
            is_same = True

            for i, (box, label) in enumerate(zip(boxes, labels)):
                x1, y1, x2, y2 = box
                ht = y1; hb = y2; hc = (y1 + y2) / 2
                vl = x1; vr = x2; vc = (x1 + x2) / 2
                w = x2 - x1; h = y2 - y1
                if i == 0:
                    HT, HB, HC = ht, hb, hc
                    VL, VR, VC = vl, vr, vc
                    LABEL = label
                    W, H = w, h
                else:
                    if abs(ht - HT) > EPS: is_HT = False
                    if abs(hb - HB) > EPS: is_HB = False
                    if abs(hc - HC) > EPS: is_HC = False
                    if abs(vl - VL) > EPS: is_VL = False
                    if abs(vr - VR) > EPS: is_VR = False
                    if abs(vc - VC) > EPS: is_VC = False
                    if abs(w - W) > EPS or abs(h - H) > EPS or label != LABEL:
                        is_same = False

            if is_H:
                children = [node.children[H_order[i]] for i in range(len(boxes))]
                for i, c in enumerate(children): c.child_idx = i

                if is_HT and is_HB and is_HC and is_same:
                    return RepeatNode(label=f'Repeat(H)', dir='H', n=len(boxes), children=children)

                elif is_HT or is_HB or is_HC:
                    flags = [is_HT, is_HB, is_HC]
                    align = ['t', 'b', 'c'][np.argmax(flags)] if sum(flags) == 1 else 'c'
                    return AlignedNode(label=f'Aligned(H, {align})', dir='H', align=align, children=children)

            if is_V:
                children = [node.children[V_order[i]] for i in range(len(boxes))]
                for i, c in enumerate(children): c.child_idx = i

                if is_VL and is_VR and is_VC and is_same:
                    return RepeatNode(label=f'Repeat(V)', dir='V', n=len(boxes), children=children)

                elif is_VL or is_VR or is_VC:
                    flags = [is_VL, is_VR, is_VC]
                    align = ['l', 'r', 'c'][np.argmax(flags)] if sum(flags) == 1 else 'c'
                    return AlignedNode(label=f'Aligned(V, {align})', dir='V', align=align, children=children)

            # raise RicoError(f'{node} not LinearLayout! {locals()}\n')
            raise RicoError()


        def __layout_node(node, node_id, level, child_idx):
            try:
                layout_node = __linear_layout(node)

            except RicoError as e:
                # print(e)
                # if node.label == 'LinearLayout':
                #     print('ERROR!!! LinearLayout can be for sure resolved.\n')

                layout_node = LayoutNode(label='Unknown', children=node.children)

            layout_node.update(dict(
                node_id=node_id,
                visible=True, is_leaf=False, is_root=False,
                level=level, child_idx=child_idx,
                box=node.box
            ))
            return layout_node


        if node is None: node = self.root

        if node.is_leaf: return node

        if node.label in target_hier.labels():
            if len(node.children) > 1:
                layout_node = __layout_node(node,
                    node_id=len(self), level=node.level + 1, child_idx=0)
                node.children = [layout_node]
        else:
            node = __layout_node(node,
                node_id=node.node_id, level=node.level, child_idx=node.child_idx)

        assert len(node.children) == 1 or isinstance(node, LayoutNode)

        for i, child in enumerate(node.children):
            child.level = node.level + 1
            node.children[i] = self.layout_(target_hier, child)

        return node


    def to_string(self, lambda_fn=None, node=None, render='terminal', labeled=True):
        """Borrowed from StructureNet codebase:
           https://github.com/daerduoCarey/structurenet/blob/master/code/data.py#L13
        """
        if node is None: node = self.root
        if not node.visible: return ''

        info = '' if lambda_fn is None else '\t' + str(lambda_fn(node))

        LABEL = node.label
        if not labeled:
            LABEL = '◯'

        if hasattr(node, 'text_color'):
            if render == 'terminal':
                LABEL = termcolor.colored(LABEL, node.text_color)
            elif render == 'html':
                LABEL = f'<span style="color:{node.text_color}">{LABEL}</span>'
            elif render is None or render == 'none' or render == 'None':
                LABEL = LABEL
            else:
                raise RicoError(f'Unknown rendering: {render}')

        ret = '  |' * (node.level - 1) + \
              '  ├' * (node.level > 0) + str(node.node_id) + ' ' + LABEL + \
              (' [LEAF] ' if node.is_leaf else '    ') + info + '\n'

        for child in node.children:
            ret += self.to_string(lambda_fn, child, render, labeled)

        return ret


    def __repr__(self):
        return self.to_string(None, self.root)
    
    def greedy_merge_at_one_level(self, node=None, costfn=None):
        node = self.root if node is None else node
        if len(node.children) > 2:
                node = self.greedy_merge_in_one_child(node) # Dynamically set attribute??? how?? 
                
                
    def greedy_merge_on_rico_hier(self, node=None, costfn=None):
        node = self.root if node is None else node
        
        if node.children != []:  # Is this leaf node if no --> recurse and then cluster at that level as well
            for child in node.children:
                child = self.greedy_merge_on_rico_hier(node=child)
        
            # print(f'\n operating at level {node.level}')    

            node = self.greedy_merge_at_one_level(node=node)
                    
        
    def compute_pairwise_adjaceny(self, node, use_tol_v=True, use_tol_h=False):
        thres = 10 # 10 pixels tolerance
        bbox = np.array([x.box for x in node.children])
        center = np.hstack([((bbox[:,0]+bbox[:,2])/2).reshape(-1,1), ((bbox[:,1]+bbox[:,3])/2).reshape(-1,1)])
        # is_vcenterA = np.equal(center[:,None,0], center[:,0])
        # is_hcenterA = np.equal(center[:,None,1], center[:,1]) 
        vcenter_dist = abs(center[:,None,0] - center[:,0])
        hcenter_dist = abs(center[:,None,1] - center[:,1])
        is_vcenterA = vcenter_dist<thres
        is_hcenterA = hcenter_dist<thres
        
        # is vertically adjacent
        vdist = np.minimum(abs(bbox[:, None, 3] - bbox[:,1]), abs(bbox[:, None, 1] - bbox[:,3]))  
        np.fill_diagonal(vdist, 0)
        is_vadj = vdist<50

        
        # is horizontally adjacent 
        hdist = np.minimum(abs(bbox[:,None,2] - bbox[:,0]), abs(bbox[:,None,0] - bbox[:,2]))
        np.fill_diagonal(hdist, 0)
        is_hadj = hdist<50  
        
        if use_tol_v: is_vcenterA = is_vcenterA * is_vadj
        if use_tol_h: is_hcenterA = is_hcenterA * is_hadj
        
        return is_vcenterA, is_hcenterA

            
    def greedy_merge_in_one_child(self, node):
        is_vcenterA, is_hcenterA = self.compute_pairwise_adjaceny(node,use_tol_v=True, use_tol_h=False)
        
        # Cost function for merging
        # Select which one to apply first [hca->vca OR vca->hca]
        node_copy_h = node.copy_()
        node_copy_v = node.copy_()
        merged_node_h, compactness_h, n_clus_h, n_new_clus_h, mAA_h = self.merge_CCA_on_one_node(node_copy_h, is_hcenterA, criteria='hca')
        merged_node_v, compactness_v, n_clus_v, n_new_clus_v, mAA_v = self.merge_CCA_on_one_node(node_copy_v, is_vcenterA, criteria='vca')
        
        # print (f'horizontal - comp: {compactness_h} n_clus: {n_clus_h} mAA_h: {mAA_h}' )
        # print (f'vertical - comp: {compactness_v} n_clus: {n_clus_v} mAA_v: {mAA_v}')
        
        #Select the criteria here [compactness, n_new_clus, mean Average Area (per unit)]
        # Lower mAA is chosen
        # Todo: Optimize these codes.
        if mAA_v < mAA_h and n_new_clus_v !=0:
            sim = is_vcenterA
            criteria = 'vca'
        else:
            sim = is_hcenterA
            criteria = 'hca'
        
        merged_node1, compactness1, nclus1, n_new_clus1, mAA1 = self.merge_CCA_on_one_node(node, sim, criteria=criteria) 

        # Second round of merging    
        is_vcenterA2, is_hcenterA2 = self.compute_pairwise_adjaceny(merged_node1,use_tol_v=True, use_tol_h=False)
        if criteria =='hca':
            criteria = 'vca'
            sim = is_vcenterA2
        elif criteria == 'vca':
            criteria = 'hca'
            sim = is_hcenterA2
    
        merged_node2, compactness2, nclus2, n_new_clus2, mAA2 = self.merge_CCA_on_one_node(merged_node1, sim, criteria=criteria) 
        
        return merged_node2

       
    def merge_CCA_on_one_node(self, parent_node, sim, criteria=None):
        bbox = np.array([x.box for x in parent_node.children])
        node_ids = [x.node_id for x in parent_node.children]
        
        # cc_sim = [np.nonzero(x)[0] for x in sim if np.nonzero(x)[0].size >1] # index of nodes >1
        cc_edges = [np.nonzero(x)[0] for x in sim]
        
        # # Method 1: remove duplicates. Note doesnot work for connected components re: adjacenct matrix with associative property
        # temp = {x.tobytes(): x for x in cc_sim}   # Removing duplicates using dictionary method
        # cc_sim = list(temp.values())
        
        # Method 2: find the connected components with graphs 
        G = to_graph(cc_edges)
        cc_sim = [list(x) for x in connected_components(G)]
               
        if len(cc_sim) == 1:  # if all nodes in parent are merged; no need of extra hier 
            return parent_node, 0, 1, 1, 0
        
        # print( f'parent node: {parent_node} num_cluster: {len(cc_sim)} {[len(x) for x in cc_sim]}')    
        compactness = self.mean_intracluster_adjacency(parent_node, cc_sim, criteria=criteria)
        
        n_clus = len(cc_sim)
        mAA = 0  # mean average Area
        cc_sim = [x for x in cc_sim if len(x)>1] # only merge if more than one element
        n_new_clus = len(cc_sim)
        for cc in cc_sim:
            #first create a new merged component. 
            # top left of union
            tl = np.min(bbox[cc, :2], axis=0)
            # bottom right of union 
            br = np.max(bbox[cc, 2:], axis=0)
            
            #Area metric for cost function 
            A = np.prod (br-tl)
            AA = A/len(cc)
            mAA += AA
            
            merged_node = {'bounds': np.concatenate([tl,br]),
                           'componentLabel': f'[Merged-{criteria}]'}            
            node = self.NodeClass(merged_node)
            node.node_id = len(self._nodes)
            node.is_leaf = False
            node.is_root = False
            node.level = parent_node.level+1
            node.child_idx = None
            node.children = []
            
            # Add children to the merged node and delete from flat one
            child_node_ids = [node_ids[x] for x in cc] 
            node.children += [x for x in parent_node.children if x.node_id in child_node_ids]

            # reset the child_idx starting from 0, 1... 
            for ii in range(len(node.children)):
                node.children[ii].child_idx = ii
    
            #for x in range(len(node.children)): node.children[x].level +=1
            node = self.increment_level(node)
            
            parent_node.children +=[node]  # add a new node to root.children
            parent_node.children = [x for x in parent_node.children if x.node_id not in child_node_ids]  # remove the merged node 
            self._nodes+=[node]

            # reset the child_idx for parent node as well (some of the nodes has been removed already.) 
            for ii in range(len(parent_node.children)):
                parent_node.children[ii].child_idx = ii
        
        try:
            mAA = mAA/len(cc_sim)
        except ZeroDivisionError:
            mAA = 0
        
        return parent_node, compactness, n_clus, n_new_clus, mAA
        
    def mean_intracluster_distance(self, node, cc_sim, criteria=None):
        bbox = np.array([x.box for x in node.children])
        node_ids = [x.node_id for x in node.children]
        S = 0
        # print('Inside mean intracluster distance {}'.format(node))
        
        for cc in cc_sim:
            center = np.hstack([((bbox[cc,0]+bbox[cc,2])/2).reshape(-1,1), ((bbox[cc,1]+bbox[cc,3])/2).reshape(-1,1)])
            diff = center[:,None,:] - center[:,:]
            dist = np.mean(np.sqrt(np.sum(diff*diff,axis=2)))
            # print (f'mean dist for nodeID {[node_ids[x] for x in cc]} is {dist}')
            S+=dist 
        
        mean_dist = S/len(cc_sim)
        return mean_dist

    def mean_intracluster_adjacency(self, node, cc_sim, criteria=None):
        bbox = np.array([x.box for x in node.children])
        node_ids = [x.node_id for x in node.children]
        S = 0
        # print('\nInside mean intracluster distance {}'.format(node))
        assert (criteria  in ('vca', 'hca'))
        
        cc_sim = [x for x in cc_sim if len(x)>1] # only merged  or new cluster
        for cc in cc_sim:
            if criteria == 'vca':
                diff = np.minimum(abs(bbox[cc, None, 3] - bbox[cc,1]), abs(bbox[cc, None, 1] - bbox[cc,3]))  
            elif criteria == 'hca':
                diff = np.minimum(abs(bbox[cc,None,2] - bbox[cc,0]), abs(bbox[cc,None,0] - bbox[cc,2]))
    
            np.fill_diagonal(diff, 0)
            pairwise_dist = np.mean(diff)
            nearest_dist = np.mean( np.sort(diff, axis=0)[1, :]) 
            # print (f'mean criteria and adjacency dist for nodeID {[node_ids[x] for x in cc]} is {criteria}, {nearest_dist}')
            S+=nearest_dist            
        
        try:
            mean_dist = S/len(cc_sim)
        except ZeroDivisionError:
            mean_dist = 0
        return mean_dist

    def increment_level(self, node):
        for i, child_node in enumerate(node.children):
            node.children[i].level+=1
            if len(child_node.children) != 0:
                node.children[i] = self.increment_level(node.children[i])
        return node
       
    def check_valid_for_dataset(self, node=None, max_total_node=50, max_child_per_node=10):
        if node is None: node = self.root
        
        if len(node.children) > max_child_per_node:
            self.is_valid = False
            
        for child in node.children:
            self.check_valid_for_dataset(child, max_total_node, max_child_per_node)
    
    
    ###################
    def draw_node(self, im, node, palette, mode, alpha, draw_root=False, lambda_fn=None):
        """inplace draw a single node to im"""
        if not node.visible: return
        if node is self.root and not draw_root: return
        if node.level == 1: alpha = 1.0

        color = palette(node.label)
        title = f'[{node.node_id}]' if lambda_fn is None else f'[{lambda_fn(node)}]'

        if mode == 'solid':
            plot_box(im, node.box, color, solid=True, alpha=alpha)
            BORDER_COLOR = BGR.WHITE
            TEXT_COLOR = BGR.WHITE
        elif mode == 'wireframe':
            BORDER_COLOR = color
            TEXT_COLOR = color

        plot_box(im, node.box, BORDER_COLOR, thickness=6)

        pos = ('lt' if (node.level > 1 or node.is_leaf) else 'br')
        plot_text(im, title, TEXT_COLOR, scale=2, thickness=5, box=node.box, pos=pos,
            minThickness=3, minScale=1.2)


    @staticmethod
    def occludes(nodeA, nodeB):
        """a completely occludes b => +1,
           b completely occludes a => -1,
           partially overlaps => 0
        """
        def _occludes(boxA, boxB):
            return np.all(boxA[:2] <= boxB[:2]) and np.all(boxA[-2:] >= boxB[-2:])

        if _occludes(nodeA.box, nodeB.box):
            return 1
        elif _occludes(nodeB.box, nodeA.box):
            return -1
        else:
            return 0


    def plot_(self, im, palette=None, mode='solid', alpha=0.8, draw_root=False, lambda_fn=None):
        """inplace"""
        if palette is not None: self.palette = palette

        Q = Queue()
        Q.put(self.root)

        while not Q.empty():
            node = Q.get()

            self.draw_node(im, node, self.palette, mode, alpha, draw_root, lambda_fn)

            for child in sorted(node.children, key=cmp_to_key(self.occludes), reverse=True):
                Q.put(child)


    def plot(self, im=None, palette=None, **kwargs):
        """copied"""
        if im is None:
            canvas = np.zeros([self.MAX_HEIGHT, self.MAX_WIDTH, 3], np.uint8)
        else:
            canvas = im.copy()

        self.plot_(canvas, palette, **kwargs)
        return canvas


    def legend(self):
        return self.palette.legend()


    def __len__(self):
        return len(self.nodes)


    def __iter__(self):
        for node in self.nodes:
            yield node


    def labels(self):
        return [node.label for node in self.nodes]


    @property
    def palette(self):
        if self._palette is None:
            self._palette = self.PaletteClass(self.labels())
        return self._palette


    @palette.setter
    def palette(self, new_palette):
        self._palette = new_palette


class Node(Dict):
    def __init__(self, **kwargs):
        super(Node, self).__init__(**kwargs)
        self.device = None
        self.xywh = None
        self.center = None
        self.scale = None
        self.center_scale = None
        self.visible = kwargs.get('visible', True)

    def __repr__(self):
        return f'<{self.label} id={self.node_id}, visible={self.visible}, is_leaf={self.is_leaf}, is_root={self.is_root}>'

    @property
    def semantic(self):
        return self.label

    """For StructureNet"""
    def to(self, device):
        self.device = device
        if self.xywh is not None: self.xywh = self.xywh.to(device)
        for c in self.children: c.to(device)
        return self

    """For StructureNet"""
    def get_semantic_id(self):
        return Hierarchy.SEMANTIC2ID[self.semantic]

    """For StructureNet"""
    def get_semantic_one_hot(self):
        out = np.zeros((1, Hierarchy.NUM_SEMANTICS), dtype=np.float32)
        out[0, self.get_semantic_id()] = 1
        return T.tensor(out, dtype=T.float32).to(self.device)
 
    """For StructureNet"""
    def get_xywh(self):
        # self.box: [x1, y1, x2, y2], [0-MAX_HEIGHT, 0-MAX_WIDTH]
        # self.xywh: [x1, y1, w, h], [-1~1, -1~1, 0~1, 0~1]
        # TODO(zhzhou): the height of most elements are quite small
        if self.xywh is None:
            x1, y1, x2, y2 = self.box
            w, h = x2 - x1, y2 - y1
            x = normalize(x1, 0, Hierarchy.MAX_WIDTH)
            y = normalize(y1, 0, Hierarchy.MAX_HEIGHT)
            w = w / Hierarchy.MAX_WIDTH
            h = h / Hierarchy.MAX_HEIGHT
            self.xywh = [x, y, w, h]
            self.xywh = T.tensor(self.xywh, dtype=T.float32).view(1, -1).to(self.device)

        return self.xywh

    """For StructureNet"""
    def set_xywh(self, xywh):
        x, y, w, h = xywh.cpu().numpy().squeeze()
        x1 = unnormalize(x, 0, Hierarchy.MAX_WIDTH)
        y1 = unnormalize(y, 0, Hierarchy.MAX_HEIGHT)
        w = w * Hierarchy.MAX_WIDTH
        h = h * Hierarchy.MAX_HEIGHT
        x2 = x1 + w; y2 = y1 + h
        self.box = np.array([x1, y1, x2, y2])


    def get_center_scale(self):
        if self.center_scale is None:
            x1, y1, x2, y2 = self.box
            center_x = normalize((x1 + x2) / 2, 0, Hierarchy.MAX_WIDTH)
            center_y = normalize((y1 + y2) / 2, 0, Hierarchy.MAX_HEIGHT)
            scale_x = (x2 - x1) / Hierarchy.MAX_WIDTH
            scale_y = (y2 - y1) / Hierarchy.MAX_HEIGHT
            self.center = [center_x, center_y]
            self.scale = [scale_x, scale_y]
            self.center_scale = T.tensor([center_x, center_y, scale_x, scale_y], dtype=T.float32) \
                                 .view(1, -1).to(self.device)
        return self.center_scale


    def set_center_scale(self, center_scale):
        center_x, center_y, scale_x, scale_y = center_scale.cpu().numpy().squeeze()
        center_x = unnormalize(center_x, 0, Hierarchy.MAX_WIDTH)
        center_y = unnormalize(center_y, 0, Hierarchy.MAX_HEIGHT)
        scale_x = scale_x * Hierarchy.MAX_WIDTH
        scale_y = scale_y * Hierarchy.MAX_HEIGHT
        x1 = center_x - scale_x / 2; x2 = center_x + scale_x / 2
        y1 = center_y - scale_y / 2; y2 = center_y + scale_y / 2
        self.box = np.array([x1, y1, x2, y2])
        
        
    def copy_(self):
        return deepcopy(self)

    """ For Hierarchical StructureNet graphs""" 
    def get_cwh(self, box):
        x_min, y_min, x_max, y_max = box
        cx = (x_min + x_max) * 0.5
        cy = (y_min + y_max) * 0.5
        w = (x_max - x_min) + 1.
        h = (y_max - y_min) + 1.
        return cx, cy, w, h
    
    """ For Hierarchical StructureNet graphs""" 
    def get_edges(self, device):
        node = self
        W, H = 1440, 2560
        scale = W * H
        diag_len = math.sqrt(W ** 2 + H ** 2)
        
        boxes = np.array([x.box for x in node.children])
        child_idxs = [x.child_idx for x in node.children]
        
        #print(boxes.shape)
        #print(child_idxs)
        
        edge_indices = []
        edge_feats = []
        for ii in range(len(child_idxs)):
            for jj in range(len(child_idxs)):
                if ii==jj:
                    continue
                
                box1, box2 = boxes[ii], boxes[jj]
            
                # Convert to xyxy format, if the box in  xywh format
                # box1 = self.convert_xywh_to_xyxy(box1)
                # box2 = self.convert_xywh_to_xyxy(box2)
                
                cx1, cy1, w1, h1 = self.get_cwh(box1)
                cx2, cy2, w2, h2 = self.get_cwh(box2)
                
                x_min1, y_min1, x_max1, y_max1 = box1
                x_min2, y_min2, x_max2, y_max2 = box2
                
                # scale (area)
                scale1 = w1 * h1
                scale2 = w2 * h2
                
                # Offset
                offsetx = cx2 - cx1
                offsety = cy2 - cy1
                
                # Aspect ratios
                aspect1 = w1 / h1
                aspect2 = w2 / h2
                
                # Width and height ratios 
                # ToDo:  use this instead of aspect ratios
                # aspect1 = w1/w2
                # aspect2 = h1/h2
                
                # Overlap (IoU)
                i_xmin = max(x_min1, x_min2)
                i_ymin = max(y_min1, y_min2)
                i_xmax = min(x_max1, x_max2)
                i_ymax = min(y_max1, y_max2)
                iw = max(i_xmax - i_xmin + 1, 0)
                ih = max(i_ymax - i_ymin + 1, 0)
                areaI = iw * ih
                areaU = scale1 + scale2 - areaI
                
                # dist
                dist = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                
                # angle
                angle = math.atan2(cy2 - cy1, cx2 - cx1)
                
                #ToDo: f1 and f2 normalzied by W and H insteas by sacle (area)?
                try:
                     f1 = offsetx / math.sqrt(scale1) 
                except ValueError:
                    print (offsetx, scale1)
                
                f1 = offsetx / math.sqrt(scale1)   # 
                f2 = offsety / math.sqrt(scale1)      
                try:
                     f3 = math.sqrt(scale2 / scale1)  
                except ValueError:
                    print (scale2, scale1)
                    print(box1, box2)
                      
                f4 = areaI / areaU
                f5 = aspect1
                f6 = aspect2
                f7 = dist / diag_len
                f8 = angle
                feat = [f1, f2, f3, f4, f5, f6, f7, f8]
                edge_indices.append([ii,jj])
                edge_feats.append(feat)
            
        
        edge_indices = torch.tensor(edge_indices, device=device, dtype=torch.long).view(1,-1,2)
        edge_feats = torch.tensor(edge_feats,device=device, dtype=torch.float32).view(1,-1,8)
        
        # print(edge_indices.shape)
        # print(edge_feats.shape)
        
        return  edge_feats, edge_indices


class SemanticNode(Node):
    def __init__(self, hier):
        """hier: json object"""
        super(SemanticNode, self).__init__(
            box = np.array(hier['bounds']),
            label = hier.get('componentLabel', '[ROOT]'),
            visible = True
        )

class FullNode(Node):
    def __init__(self, hier):
        """hier: json object"""
        super(FullNode, self).__init__(
            box = np.array(hier['bounds']),
            label = hier.get('class', '[ROOT]').split('.')[-1],
            visible = hier['visible-to-user'] and hier['visibility'],
        )

class RicoNode(Node):
    def __init__(self, hier):
        """hier: Node object"""
        super(RicoNode, self).__init__(
            box = np.array(hier['box']),
            label = hier['label'],
            visible = True
        )
        for attr in ['channel']:
            if hasattr(hier, attr):
                setattr(self, attr, hier[attr])

class LayoutNode(Node):
    def __init__(self, *args, **kwargs):
        super(LayoutNode, self).__init__(*args, **kwargs)
        self.visible = True

class AlignedNode(LayoutNode):
    def __repr__(self):
        return f'<{self.label} id={self.node_id}, align={self.align}, dir={self.dir}>'

    @property
    def semantic(self):
        return f'Aligned({self.dir}, {self.align})'

class RepeatNode(LayoutNode):
    def __repr__(self):
        return f'<{self.label} id={self.node_id}, n={self.n}, dir={self.dir}>'

    @property
    def semantic(self):
        # TODO(zhzhou): also encode the `n` attribute
        return f'Repeat({self.dir})'

class SemanticHierarchy(Hierarchy):
    def __init__(self, hier, uxid=None, flattened=False):
        super(SemanticHierarchy, self).__init__(hier, SemanticNode, RicoPalette, uxid=uxid, flattened=flattened)


class FullHierarchy(Hierarchy):
    def __init__(self, hier):
        super(FullHierarchy, self).__init__(hier['activity']['root'], FullNode, AndroidPalette)

class RicoHierarchy(Hierarchy):
    def __init__(self, hier):
        if isinstance(hier, Hierarchy): hier = hier.root
        super(RicoHierarchy, self).__init__(hier, RicoNode, RicoPalette)


class RicoError(Exception): pass


##################################################################################
# Supporting functions for connected component while extracting hierarchy
def to_graph(l):
    G = networkx.Graph()
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """ 
        treat `l` as a Graph and returns it's edges 
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current 


# --------------------------------------------------------------------------------
# ===========================================================================
# ============================================================================

class Rico(object):
    def __init__(self, rico_root, is_cached=True):
        self.RICO_ROOT = rico_root
        self.index_file = f'{self.RICO_ROOT}/index.txt'
        self.index = open(self.index_file).readlines()
        self._hier_cache = dict()
        self.is_cached = is_cached

    def __iter__(self):
        for line in self.index:
            yield int(line)

    def __len__(self):
        return len(self.index)

    def get_hierarchy_path(self, uxid):
        return f'{self.RICO_ROOT}/semantic_annotations/{uxid}.json'

    def get_full_hierarchy_path(self, uxid):
        return f'{self.RICO_ROOT}/combined/{uxid}.json'

    def get_image_path(self, uxid):
        return f'{self.RICO_ROOT}/combined/{uxid}.jpg'

    def get_semantic_image_path(self, uxid):
        return f'{self.RICO_ROOT}/semantic_annotations/{uxid}.png'

    def get_hierarchy(self, uxid, layout=False):
        if not layout:
            if self.is_cached and uxid in self._hier_cache:
                return self._hier_cache[uxid]

            sem_json = self.get_hierarchy_path(uxid)
            hier = SemanticHierarchy(json.load(open(sem_json)))

            if self.is_cached:
                self._hier_cache[uxid] = hier

        else:
            sem_hier = self.get_hierarchy(uxid, layout=False)
            hier = self.get_full_hierarchy(uxid)
            hier.clean_(sem_hier)
            hier.layout_(sem_hier)
            hier = RicoHierarchy(hier)

        return hier

    def flattened(self, hier, key):
        if isinstance(hier, Hierarchy):
            root, hier = hier.root, hier
        elif isinstance(hier, Node):
            root, hier = hier, RicoHierarchy(hier)
        else:
            raise ValueError(f'Invalid hierarchy: `{hier}`')
        root.children = [x for x in hier.nodes if not x.is_root]
        root.children = sorted(root.children, key=key)
        for i, c in enumerate(root.children):
            c.parent = root
            c.child_idx = i
            c.level = 1
            c.children = []
            c.is_leaf = True
        hier = RicoHierarchy(root)
        return hier

    def get_flattened_hierarchy(self, uxid, key=lambda x: (x.box[1], x.box[0])):
        hier = self.get_hierarchy(uxid, layout=False)
        return self.flattened(hier, key=key)
    
    # Added by dipu
    def get_rico_flattened_hierarchy (self, uxid, extract_hier=False, key=lambda x: (x.box[1], x.box[0])):
        if self.is_cached and uxid in self._hier_cache:
                return self._hier_cache[uxid]

        sem_json = self.get_hierarchy_path(uxid)
        hier = SemanticHierarchy(json.load(open(sem_json)), uxid=uxid)
        if extract_hier:
            hier.greedy_merge_on_rico_hier()
        hier = self.flattened(hier, key=key)
        
        if self.is_cached:
            self._hier_cache[uxid] = hier
        
        return hier

    def get_rico_hierarchy(self, uxid, extract_hier=True, key=lambda x: (x.box[1], x.box[0])):
        sem_json = self.get_hierarchy_path(uxid)
        hier = SemanticHierarchy(json.load(open(sem_json)), uxid=uxid)
        if extract_hier:
            hier.greedy_merge_on_rico_hier()
        return hier

        
    def get_full_hierarchy(self, uxid):
        full_json = self.get_full_hierarchy_path(uxid)
        return FullHierarchy(json.load(open(full_json)))

    def get_image(self, uxid):
        full_im = self.get_image_path(uxid)
        return cv2.imread(full_im)

    def get_semantic_image(self, uxid):
        sem_im = self.get_semantic_image_path(uxid)
        return cv2.imread(sem_im)

# ===========================================================================
# ---------------------------------------------------------------------------
# ===========================================================================

import apted

class ALGO:
    class TreeEditDistanceConfig(apted.Config):
        def rename(self, node1, node2):
            """Compares attribute .value of trees"""
            return 1 if node1.label != node2.label else 0

        def children(self, node):
            """Get left and right children of binary tree"""
            return node.children


    class UnlabeledTreeEditDistanceConfig(apted.Config):
        def rename(self, node1, node2):
            """Compares attribute .value of trees"""
            return 0

        def children(self, node):
            """Get left and right children of binary tree"""
            return node.children


    @classmethod
    def sort_(cls, node):
        """Inplace sort the children of `node` by its semantic ID."""
        node.children = sorted(node.children, key=lambda c: Hierarchy.SEM2ID[c.label])
        for i, c in enumerate(node.children):
            node.children[i] = cls.sort_(c)
        return node


    @classmethod
    def sort(cls, hierarchy):
        """Copy and sort a Hierarchy."""
        return RicoHierarchy(cls.sort_(hierarchy.copy().root))


    @classmethod
    def markup(cls, A, B, edit_mapping, labeled=True):
        for a, b in edit_mapping:
            if a is not None and b is not None:
                if a.label == b.label or (not labeled):
                    a.text_color = b.text_color = 'green'
                else:
                    a.text_color = b.text_color = 'yellow'
            elif b is None: a.text_color = 'red'
            elif a is None: b.text_color = 'red'

    @classmethod
    def edit_distance(cls, A, B, markup=True, labeled=True):
        """computes the edit distance (order-sensitive) between two trees."""
        config = cls.TreeEditDistanceConfig() if labeled else cls.UnlabeledTreeEditDistanceConfig()

        TED = apted.APTED(A.root, B.root, config)
        edit_distance = TED.compute_edit_distance()

        if markup:
            edit_mapping = TED.compute_edit_mapping()
            cls.markup(A, B, edit_mapping, labeled)

        return edit_distance


    @classmethod
    def edit_mapping(cls, A, B, markup=True, labeled=True):
        config = cls.TreeEditDistanceConfig() if labeled else cls.UnlabeledTreeEditDistanceConfig()

        TED = apted.APTED(A.root, B.root, config)
        edit_mapping = TED.compute_edit_mapping()

        if markup:
            cls.markup(A, B, edit_mapping, labeled)

        return edit_mapping


    @classmethod
    def permute(cls, A, perms):
        """permute children according to `perms`, in DFS order"""
        assert len(A.nodes) == len(perms), [A.nodes, perms]

        A = A.copy()
        for node, perm in zip(A.nodes, perms):
            node.children = [node.children[i] for i in perm]
        return A


    @classmethod
    def get_permutations(cls, node):
        """Assume the children of nodes are already sorted by semantic id"""
        return itertools.permutations(
            list(range(len(node.children)))
        )


    @classmethod
    def __all_permutations(cls, A, node, perms, results):
        """permute each node in the dfs order; assume A is already sorted"""
        N = len(A)
        dfs_nodes = A.nodes

        for perm in cls.get_permutations(node):
            if node.node_id < N - 1:
                cls.__all_permutations(A, dfs_nodes[node.node_id + 1], perms + [perm], results)
            else:
                results += [cls.permute(A, perms + [perm])]


    @classmethod
    def all_permutations(cls, A):
        results = []
        cls.__all_permutations(A, A.root, [], results)
        return results


    @classmethod
    def minimum_edit_distance(cls, A, B, markup=True, labeled=True):
        """computes the edit distance (order-insensitive) between two trees."""
        if len(A) < len(B):
            source, target = cls.sort(A), cls.sort(B)
        else:
            source, target = cls.sort(B), cls.sort(A)

        candidates = cls.all_permutations(source)
        edit_distances = [cls.edit_distance(target, C, labeled) for C in candidates]

        best_idx = np.argmin(edit_distances)
        best_edit_distance = edit_distances[best_idx]
        best_source = candidates[best_idx]

        if len(A) < len(B):
            A = best_source; B = target
        else:
            B = best_source; A = target

        if markup:
            cls.edit_mapping(A, B, markup=True)

        return A, B, best_edit_distance


    @classmethod
    def iou(cls, a, b, epsilon=1e-5):
        """ From: http://ronny.rest/tutorials/module/localization_001/iou/
        Given two boxes `a` and `b` defined as a list of four numbers:
                [x1,y1,x2,y2]
            where:
                x1,y1 represent the upper left corner
                x2,y2 represent the lower right corner
            It returns the Intersect of Union score for these two boxes.

        Args:
            a:          (list of 4 numbers) [x1,y1,x2,y2]
            b:          (list of 4 numbers) [x1,y1,x2,y2]
            epsilon:    (float) Small value to prevent division by zero

        Returns:
            (float) The Intersect of Union score.
        """
        # COORDINATES OF THE INTERSECTION BOX
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        # AREA OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)
        # handle case where there is NO overlap
        if (width < 0) or (height < 0):
            return 0.0
        area_overlap = width * height

        # COMBINED AREA
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined + epsilon)
        return iou


# --------------------------------------------------------------------------------
if __name__ == '__main__':
    Hierarchy.set_semantics(['Image', 'Text', 'Text Button', 'Icon', 'Toolbar'])

    RICO_ROOT = f'../../data/rico-dataset'
    RICO = Rico(RICO_ROOT)

    EXP = '5classes'
    index_path = f'{RICO_ROOT}/index/index_{EXP}.txt'

    uxids = list(np.loadtxt(index_path).astype(int))
    print('total #UIs:', len(uxids))

    dataset = RicoDataset(RICO_ROOT, index_path, ['object', 'uxid'],
                          is_cached=False, layout_nodes=False)

    sizes = [len(hier) for hier, _ in dataset]
    print('mean hierarchy size:', np.mean(sizes))

    indices = random.sample(range(len(dataset)), 100)
    columns = []

    for i in tqdm(indices):
        hier, uxid = dataset[i]

        columns.append(dict(
            id = uxid,
            hierarchy = hier,
            flatten = dataset.RICO.get_flattened_hierarchy(uxid),
            layout = dataset.RICO.get_flattened_hierarchy(uxid).plot(),
        ))

    html = HTML(f'www/{EXP}_trainset', EXP, overwrite=True, inverted=True)
    html.add_bigtable(5).add(columns)
    html.save()
    html.show()
