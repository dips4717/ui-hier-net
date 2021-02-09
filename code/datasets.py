from rico import *
from moka import *


##################################################################################
class RicoDataset(data.Dataset):

    def __init__(self, rico_root, index, data_features, is_train, permute=False, n_permutes=1, is_cached=True, layout_nodes=False, flatten=False):
        """
        rico_root: str, RICO_ROOT
        index: str, path to index file, one uxid per line
        data_features: list of str, names of features returned by __getitem__
        """
        self.rico_root = rico_root
        self.RICO = Rico(rico_root, is_cached=False)

        self.data_features = data_features
        self._cache = dict()
        self.is_train = is_train
        self.permute = permute
        self.n_permutes = n_permutes
        self.is_cached = is_cached
        self.layout_nodes = layout_nodes
        self.flatten = flatten
        self.key = lambda x: (x.box[1], x.box[0])

        self.uxids = np.reshape([[int(uxid) for _ in range(n_permutes)] for uxid in open(index).readlines()], -1)
        print('is_train:', is_train, '#examples:', len(self.uxids), '#original:', len(list(open(index))), '#permutes:', n_permutes)


    def group_id(self, uxid):
        return self.__class__.__name__.replace('Dataset', '')


    def random_permutation(self, hier):
        assert self.flatten and self.is_train and self.permute
        o = deepcopy(hier)
        random.shuffle(o.root.children)
        return o


    def get_hierarchy(self, uxid):
        if not self.flatten:
            try:
                hier = self.RICO.get_hierarchy(uxid, layout=self.layout_nodes)
            except:
                raise RicoError(f'Unsupported hierarchy: UI #{uxid}')
        else:
            hier = self.RICO.get_flattened_hierarchy(uxid, key=self.key)

        return hier


    def get_cached_hierarchy(self, id):
        if self.is_cached and id in self._cache:
            return self._cache[id]
        else:
            ret = self.get_hierarchy(id)
            if self.is_cached:
                self._cache[id] = ret 
            return ret


    def __getitem__(self, i):
        """
        Returns: tuple according to self.data_features
        e.g. ['object'] ==> (Hierarchy,)
        """
        id = self.uxids[i]

        data_feats = ()
        for feat in self.data_features:

            if feat == 'object':
                hier = self.get_cached_hierarchy(id)

                # if self.permute and self.is_train:
                #     hier = self.random_permutation(hier)

                data_feats = data_feats + (hier,)

            elif feat == 'uxid':
                data_feats = data_feats + (self.uxids[i],)

            else:
                raise RicoError(f'Unknown feature type: {feat}')

        return data_feats


    def __len__(self):
        return len(self.uxids) 


##################################################################################
class FiveClassesDataset(RicoDataset):

    def __init__(self, *args, **kwargs):
        super(FiveClassesDataset, self).__init__(*args, **kwargs)

    @staticmethod
    def sort_key(dir):
        def _key(x):
            x1, y1, x2, y2 = x.box
            center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            if dir == 'H':
                return (center[0], center[1])
            elif dir == 'V':
                return (center[1], center[0])
            else:
                raise ValueError(dir)
        return _key


    def sort(self, node):
        if isinstance(node, LayoutNode):
            node.children = sorted(node.children, key=self.sort_key(node.dir))
        else:
            assert len(node.children) <= 1, f'{node}, {type(node)}'

        for i, c in enumerate(node.children):
            node.children[i] = self.sort(c)
        return node


    def get_hierarchy(self, uxid):
        hier = self.RICO.get_hierarchy(uxid, layout=True)
        self.sort(hier.root)

        root = Node(label='[ROOT]', 
                    box=[0, 0, Hierarchy.MAX_WIDTH, Hierarchy.MAX_HEIGHT],
                    children=[])
        dfs = [x for x in hier.dfs() if (not x.is_root and not isinstance(x, LayoutNode))]
        for i, node in enumerate(dfs):
            c = deepcopy(node)
            c.parent = root
            c.child_idx = i
            c.is_leaf = True
            c.level = 1
            c.children = []
            root.children += [c]
        return RicoHierarchy(root)
        
# class FiveClassesDataset(RicoDataset):

#     def __init__(self, *args, **kwargs):
#         super(FiveClassesDataset, self).__init__(*args, **kwargs) 
#         self.key = self.sort_key

#     @staticmethod
#     def sort_key(x):
#         x1, y1, x2, y2 = x.box 
#         center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
#         return (center[1], center[0])


##################################################################################
class ToolbarDataset(RicoDataset):
    
    def __init__(self, *args, **kwargs):
        super(ToolbarDataset, self).__init__(*args, **kwargs)

    
    def get_hierarchy(self, uxid):
        try:
            hier = self.RICO.get_hierarchy(uxid, layout=self.layout_nodes)
        except Exception as e:
            print(e)
            raise RicoError(f'Unsupported hierarchy: UI #{uxid}')

        toolbars = [x for x in hier.nodes if x.label == 'Toolbar']

        if len(toolbars) != 1:
            raise RicoError(f'Unsupported toolbar num: {len(toolbars)}')

        toolbar = toolbars[0]

        if self.flatten:
            toolbar_node = Node(label='Toolbar', box=toolbar.box, children=[])
            root = Node(label='?', box=[0, 0, Hierarchy.MAX_WIDTH, Hierarchy.MAX_HEIGHT])
            root.children = [toolbar_node] + self.RICO.flattened(toolbar, key=lambda x: (x.box[0], x.box[1])).root.children
            o = RicoHierarchy(root)
        
        else:
            o = RicoHierarchy(toolbar)

        return o

#################################################################################
class RicoFlatDataset(RicoDataset):
    def __init__(self, *args, **kwargs):
        super(RicoFlatDataset, self).__init__(*args, **kwargs)
        self.extract_hier = kwargs.pop('extract_hier', False)
        self.sort_children = kwargs.pop('sort_children', True) 
        
    def get_hierarchy(self, uxid):
        hier = self.RICO.get_rico_flattened_hierarchy(uxid, extract_hier=self.extract_hier)
        o = RicoHierarchy(hier)   # convert Node object to Hierarchy object
        if self.sort_children:
            key=lambda x: (x.box[0], x.box[1])
            o.root.children = sorted(o.root.children, key=key)
        return o

class RicoHierDataset(RicoDataset):
    def __init__(self, *args, **kwargs):
        super(RicoHierDataset, self).__init__(*args, **kwargs)
        self.extract_hier = kwargs.pop('extract_hier', True)
        self.sort_children = kwargs.pop('sort_children', True) 
        
    def get_hierarchy(self, uxid):
        hier = self.RICO.get_rico_hierarchy(uxid, extract_hier=self.extract_hier)
        o = RicoHierarchy(hier)   # convert Node object to Hierarchy object
        if self.sort_children:
            key=lambda x: (x.box[0], x.box[1])
            o.root.children = sorted(o.root.children, key=key)        
        return o    
##################################################################################
class ABCDataset(RicoDataset):
    """3"""
    def __init__(self, *args, **kwargs):
        super(ABCDataset, self).__init__(*args, **kwargs)

    def get_hierarchy(self, uxid):
        with open(f'{self.rico_root}/ABC/{uxid}.json') as fp:
            o = RicoHierarchy(json.load(fp))

        return o


class ABCv2Dataset(RicoDataset):
    """3+8"""
    def __init__(self, *args, **kwargs):
        super(ABCv2Dataset, self).__init__(*args, **kwargs)

    def get_hierarchy(self, uxid):
        with open(f'{self.rico_root}/ABCv2/{uxid}.json') as fp:
            o = RicoHierarchy(json.load(fp))

        return o


class ABC34Dataset(RicoDataset):
    """3+4"""
    def __init__(self, *args, **kwargs):
        super(ABC34Dataset, self).__init__(*args, **kwargs)

    def get_hierarchy(self, uxid):
        with open(f'{self.rico_root}/ABC34/{uxid}.json') as fp:
            o = RicoHierarchy(json.load(fp))

        return o

    def group_id(self, uxid):
        if 0 <= uxid < 1000:
            return 'ABC3'
        elif 1000 <= uxid < 2000:
            return 'ABCD4'
        else:
            raise ValueError(f'Unknown UXID: {uxid}')


class ABCDDataset(RicoDataset):
    """4"""
    def __init__(self, *args, **kwargs):
        super(ABCDDataset, self).__init__(*args, **kwargs)

    def get_hierarchy(self, uxid):
        with open(f'{self.rico_root}/ABCD/{uxid}.json') as fp:
            o = RicoHierarchy(json.load(fp))

        return o


class ABC8Dataset(RicoDataset):
    """8"""
    def __init__(self, *args, **kwargs):
        super(ABC8Dataset, self).__init__(*args, **kwargs)

    def get_hierarchy(self, uxid):
        with open(f'{self.rico_root}/ABC8/{uxid}.json') as fp:
            o = RicoHierarchy(json.load(fp))

        return o


##################################################################################
if __name__ == '__main__':
    EXP = 'toolbar_train' # 5classes
    RICO_ROOT = os.environ['RICO_ROOT']
    index_path = f'{RICO_ROOT}/index/index_{EXP}.txt'

    dataset = FiveClassesDataset(RICO_ROOT, index_path, ['uxid', 'object'], is_train=True,
                                 layout_nodes=False, flatten=True)

    samples = random.sample(list(dataset), min(100, len(dataset)))

    html = HTML('/5classes', '5classes', base_url='www', overwrite=True)
    html.add_bigtable(6).add([dict(id=id, tree=str(o), layout=o.plot()) for (id, o) in samples])
    html.save()
    html.show('localhost:99')