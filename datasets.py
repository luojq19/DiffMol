import torch
from torch.utils.data import Dataset
from utils import parse_sdf_file, torchify_dict, AROMATIC_FEAT_MAP_IDX, get_index
from utils import MAP_ATOM_TYPE_ONLY_TO_INDEX, MAP_INDEX_TO_ATOM_TYPE_AROMATIC
from tqdm.auto import tqdm
import os, json
import torch_geometric as pyg

class SynthDataset(Dataset):
    def __init__(self, data_file='molecule_synth_data/cid-smile-label.txt'):
        super().__init__()
        self.data = [] # each entry in self.data should be a triplet of [cid, smile, label]
        with open(data_file) as f:
            lines = f.readlines()
        for line in lines:
            cid, smile, label = line.split()
            self.data.append([cid, smile, int(label)])
        
    def __getitem__(self, index):
        cid, smile, label = self.data[index]
        
        return cid, smile, label
    
    def __len__(self):
        return len(self.data)
    
    
class Synth3DDataset(pyg.data.Dataset):
    def __init__(self, 
                 mode='add_aromatic',
                 sdf_path='molecule_synth_data/pubchem-3d/pubchem-cid-sdf/',
                 label_file='molecule_synth_data/cid-smile-label.txt',
                 cid_subset=None,
                 ignore_cid_subset=False,
                 preprocessed_path=None,
                 simplify=False,
                 atom_type_only=False,
                 pos_only=False) -> None:
        super().__init__()
        self.mode = mode
        self.sdf_path = sdf_path
        self.simplify = simplify
        self.atom_type_only = atom_type_only
        self.atom_feature_dim = 8 if atom_type_only else 13
        self.pos_only = pos_only
        if pos_only:
            self.atom_feature_dim = 1
        self.cid2label = {}
        self.raw_data = []
        self.graphs = []
        self.idx2label = []
        self.idx2cids = []
        with open(label_file) as f:
            lines = f.readlines()
        for line in lines:
            if len(line.split()) == 2:
                cid, label = line.split()
            else:
                cid, smile, label = line.split()
            self.cid2label[cid] = int(label)
        if cid_subset is None:
            self.cid_subset = list(self.cid2label.keys())
            # self.cid_subset = list(self.cid2label.keys())[:100] # uncomment for fast debug
        else:
            self.cid_subset = cid_subset
        
        if preprocessed_path is not None:
            print(f'Loading preprocessed dataset from {preprocessed_path}')
            self.raw_data, self.graphs, self.idx2label, self.idx2cids = self.load_from_preprocessed(preprocessed_path, self.cid_subset if not ignore_cid_subset else None)
            print(f'Loaded dataset: {len(self.graphs)} molecules.')
            return
        
        try:
            for cid in tqdm(self.cid_subset, dynamic_ncols=True, desc='Loading sdf'):
                file = f'{cid}.sdf'
                if not os.path.exists(os.path.join(self.sdf_path, file)):
                    continue
                try:
                    if file.split('.')[0] in self.cid_subset or int(file.split('.')[0]) in self.cid_subset:
                        mol_data = parse_sdf_file(os.path.join(self.sdf_path, file))
                        mol_data['cid'] = file.split('.')[0]
                        mol_data['label'] = self.cid2label[mol_data['cid']]
                        self.raw_data.append(torchify_dict(mol_data))
                except Exception as e:
                    print(f'Exception: {e}')
        except KeyboardInterrupt:
                print(f'Terminated.')
        
        for idx, data in enumerate(self.raw_data):
            try:
                element_list = data['element']
                hybridization_list = data['hybridization']
                aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data['atom_feature']]
                feature_full = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
                feature_full = torch.tensor(feature_full)
                self.raw_data[idx]['atom_feature_full'] = feature_full
                graph = pyg.data.Data(x=data['element'], 
                                    y=data['label'],
                                    edge_index=data['bond_index'],
                                    edge_attr=data['bond_type'],
                                    pos=data['pos'],
                                    smiles=data['smiles'],
                                    com=data['center_of_mass'],
                                    atom_feature=data['atom_feature'],
                                    hybridization=data['hybridization'],
                                    atom_feature_full=feature_full,
                                    cid=data['cid'],
                                    timestep=data.get('timestep', torch.tensor([0])).cpu())
                self.graphs.append(graph)
                self.idx2label.append(data['label'])
                self.idx2cids.append(data['cid'])
            except:
                pass

        print(f'Loaded dataset: {len(self.graphs)} molecules.')
    
    def __getitem__(self, index):
        return self.graphs[index]
    
    def __len__(self):
        return len(self.graphs)
    
    def get_idx2label(self):
        return self.idx2label
    
    def get_cids(self):
        return self.idx2cids
    
    def save_data(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.raw_data, path)
        print(f'Saving {len(self.raw_data)} molecules to {path}')
    
    def type_aromatic_idx2type_only_idx(self, idx):
        if type(idx) is int:
            return MAP_ATOM_TYPE_ONLY_TO_INDEX[MAP_INDEX_TO_ATOM_TYPE_AROMATIC[idx][0]]
        else:
            for i in range(len(idx)):
                idx[i] = MAP_ATOM_TYPE_ONLY_TO_INDEX[MAP_INDEX_TO_ATOM_TYPE_AROMATIC[int(idx[i])][0]]
            return idx
    
    def load_from_preprocessed(self, path, cid_subset=None):
        preprocessed = torch.load(path)
        graphs = []
        idx2label, idx2cids = [], []
        for data in tqdm(preprocessed, dynamic_ncols=True, desc='preprocessed'):
            if cid_subset is None or data['cid'] in cid_subset:
                aft = data['atom_feature_full'].cpu()
                if self.atom_type_only:
                    aft = self.type_aromatic_idx2type_only_idx(aft)
                if self.pos_only:
                    aft = torch.zeros_like(aft, dtype=aft.dtype)
                if self.simplify:
                    graph = pyg.data.Data(y=data['label'],
                                        edge_index=data.get('bond_index', torch.tensor([[0,1],[1,0]])),
                                        # edge_attr=data['bond_type'],
                                        pos=data['pos'].cpu().to(torch.float32),
                                        # smiles=data['smiles'],
                                        # com=data['center_of_mass'],
                                        # atom_feature=data['atom_feature'],
                                        # hybridization=data['hybridization'],
                                        atom_feature_full=aft,
                                        cid=data['cid'],
                                        timestep=data.get('timestep', torch.tensor([0])).cpu())
                else:
                    graph = pyg.data.Data(x=data['element'], 
                                        y=data['label'],
                                        edge_index=data['bond_index'],
                                        edge_attr=data['bond_type'],
                                        pos=data['pos'].cpu(),
                                        smiles=data['smiles'],
                                        com=data['center_of_mass'],
                                        atom_feature=data['atom_feature'],
                                        hybridization=data['hybridization'],
                                        atom_feature_full=aft,
                                        cid=data['cid'],
                                        timestep=data.get('timestep', torch.tensor([0])).cpu())
                graphs.append(graph)
                idx2label.append(data['label'])
                idx2cids.append(data['cid'])
        
        return preprocessed, graphs, idx2label, idx2cids
    
if __name__ == '__main__':
    ds = Synth3DDataset(label_file='molecule_synth_data/cid-smile-label170.txt',
                        preprocessed_path='molecule_synth_data/synth3ddataset-balanced-noised-100.pt',
                        ignore_cid_subset=True)
    # for i in range(len(ds)):
    #     # print(ds[i])
    #     print(ds[i].atom_feature_full)
    print(len(ds))
    print(ds[0])
    dl = pyg.loader.DataLoader(ds, batch_size=2)
    for data in dl:
        print(data)
        break
    # ds.save_data('molecule_synth_data/synth3ddataset.pt')
    # available_cids = ds.get_cids()
    # with open('molecule_synth_data/sdf_available_cids.json', 'w') as f:
    #     json.dump(available_cids, fd)
    idx2cid = ds.get_cids()
    print(len(idx2cid), len(set(idx2cid)))

