import torch
import numpy as np
import random
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from sklearn.metrics import confusion_matrix

# https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
def seed_all(seed=42):
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return None

# move torch/GPU tensor to numpy/CPU
def toCPU(data):
    return data.cpu().detach().numpy()

# count number of free parameters in the network
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def sec2min_sec(t):
    mins = int(t) // 60
    secs = int(t) % 60
    
    return f'{mins}[min]{secs}[sec]'

def get_random_indices(length, seed=123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices


ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
BOND_TYPES = {
    BondType.UNSPECIFIED: 0,
    BondType.SINGLE: 1,
    BondType.DOUBLE: 2,
    BondType.TRIPLE: 3,
    BondType.AROMATIC: 4,
}
BOND_NAMES = {v: str(k) for k, v in BOND_TYPES.items()}
HYBRIDIZATION_TYPE = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']
HYBRIDIZATION_TYPE_ID = {s: i for i, s in enumerate(HYBRIDIZATION_TYPE)}

AROMATIC_FEAT_MAP_IDX = ATOM_FAMILIES_ID['Aromatic']

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}

def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'add_aromatic':
        # self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 9, 15, 16, 17])  # H, C, N, O, F, P, S, Cl
        if (int(atom_num), bool(is_aromatic)) in MAP_ATOM_TYPE_AROMATIC_TO_INDEX:
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
        else:
            # print(int(atom_num), bool(is_aromatic))
            return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[(1, False)]
    else:
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]

class PDBProtein(object):
    AA_NAME_SYM = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
        'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
        'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    }

    AA_NAME_NUMBER = {
        k: i for i, (k, _) in enumerate(AA_NAME_SYM.items())
    }

    BACKBONE_NAMES = ["CA", "C", "N", "O"]

    def __init__(self, data, mode='auto'):
        super().__init__()
        if (data[-4:].lower() == '.pdb' and mode == 'auto') or mode == 'path':
            with open(data, 'r') as f:
                self.block = f.read()
        else:
            self.block = data

        self.ptable = Chem.GetPeriodicTable()

        # Molecule properties
        self.title = None
        # Atom properties
        self.atoms = []
        self.element = []
        self.atomic_weight = []
        self.pos = []
        self.atom_name = []
        self.is_backbone = []
        self.atom_to_aa_type = []
        # Residue properties
        self.residues = []
        self.amino_acid = []
        self.center_of_mass = []
        self.pos_CA = []
        self.pos_C = []
        self.pos_N = []
        self.pos_O = []

        self._parse()

    def _enum_formatted_atom_lines(self):
        for line in self.block.splitlines():
            if line[0:6].strip() == 'ATOM':
                element_symb = line[76:78].strip().capitalize()
                if len(element_symb) == 0:
                    element_symb = line[13:14]
                yield {
                    'line': line,
                    'type': 'ATOM',
                    'atom_id': int(line[6:11]),
                    'atom_name': line[12:16].strip(),
                    'res_name': line[17:20].strip(),
                    'chain': line[21:22].strip(),
                    'res_id': int(line[22:26]),
                    'res_insert_id': line[26:27].strip(),
                    'x': float(line[30:38]),
                    'y': float(line[38:46]),
                    'z': float(line[46:54]),
                    'occupancy': float(line[54:60]),
                    'segment': line[72:76].strip(),
                    'element_symb': element_symb,
                    'charge': line[78:80].strip(),
                }
            elif line[0:6].strip() == 'HEADER':
                yield {
                    'type': 'HEADER',
                    'value': line[10:].strip()
                }
            elif line[0:6].strip() == 'ENDMDL':
                break  # Some PDBs have more than 1 model.

    def _parse(self):
        # Process atoms
        residues_tmp = {}
        for atom in self._enum_formatted_atom_lines():
            if atom['type'] == 'HEADER':
                self.title = atom['value'].lower()
                continue
            self.atoms.append(atom)
            atomic_number = self.ptable.GetAtomicNumber(atom['element_symb'])
            next_ptr = len(self.element)
            self.element.append(atomic_number)
            self.atomic_weight.append(self.ptable.GetAtomicWeight(atomic_number))
            self.pos.append(np.array([atom['x'], atom['y'], atom['z']], dtype=np.float32))
            self.atom_name.append(atom['atom_name'])
            self.is_backbone.append(atom['atom_name'] in self.BACKBONE_NAMES)
            self.atom_to_aa_type.append(self.AA_NAME_NUMBER[atom['res_name']])

            chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
            if chain_res_id not in residues_tmp:
                residues_tmp[chain_res_id] = {
                    'name': atom['res_name'],
                    'atoms': [next_ptr],
                    'chain': atom['chain'],
                    'segment': atom['segment'],
                }
            else:
                assert residues_tmp[chain_res_id]['name'] == atom['res_name']
                assert residues_tmp[chain_res_id]['chain'] == atom['chain']
                residues_tmp[chain_res_id]['atoms'].append(next_ptr)

        # Process residues
        self.residues = [r for _, r in residues_tmp.items()]
        for residue in self.residues:
            sum_pos = np.zeros([3], dtype=np.float32)
            sum_mass = 0.0
            for atom_idx in residue['atoms']:
                sum_pos += self.pos[atom_idx] * self.atomic_weight[atom_idx]
                sum_mass += self.atomic_weight[atom_idx]
                if self.atom_name[atom_idx] in self.BACKBONE_NAMES:
                    residue['pos_%s' % self.atom_name[atom_idx]] = self.pos[atom_idx]
            residue['center_of_mass'] = sum_pos / sum_mass

        # Process backbone atoms of residues
        for residue in self.residues:
            self.amino_acid.append(self.AA_NAME_NUMBER[residue['name']])
            self.center_of_mass.append(residue['center_of_mass'])
            for name in self.BACKBONE_NAMES:
                pos_key = 'pos_%s' % name  # pos_CA, pos_C, pos_N, pos_O
                if pos_key in residue:
                    getattr(self, pos_key).append(residue[pos_key])
                else:
                    getattr(self, pos_key).append(residue['center_of_mass'])

    def to_dict_atom(self):
        return {
            'element': np.array(self.element, dtype=np.long),
            'molecule_name': self.title,
            'pos': np.array(self.pos, dtype=np.float32),
            'is_backbone': np.array(self.is_backbone, dtype=np.bool),
            'atom_name': self.atom_name,
            'atom_to_aa_type': np.array(self.atom_to_aa_type, dtype=np.long)
        }

    def to_dict_residue(self):
        return {
            'amino_acid': np.array(self.amino_acid, dtype=np.long),
            'center_of_mass': np.array(self.center_of_mass, dtype=np.float32),
            'pos_CA': np.array(self.pos_CA, dtype=np.float32),
            'pos_C': np.array(self.pos_C, dtype=np.float32),
            'pos_N': np.array(self.pos_N, dtype=np.float32),
            'pos_O': np.array(self.pos_O, dtype=np.float32),
        }

    def query_residues_radius(self, center, radius, criterion='center_of_mass'):
        center = np.array(center).reshape(3)
        selected = []
        for residue in self.residues:
            distance = np.linalg.norm(residue[criterion] - center, ord=2)
            print(residue[criterion], distance)
            if distance < radius:
                selected.append(residue)
        return selected

    def query_residues_ligand(self, ligand, radius, criterion='center_of_mass'):
        selected = []
        sel_idx = set()
        # The time-complexity is O(mn).
        for center in ligand['pos']:
            for i, residue in enumerate(self.residues):
                distance = np.linalg.norm(residue[criterion] - center, ord=2)
                if distance < radius and i not in sel_idx:
                    selected.append(residue)
                    sel_idx.add(i)
        return selected

    def residues_to_pdb_block(self, residues, name='POCKET'):
        block = "HEADER    %s\n" % name
        block += "COMPND    %s\n" % name
        for residue in residues:
            for atom_idx in residue['atoms']:
                block += self.atoms[atom_idx]['line'] + "\n"
        block += "END\n"
        return block


def parse_pdbbind_index_file(path):
    pdb_id = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('#'): continue
        pdb_id.append(line.split()[0])
    return pdb_id


def parse_sdf_file(path):
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    # read mol
    if path.endswith('.sdf'):
        rdmol = Chem.MolFromMolFile(path, sanitize=False)
    elif path.endswith('.mol2'):
        rdmol = Chem.MolFromMol2File(path, sanitize=False)
    else:
        raise ValueError
    Chem.SanitizeMol(rdmol)
    rdmol = Chem.RemoveHs(rdmol)

    # Remove Hydrogens.
    # rdmol = next(iter(Chem.SDMolSupplier(path, removeHs=True)))
    rd_num_atoms = rdmol.GetNumAtoms()
    feat_mat = np.zeros([rd_num_atoms, len(ATOM_FAMILIES)], dtype=np.compat.long)
    for feat in factory.GetFeaturesForMol(rdmol):
        feat_mat[feat.GetAtomIds(), ATOM_FAMILIES_ID[feat.GetFamily()]] = 1

    # Get hybridization in the order of atom idx.
    hybridization = []
    for atom in rdmol.GetAtoms():
        hybr = str(atom.GetHybridization())
        idx = atom.GetIdx()
        hybridization.append((idx, hybr))
    hybridization = sorted(hybridization)
    hybridization = [v[1] for v in hybridization]

    ptable = Chem.GetPeriodicTable()

    pos = np.array(rdmol.GetConformers()[0].GetPositions(), dtype=np.float32)
    element = []
    accum_pos = 0
    accum_mass = 0
    # print(f'rd_num_atoms: {rd_num_atoms}')
    for atom_idx in range(rd_num_atoms):
        atom = rdmol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        # print(f'atom: {atom}; atom_num: {atom_num}')
        element.append(atom_num)
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass
    element = np.array(element, dtype=np.int)

    # in edge_type, we have 1 for single bond, 2 for double bond, 3 for triple bond, and 4 for aromatic bond.
    row, col, edge_type = [], [], []
    for bond in rdmol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [BOND_TYPES[bond.GetBondType()]]

    edge_index = np.array([row, col], dtype=np.long)
    edge_type = np.array(edge_type, dtype=np.long)

    perm = (edge_index[0] * rd_num_atoms + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]

    data = {
        'smiles': Chem.MolToSmiles(rdmol),
        'element': element,
        'pos': pos,
        'bond_index': edge_index,
        'bond_type': edge_type,
        'center_of_mass': center_of_mass,
        'atom_feature': feat_mat,
        'hybridization': hybridization
    }
    return data

def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

def eval_molecules_atom_nums(dataset, y_true, y_pred):
    all_atom_nums = []
    for data in dataset:
        all_atom_nums.append(len(data.pos))
    assert len(all_atom_nums) == len(y_true) == len(y_pred), print(len(all_atom_nums), len(y_true), len(y_pred))
    cfm = confusion_matrix(y_true, y_pred)
    print(f'cfm: {cfm}')
    gt_pos_atom_nums, gt_neg_atom_nums = [], []
    tp, tn, fp, fn = [], [], [], []
    for i in range(len(all_atom_nums)):
        if y_true[i] == 1:
            gt_pos_atom_nums.append(all_atom_nums[i])
            if y_pred[i] == 1:
                tp.append(all_atom_nums[i])
            else:
                fn.append(all_atom_nums[i])
        else:
            gt_neg_atom_nums.append(all_atom_nums[i])
            if y_pred[i] == 1:
                fp.append(all_atom_nums[i])
            else:
                tn.append(all_atom_nums[i])
    res = {'tp': np.mean(tp),
           'tn': np.mean(tn),
           'fp': np.mean(fp),
           'fn': np.mean(fn),
           'pred_pos': np.mean(tp+fp),
           'pred_neg': np.mean(tn+fn),
           'gt_pos': np.mean(gt_pos_atom_nums),
           'gt_neg': np.mean(gt_neg_atom_nums),
           'all': np.mean(all_atom_nums)}
    
    return res

def eval_molecules(data_list):
    results = {'avg_atom_num': 0,
               'atom_nums': [],
               'has_aromatic': [],
               'aromatic_rate': 0,
               'has_benzene': [],
               'benzene_rate': 0}
    benzene_pattern = Chem.MolFromSmarts('c1ccccc1')
    for mol in data_list:
        atom_num = mol.pos.shape[0]
        # print(atom_num, mol.pos.shape)
        results['atom_nums'].append(atom_num)
        rdmol = Chem.MolFromSmiles(mol.smiles)
        aromatic_atoms = rdmol.GetAromaticAtoms()
        results['has_aromatic'].append(1 if aromatic_atoms else 0)
        results['has_benzene'].append(1 if rdmol.HasSubstructMatch(benzene_pattern) else 0)
        # print(1)
        # results['density'].append(compute_mol_density(rdmol))
    results['avg_atom_num'] = int(np.mean(results['atom_nums']))
    results['aromatic_rate'] = np.sum(results['has_aromatic']) / len(data_list)
    results['benzene_rate'] = np.sum(results['has_benzene']) / len(data_list)
    # results['avg_density'] = np.mean(results['density'])
    
    return results

def eval_all_mol(all_data, y_true, y_pred, save_path='tmp.txt'):
    with open(save_path, 'w') as f:
        for i in range(len(y_true)):
            f.write(f'{all_data[i].smiles} {y_true[i]} {y_pred[i]}\n')
    tp, tn, fp, fn = [], [], [], []
    gt_pos_idx, gt_neg_idx = [], []
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp.append(i)
            gt_pos_idx.append(i)
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn.append(i)
            gt_pos_idx.append(i)
        elif y_true[i] == 0 and y_pred[i] == 1:
            fp.append(i)
            gt_neg_idx.append(i)
        elif y_true[i] == 0 and y_pred[i] == 0:
            tn.append(i)
            gt_neg_idx.append(i)
        else:
            raise NotImplementedError
    print(f'tp: {len(tp)}, tn: {len(tn)}, fp: {len(fp)}, fn: {len(fn)}')
    print(all_data[0])
    # input()
    tp_res = eval_molecules([all_data[i] for i in tp])
    tn_res = eval_molecules([all_data[i] for i in tn])
    fp_res = eval_molecules([all_data[i] for i in fp])
    fn_res = eval_molecules([all_data[i] for i in fn])
    pred_pos_res = eval_molecules([all_data[i] for i in tp + fp])
    pred_neg_res = eval_molecules([all_data[i] for i in tn + fn])
    all_res = eval_molecules(all_data)
    gt_pos_res = eval_molecules([all_data[i] for i in gt_pos_idx])
    gt_neg_res = eval_molecules([all_data[i] for i in gt_neg_idx])
    print(f'avg atom nums:\ntp: {tp_res["avg_atom_num"]}\ntn: {tn_res["avg_atom_num"]}\nfp: {fp_res["avg_atom_num"]}\nfn: {fn_res["avg_atom_num"]}\nall: {all_res["avg_atom_num"]}\ngt_pos: {gt_pos_res["avg_atom_num"]}\ngt_neg: {gt_neg_res["avg_atom_num"]}\npred_pos: {pred_pos_res["avg_atom_num"]}\npred_neg: {pred_neg_res["avg_atom_num"]}')
    # print(f'atom nums:\ntp: {tp_res["atom_nums"][:30]}\ntn: {tn_res["atom_nums"][:30]}\nfp: {fp_res["atom_nums"][:30]}\nfn: {fn_res["atom_nums"][:30]}\ngt_pos: {gt_pos_res["atom_nums"][:30]}\ngt_neg: {gt_neg_res["atom_nums"][:30]}\npred_pos: {pred_pos_res["atom_nums"][:30]}\npred_neg: {pred_neg_res["atom_nums"][:30]}\nall: {all_res["atom_nums"]}')
    print(f'aromatic rate:\ntp: {tp_res["aromatic_rate"]}\ntn: {tn_res["aromatic_rate"]}\nfp: {fp_res["aromatic_rate"]}\nfn: {fn_res["aromatic_rate"]}\nall: {all_res["aromatic_rate"]}\ngt_pos: {gt_pos_res["aromatic_rate"]}\ngt_neg: {gt_neg_res["aromatic_rate"]}\npred_pos: {pred_pos_res["aromatic_rate"]}\npred_neg: {pred_neg_res["aromatic_rate"]}')
    
    print(f'benzene rate:\ntp: {tp_res["benzene_rate"]}\ntn: {tn_res["benzene_rate"]}\nfp: {fp_res["benzene_rate"]}\nfn: {fn_res["benzene_rate"]}\nall: {all_res["benzene_rate"]}\ngt_pos: {gt_pos_res["benzene_rate"]}\ngt_neg: {gt_neg_res["benzene_rate"]}\npred_pos: {pred_pos_res["benzene_rate"]}\npred_neg: {pred_neg_res["benzene_rate"]}')
    
    all_results = {'tp': tp_res,
                   'tn': tn_res,
                   'fp': fp_res,
                   'fn': fn_res,
                   'pred_pos': pred_pos_res,
                   'pred_neg': pred_neg_res,
                   'all': all_res,
                   'gt_pos': gt_pos_res,
                   'gt_neg': gt_neg_res}
    
    return all_results