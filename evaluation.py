import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import random, time, os, gc, copy, argparse, yaml, datetime, logging, json
from pathlib import Path
from utils import *
from models.synth_model import SynthEGNN
from datasets import Synth3DDataset
from tqdm.auto import tqdm
import torch_geometric as pyg
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix


torch.set_num_threads(1)
date = datetime.datetime.now().strftime('%Y%m%d-%H%M')

def evaluate(model, val_loader, logger, args, device='cuda:0'):
    model.eval()
    
    correct = 0
    total_num = 0
    total_y_true, total_pred = [], []
    results = {'cids': []}
    for data in val_loader:
        y_true = data.y
        total_y_true.extend(y_true.tolist())
        cids = data.cid
        results['cids'].extend(list(cids))
        data = data.to(device)
        atom_feature_full = torch.nn.functional.one_hot(data.atom_feature_full, model.atom_feature_dim)
        output = model(data.pos, atom_feature_full, data.edge_index, t=data.timestep,
                       batch=data.batch if hasattr(data, 'batch') else None)
        _, predicts = torch.max(output, 1)
        predicts = predicts.detach().cpu()
        total_pred.extend(predicts.tolist())
        assert len(y_true) == len(predicts), print(len(y_true), len(predicts))
        correct += (predicts == y_true).sum().item()
        total_num += len(y_true)
    acc = correct / total_num
    confmat = confusion_matrix(total_y_true, total_pred)
    false_pos_rate = confmat[0, 1] / total_num
    false_neg_rate = confmat[1, 0] / total_num
    results['y_true'] = total_y_true
    results['y_pred'] = total_pred
    results['acc'] = acc
    
    model.train()
    return acc, false_pos_rate, false_neg_rate, results

def balanced_label_indices(idx2label, logger, seed=42):
    pos_idx, neg_idx = [], []
    for idx, label in enumerate(idx2label):
        if label == 1:
            pos_idx.append(idx)
        else:
            neg_idx.append(idx)
    logger.info(f'Original dataset: {len(pos_idx)} positive, {len(neg_idx)} negative')
    balanced_pos_idx = random.sample(pos_idx, min(len(pos_idx), len(neg_idx)))
    balanced_neg_idx = random.sample(neg_idx, min(len(pos_idx), len(neg_idx)))
    balanced_idx = balanced_pos_idx + balanced_neg_idx
    random.shuffle(balanced_idx)
    logger.info(f'Balanced dataset: {len(balanced_pos_idx)} positive, {len(balanced_neg_idx)} negative, {len(balanced_idx)} in total')
    
    return balanced_idx

# load datasets
all_data = Synth3DDataset(sdf_path=args.sdf_path, 
                            label_file=args.label_file, 
                            preprocessed_path=args.preprocessed_path,
                            ignore_cid_subset=True,
                            simplify=False)
# print(all_data[0])
# input()
# balance pos and neg samples
idx2label = all_data.get_idx2label()
balanced_indices = balanced_label_indices(idx2label, logger, args.seed)
if args.not_balance:
    balanced_indices = [i for i in range(len(idx2label))]
    random.shuffle(balanced_indices)
    logger.info(f'Use original unbalanced data.')
total = len(balanced_indices)
train_idx = balanced_indices[:int(total * 0.8)]
val_idx = balanced_indices[int(total * 0.8): int(total * 0.9)]
test_idx = balanced_indices[int(total * 0.9):]
logger.info(f'all_data: {len(balanced_indices)}; train_data: {len(train_idx)}; val_data: {len(val_idx)}; test_data: {len(test_idx)}')

train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size, num_workers=8)
test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size, num_workers=8)