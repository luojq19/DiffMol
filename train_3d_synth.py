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

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    
    # general arguments
    parser.add_argument('--exp_name', help='experiment name', type=str, default='demo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    
    #dataset
    parser.add_argument('--label_file', type=str, help='dataset label file', default='molecule_synth_data/cid-smile-label170.txt')
    parser.add_argument('--sdf_path', type=str, help='directory storing .sdf files', default='molecule_synth_data/pubchem-3d/pubchem-cid-sdf')
    parser.add_argument('--preprocessed_path', type=str, help='preprocessed data path', default='molecule_synth_data/synth3ddataset-balanced-noised-100.pt')
    parser.add_argument('--ft_data_path', type=str, help='preprocessed data path for fine tune test', default='molecule_synth_data/crossdocked100k_balanced.pt')
    parser.add_argument('--atom_type_only', action='store_true')
    parser.add_argument('--pos_only', action='store_true')
    
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=128)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', help='weight decay value', type=float, default=0.0001)
    parser.add_argument('--early_stop_threshold', help='number of epochs to early stop if no best performance is found', type=int, default=30)
    parser.add_argument('--pos_threshold', help='threshold to seperate postive and negative labels', type=float, default=0.5)
    parser.add_argument('--not_balance', action='store_true')
    
    # model arguments
    parser.add_argument('--num_layers', type=int, help='number of egnn layers', default=3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--fine_tune', action='store_true')
    parser.add_argument('--no_edge_index', action='store_true')
    
    # debug or custom usage
    parser.add_argument('--custom', action='store_true')    

    args = parser.parse_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    else:
        config_dict = {}
        
    return args

def save_args(args, save_path):
    '''save argument configurations to .yml file'''
    if 'config' in args.__dict__:
        args.__dict__.pop('config')
    with open(save_path, 'w') as f:
        yaml.dump(args.__dict__, f)
        
def get_logger(log_file):
    '''return a logger to output on the console and save to the log file in the same time'''
    logger = logging.getLogger()
    logging_filename = log_file
    formater = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(filename=logging_filename, mode='w')
    console_handler = logging.StreamHandler()
    file_handler.setFormatter(formater)
    console_handler.setFormatter(formater)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    
    return logger

def evaluate(model, val_loader, logger, args):
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

def train(model, train_loader, val_loader, num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args):
    model.train()
    n_bad = 0
    best_acc = 0
    all_loss = []
    epsilon = 1e-4
    for epoch in range(num_epochs):
        start = time.time()
        acc, fp_rate, fn_rate, _ = evaluate(model, val_loader, logger, args)
        end_test = time.time()
        if acc < best_acc + epsilon:
            n_bad += 1
            if n_bad >= args.early_stop_threshold:
                logger.info(f'No performance improvement for {args.early_stop_threshold} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! acc={acc:.4f}')
            n_bad = 0
            best_acc = acc
            torch.save(model.state_dict(), exp_dir / 'best_checkpoint.pt')
        losses = []
        for data in tqdm(train_loader, dynamic_ncols=True, desc='training'):
            data = data.to(device)
            y_true = data.y
            atom_feature_full = torch.nn.functional.one_hot(data.atom_feature_full, model.atom_feature_dim)
            output = model(data.pos, atom_feature_full, data.edge_index, 
                           t=data.timestep, batch=data.batch if hasattr(data, 'batch') else None)
            loss = criterion(output, y_true)
            losses.append(toCPU(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(sum(losses) / len(losses))
        all_loss.append(sum(losses) / len(losses))
        torch.save(model.state_dict(), exp_dir / 'last_checkpoint.pt')
        end_epoch = time.time()
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]: loss: {sum(losses) / len(losses):.4f}; acc: {acc:.4f}; fp: {fp_rate:.4f}; fn: {fn_rate:.4f}; train time: {sec2min_sec(end_epoch - end_test)}')
    
    return all_loss

def fine_tune(model, train_loader, val_loader, ft_loader, num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args):
    model.train()
    n_bad = 0
    best_acc = 0
    all_loss = []
    epsilon = 1e-4
    for epoch in range(num_epochs):
        start = time.time()
        acc, fp_rate, fn_rate, _ = evaluate(model, val_loader, logger, args)
        acc_ft, fp_rate_ft, fn_rate_ft, _ = evaluate(model, ft_loader, logger, args)
        end_test = time.time()
        if acc < best_acc + epsilon:
            n_bad += 1
            if n_bad >= args.early_stop_threshold:
                logger.info(f'No performance improvement for {args.early_stop_threshold} epochs. Early stop training!')
                break
        else:
            logger.info(f'New best performance found! acc={acc:.4f}')
            n_bad = 0
            best_acc = acc
            torch.save(model.state_dict(), exp_dir / 'best_checkpoint.pt')
        losses = []
        for data in tqdm(train_loader, dynamic_ncols=True, desc='training'):
            data = data.to(device)
            y_true = data.y
            atom_feature_full = torch.nn.functional.one_hot(data.atom_feature_full, model.atom_feature_dim)
            output = model(data.pos, atom_feature_full, data.edge_index, 
                           t=data.timestep, batch=data.batch if hasattr(data, 'batch') else None)
            loss = criterion(output, y_true)
            losses.append(toCPU(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(sum(losses) / len(losses))
        all_loss.append(sum(losses) / len(losses))
        torch.save(model.state_dict(), exp_dir / 'last_checkpoint.pt')
        end_epoch = time.time()
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]: loss: {sum(losses) / len(losses):.4f}; acc: {acc:.4f}; fp: {fp_rate:.4f}; fn: {fn_rate:.4f}; train time: {sec2min_sec(end_epoch - end_test)}')
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]: loss: {sum(losses) / len(losses):.4f}; acc_ft: {acc_ft:.4f}; fp_ft: {fp_rate_ft:.4f}; fn_ft: {fn_rate_ft:.4f}')
    
    return all_loss    
    
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

def get_ft_loader(args):
    all_data = Synth3DDataset(sdf_path=args.sdf_path, 
                              label_file=args.label_file, 
                              preprocessed_path=args.ft_data_path,
                              ignore_cid_subset=True,
                              simplify=True)
    
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
    logger.info(f'fine tune test data: {len(test_idx)}')
    
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size, num_workers=8)
    
    return test_loader

if __name__ == '__main__':
    # parse arguments
    args = get_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else "cpu")
    start_overall = time.time()
    print(args)
    
    # set seeds
    seed_all(args.seed)
    
    # set directories
    root = Path.cwd()
    log_dir = root / '3d_synth_noised_log'
    exp_name = f'{args.exp_name}-{date}'
    # exp_name = args.exp_name
    exp_dir = log_dir / exp_name
    if not exp_dir.is_dir():
        print(f"Creating new log-directory: {exp_dir}")
        exp_dir.mkdir(parents=True)
    logger = get_logger(exp_dir / 'log.txt')
    save_args(args, exp_dir / 'config.yml')
    os.system(f'cp -r models {str(exp_dir)}/')
    
    # load datasets
    all_data = Synth3DDataset(sdf_path=args.sdf_path, 
                              label_file=args.label_file, 
                              preprocessed_path=args.preprocessed_path,
                              ignore_cid_subset=True,
                              simplify=False,
                              atom_type_only=args.atom_type_only,
                              pos_only=args.pos_only)
    # print(all_data[1])
    # print(all_data[1].atom_feature_full)
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
    if args.fine_tune:
        ft_loader = get_ft_loader(args)
    
    # load model and set hyperparameters
    model = SynthEGNN(num_layers=args.num_layers, dropout=args.dropout, no_edge_index=args.no_edge_index,
                      atom_feature_dim=all_data.atom_feature_dim)
    
    if args.custom:
        ckpt = torch.load(args.checkpoint)
        print(f'Loading checkpoint {args.checkpoint}')
        model.load_state_dict(ckpt)
        model.to(device)
        test_acc, fp, fn, results = evaluate(model, test_loader, logger, args)
        logger.info(f'test acc: {test_acc:.4f}; false positive: {fp:.4f}; false negative: {fn:.4f}')
        exit()
        
    if args.checkpoint is not None and args.eval:
        logger.info(f'loading model from {args.checkpoint}')
        state_dict = torch.load(args.checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        logger.info(f'loading checkpoint from {args.checkpoint}')
        test_acc, fp, fn, results = evaluate(model, train_loader, logger, args)
        # print(results['y_pred'], len(results['y_true']), len(results['y_pred']))
        logger.info(f'test acc: {test_acc:.4f}; false positive: {fp:.4f}; false negative: {fn:.4f}')
        from utils import eval_all_mol
        res = eval_all_mol(Subset(all_data, train_idx), results['y_true'], results['y_pred'], save_path=exp_dir / 'smiles_gt_pred.txt')
        # for k, v in res.items():
        #     logger.info(f'{k}: {v}')
        with open(exp_dir / 'test_results.json', 'w') as f:
            json.dump(results, f)
        exit()
        
    if args.checkpoint is not None and args.fine_tune:
        ckpt = torch.load(args.checkpoint)
        model.load_state_dict(ckpt)
        logger.info(f'Load checkpoint from {args.checkpoint}')
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    # train
    logger.info(f'Experiment name: {exp_name}')
    logger.info(f'LR: {args.lr}, BS: {args.batch_size}, free Paras.: {count_parameters(model)}, n_epochs: {args.num_epochs}')
    if args.fine_tune:
        logger.info('Begin fine tune:')
        fine_tune(model, train_loader, val_loader, ft_loader, args.num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args)
    else:
        train(model, train_loader, val_loader, args.num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args)
    
    # test
    best_ckpt = torch.load(exp_dir / 'best_checkpoint.pt')
    model.load_state_dict(best_ckpt)
    test_acc, fp, fn, _ = evaluate(model, test_loader, logger, args)
    logger.info(f'test acc: {test_acc:.4f}; false positive: {fp:.4f}; false negative: {fn:.4f}')
    if args.fine_tune:
        test_acc, fp, fn, _ = evaluate(model, ft_loader, logger, args)
        logger.info(f'Original dataset test acc: {test_acc:.4f}; false positive: {fp:.4f}; false negative: {fn:.4f}')
    