import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
import random, time, os, gc, copy, argparse, yaml, datetime, logging, json
from pathlib import Path
from utils import *
from models.synth_model import SynthModel
from datasets import SynthDataset
from tqdm import tqdm

# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)
date = datetime.datetime.now().strftime('%m%d-%H%M')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    
    # general arguments
    parser.add_argument('--exp_name', help='experiment name', type=str, default='demo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    
    #dataset
    parser.add_argument('--data_file', type=str, help='dataset file', default='molecule_synth_data/cid-smile-label.txt')
    
    # training arguments
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=256)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', help='weight decay value', type=float, default=0.0001)
    parser.add_argument('--early_stop_threshold', help='number of epochs to early stop if no best performance is found', type=int, default=50)
    parser.add_argument('--pos_threshold', help='threshold to seperate postive and negative labels', type=float, default=0.5)
    
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
    formater = logging.Formatter("%(asctime)s - %(message)s")
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
    for cids, smiles, labels in val_loader:
        y_true = labels
        output = model(cids, smiles)
        _, predicts = torch.max(output, 1)
        predicts = predicts.detach().cpu()
        assert len(y_true) == len(predicts), print(len(y_true), len(predicts))
        correct += (predicts == y_true).sum().item()
        total_num += len(y_true)
    acc = correct / total_num
    
    model.train()
    return acc


def train(model, train_loader, val_loader, num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args):
    model.train()
    
    n_bad = 0
    best_acc = 0
    all_loss = []
    
    for epoch in range(num_epochs):
        start = time.time()
        acc = evaluate(model, val_loader, logger, args)
        end_test = time.time()
        if acc < best_acc:
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
        for cids, smiles, labels in tqdm(train_loader, dynamic_ncols=True):
            y_true = labels.to(device)
            output = model(cids, smiles)
            loss = criterion(output, y_true)
            losses.append(toCPU(loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(sum(losses) / len(losses))
        all_loss.append(sum(losses) / len(losses))
        torch.save(model.state_dict(), exp_dir / 'last_checkpoint.pt')
        end_epoch = time.time()
        logger.info(f'Epoch [{epoch + 1}/{num_epochs}]: loss: {sum(losses) / len(losses):.4f}; acc: {acc}; train time: {sec2min_sec(end_epoch - end_test)}')

    return all_loss

def precompute(model, data_loader):
    for cids, smiles, labels in tqdm(data_loader, desc='precompute', dynamic_ncols=True):
        output = model(cids, smiles)

def get_pos_neg(dataset, indices=None):
    pos, neg = 0, 0
    if indices is None:
        for cid, smile, label in dataset:
            if label == 1:
                pos += 1
            else:
                neg += 1
    else:
        for i in indices:
            cid, smile, label = dataset[i]
            if label == 1:
                pos += 1
            else:
                neg += 1
    
    return pos, neg

def get_target_label_indices(dataset, target=0):
    idx = []
    for i, (cid, smile, label) in enumerate(dataset):
        if label == target:
            idx.append(i)
    
    return idx

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
    log_dir = root / 'synth_log'
    exp_name = f'{args.exp_name}-{date}'
    # exp_name = args.exp_name
    exp_dir = log_dir / exp_name
    if not exp_dir.is_dir():
        print(f"Creating new log-directory: {exp_dir}")
        exp_dir.mkdir(parents=True)
    logger = get_logger(exp_dir / 'log.txt')
    save_args(args, exp_dir / 'config.yml')
    os.system(f'cp models.py {str(exp_dir)}/')
    
    # load datasets
    all_data = SynthDataset(args.data_file)
    
    # balance pos and neg samples
    pos, neg = get_pos_neg(all_data)
    logger.info(f'Original dataset: pos: {pos/len(all_data)*100:.2f}%; neg: {neg/len(all_data)*100:.2f}%')
    pos_idx = get_target_label_indices(all_data, 1)
    neg_idx = get_target_label_indices(all_data, 0)
    assert len(pos_idx) == pos and len(neg_idx) == neg, print(len(pos_idx), pos, len(neg_idx), neg)
    balanced_pos_idx = random.sample(pos_idx, len(neg_idx))
    all_idx = balanced_pos_idx + neg_idx
    random.shuffle(all_idx)
    pos, neg = get_pos_neg(all_data, all_idx)
    logger.info(f'Balanced dataset: pos: {pos/len(all_idx)*100:.2f}%; neg: {neg/len(all_idx)*100:.2f}%')
    
    # all_idx = get_random_indices(len(all_data), args.seed)
    train_idx = all_idx[:int(len(all_idx) * 0.8)]
    val_idx = all_idx[int(len(all_idx) * 0.8): int(len(all_idx) * 0.9)]
    test_idx = all_idx[int(len(all_idx) * 0.9):]
    logger.info(f'all_data: {len(all_idx)}; train_data: {len(train_idx)}; val_data: {len(val_idx)}; test_data: {len(test_idx)}')
    
    train_loader = DataLoader(Subset(all_data, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(Subset(all_data, val_idx), batch_size=args.batch_size, num_workers=8)
    test_loader = DataLoader(Subset(all_data, test_idx), batch_size=args.batch_size, num_workers=8)
    
    # for data in train_loader:
    #     print(data)
    #     input()
    
    # load model and set hyperparameters
    model = SynthModel(device=device)
    model.to(device)
    print(model.classifier)
    precompute(model, train_loader)
    precompute(model, val_loader)
    precompute(model, test_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, verbose=True)
    criterion = torch.nn.CrossEntropyLoss()
    
    # training
    logger.info(f'Experiment name: {exp_name}')
    logger.info(f'LR: {args.lr}, BS: {args.batch_size}, free Paras.: {count_parameters(model)}, n_epochs: {args.num_epochs}')
    train(model, train_loader, val_loader, args.num_epochs, optimizer, scheduler, criterion, device, exp_dir, logger, args)
    
    # test
    test_acc = evaluate(model, test_loader, logger, args)
    logger.info(f'test acc = {test_acc:.4f}')
    
    
    