import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
# import egnn_clean as eg
# from egnn_clean import EGNN
from egnn_pytorch import EGNN_Sparse, EGNN_Sparse_Network
import torch_geometric as pyg
from torch_scatter import scatter_mean
import numpy as np
import torch_cluster

class LMEmbedder():
    def __init__(self, model_name="DeepChem/ChemBERTa-77M-MLM", device=torch.device('cuda:0')):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.embed_dim = self.model.config.to_dict()['vocab_size']
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        self.cache = {}
        
    def embed(self, data, names, batch_size=16, max_len=510):
        data = [d[:max_len] for d in data]
        if set(names).issubset(set(self.cache.keys())):
            # print('using cache')
            features = []
            for n in names:
                features.append(self.cache[n])
        else:
            features = []
            for i in range((len(data) - 1) // batch_size + 1):
                sequences_Example = data[i * batch_size: (i + 1) * batch_size]
                names_Example = names[i * batch_size: (i + 1) * batch_size]
                ids = self.tokenizer.batch_encode_plus(sequences_Example, padding=True)
                input_ids = torch.tensor(ids['input_ids'], device=self.device)
                attention_mask = torch.tensor(ids['attention_mask'], device=self.device)
                with torch.no_grad():
                    # print(input_ids.device, attention_mask.device)
                    embedding = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                for seq_num in range(len(embedding)):
                    # print(len(sequences_Example[seq_num]))
                    seq_len = (attention_mask[seq_num] == 1).sum()
                    try:
                        seq_emb = embedding[seq_num][:seq_len - 1]
                        avg_seq_emb = torch.mean(seq_emb, dim=0)
                        features.append(avg_seq_emb)
                        self.cache[names_Example[seq_num]] = avg_seq_emb
                    except:
                        print(embedding.shape)
                    # avg_seq_emb = torch.mean(seq_emb, dim=0)
                    # features.append(avg_seq_emb)
                    # self.cache[names_Example[i]] = avg_seq_emb
        features = torch.vstack(features)
        
        return features

class SynthSMILESModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embedder = LMEmbedder(device=device)
        self.embed_dim = self.embedder.embed_dim
        self.hidden_dim = [2000, 1500, 1000]
        self.classifier = nn.ModuleList()
        self.classifier.append(nn.Linear(self.embed_dim, self.hidden_dim[0]))
        for i in range(len(self.hidden_dim) - 1):
            self.classifier.append(nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))
        self.classifier.append(nn.Linear(self.hidden_dim[-1], 2))
            
        # self.classifier = nn.Sequential(nn.Linear(self.embed_dim, 1200),
        #                                 nn.ReLU(),
        #                                 nn.Linear(1200, 600),
        #                                 nn.ReLU(),
        #                                 nn.Linear(600, 2))
    
    def forward(self, cids, smiles):
        embeddings = self.embedder.embed(smiles, cids)
        # output = self.classifier(embeddings)
        output = embeddings
        for i in range(len(self.hidden_dim)):
            output = self.classifier[i](output)
            output = F.relu(output)
        output = self.classifier[-1](output)
        
        return output
    
# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class SynthEGNN(nn.Module):
    def __init__(self, num_layers=3, dropout=0.0, time_emb_dim=10, no_edge_index=False, atom_feature_dim=13) -> None:
        super().__init__()
        self.atom_feature_dim=atom_feature_dim
        self.num_layers = num_layers
        self.time_emb_dim = time_emb_dim
        self.no_edge_index = no_edge_index
        self.input_dim = self.atom_feature_dim + self.time_emb_dim
        self.time_embedder = nn.Sequential(SinusoidalPosEmb(self.time_emb_dim),
                                           nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                                           nn.GELU(),
                                           nn.Linear(self.time_emb_dim * 4, self.time_emb_dim))
        self.egnn = nn.ModuleList()
        for i in range(self.num_layers):
            self.egnn.append(EGNN_Sparse(feats_dim=self.input_dim, dropout=dropout))
        self.dense = nn.Sequential(nn.Linear(self.input_dim, 2 * self.input_dim),
                                   nn.ReLU(),
                                   nn.Linear(2 * self.input_dim, 2))
    
    def forward(self, ligand_pos, ligand_v, edge_index, t, edge_attr=None, batch=None):
        time_emb = self.time_embedder(t)
        if batch is None:
            batch = torch.zeros(len(ligand_pos), dtype=torch.int32)
        x = torch.cat([ligand_pos, ligand_v, time_emb[batch]], dim=1)
        if self.no_edge_index:
            edge_index = torch_cluster.radius_graph(ligand_pos, r=1.6, batch=batch)
        for i in range(self.num_layers):
            x = self.egnn[i](x=x, edge_index=edge_index, batch=batch, edge_attr=edge_attr)
        # print('x.shape:', x.shape)
        ligand_pos = x[:, :3]
        ligand_v = x[:, 3:]
        # print('ligand_pos, ligand_v: ', ligand_pos.shape, ligand_v.shape)
        if batch is None: 
            out = ligand_v.mean(dim=0, keepdims=True)
        else: 
            out = scatter_mean(ligand_v, batch, dim=0)
        out = self.dense(out)
        
        return out

class SynthModel(nn.Module):
    def __init__(self):
        pass
    
    
    def forward(self, ligand_pos, ligand_v, edge_index, t, edge_attr=None, batch=None):
        pass

if __name__ == '__main__':
    model = SinusoidalPosEmb(dim=10)
    x = torch.tensor([1, 2, 3])
    y = model(x)
    print(y)
    print(y.shape)
    