# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import copy
import os
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import util as utils
from dataset import *
from logger import Logger
from MeLU import *
import argparse
import torch
import copy
import random
from util import *
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.sparse as sp
from layers import *

from sklearn.metrics import average_precision_score, ndcg_score
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser([],description='Fast Context Adaptation via Meta-Learning (CAVIA),'
                                                 'Clasification experiments.')

    parser.add_argument('--seed', type=int, default=53)
    parser.add_argument('--task', type=str, default='multi', help='problem setting: sine or celeba')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=32, help='number of tasks in each batch per meta-update')#32

    parser.add_argument('--lr_inner', type=float, default=0.01, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='outer-loop learning rate (used with Adam optimiser)')
    parser.add_argument('--lr_tail', type=float, default=1e-6, help='outer-loop learning rate (used with Adam optimiser)')#1e-3 1e-4是3的结果
    parser.add_argument("--lamda", type=float, default=0.0001, help='l2 parameter')
    #parser.add_argument('--lr_meta_decay', type=float, default=0.9, help='decay factor for meta learning rate')

    parser.add_argument('--num_grad_steps_inner', type=int, default=5, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=1, help='number of gradient updates at test time (for evaluation)')

    parser.add_argument('--first_order', action='store_true', default=False, help='run first order approximation of CAVIA')

    parser.add_argument('--data_root', type=str, default="./data/", help='path to data root')

    parser.add_argument('--dataset', type=str, default="movielens", help='path to data root')#yelp, dbook
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
    parser.add_argument('--test', action='store_true', default=False, help='num of workers to use')
    
    parser.add_argument("--eta", type=float, default=1, help='adversarial constraint')
    parser.add_argument("--mu", type=float, default=0.001, help='missing info constraint')
    parser.add_argument("--hidden", type=int, default=8, help='hidden layer dimension')#32
    parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
    parser.add_argument("--ablation", type=int, default=0, help='ablation mode')
    parser.add_argument("--g_sigma", type=float, default=1, help='G deviation')

    parser.add_argument('--embedding_dim', type=int, default=32, help='num of workers to use')#32
    parser.add_argument('--first_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--second_fc_hidden_dim', type=int, default=64, help='num of workers to use')
    parser.add_argument('--num_epoch', type=int, default=500, help='num of workers to use')
    #movielens
    parser.add_argument('--num_genre', type=int, default=26, help='num of workers to use')
    parser.add_argument('--num_director', type=int, default=2187, help='num of workers to use')
    parser.add_argument('--num_actor', type=int, default=8031, help='num of workers to use')
    parser.add_argument('--num_rate', type=int, default=7, help='num of workers to use')
    parser.add_argument('--num_gender', type=int, default=3, help='num of workers to use')
    parser.add_argument('--num_age', type=int, default=8, help='num of workers to use')
    parser.add_argument('--num_occupation', type=int, default=22, help='num of workers to use')
    parser.add_argument('--num_zipcode', type=int, default=3403, help='num of workers to use')
    #yelp
    parser.add_argument('--num_stars', type=int, default=10, help='num of workers to use')
    parser.add_argument('--num_postalcode', type=int, default=7130, help='num of workers to use')
    parser.add_argument('--num_fans', type=int, default=422, help='num of workers to use')#422
    parser.add_argument('--num_avgrating', type=int, default=369, help='num of workers to use')
    #dbook
    parser.add_argument('--num_location', type=int, default=4550, help='num of workers to use')
    parser.add_argument('--num_publisher', type=int, default=17000, help='num of workers to use')
    #lastfm num_lastfm_user
    parser.add_argument('--num_lastfm_user', type=int, default=100000, help='num of workers to use')
    parser.add_argument('--num_lastfm_item', type=int, default=100000, help='num of workers to use')

    parser.add_argument('--rerun', action='store_true', default=False,
                        help='Re-run experiment (will override previously saved results)')

    args = parser.parse_args()
    # use the GPU if available
    #args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print('Running on device: {}'.format(args.device))
    return args



class TransGCN(nn.Module):
    def __init__(self, nfeat, nhid, g_sigma,device, ver, ablation=0):
        super(TransGCN, self).__init__()

        self.device = device
        self.ablation = ablation

        self.n_fold = 10
        if ver == 1:
            self.r = Relation(nfeat, ablation)
        else:
            self.r = Relationv2(nfeat,nhid, ablation)
        self.g = Generator(nfeat, g_sigma, ablation)
        self.gc = GraphConv(nfeat, nhid)
      

    def forward(self, user_emb, item_emb, user_item_adj, head, test=False):
        mean = F.normalize(user_item_adj, p=1, dim=1)
        neighbor = torch.mm(mean.to(item_emb.device), item_emb)
        output = self.r(user_emb, neighbor, if_test=test)
        if head or self.ablation == 2:
            norm = F.normalize(user_item_adj, p=1, dim=1)
            h_k = self.gc(item_emb, norm)
        else:
            if self.ablation == 1:
                h_s = self.g(output)
            else:
                h_s = output
            # h_s = torch.randn(output.shape[0], output.shape[1]).to(output.device)
            h_k = self.gc(item_emb, user_item_adj)
            h_s = torch.mm(h_s, self.gc.weight)
            h_k = h_k + h_s

            num_neighbor = torch.sum(user_item_adj, dim=1, keepdim=True)
            h_k = h_k / (num_neighbor+1)

        return h_k, output 



def get_movielens_item_idx(item_emb_length, item_dict):
    rate_idx = []
    genre_idx = []
    director_idx = []
    actor_idx = []
    for i in range(item_emb_length):
        item_id = i + 1
        if item_id in item_dict:
            a, b, c, d = item_dict[item_id][0][0], item_dict[item_id][0][1:26], item_dict[item_id][0][26:2212], item_dict[item_id][0][2212:10242]
            rate_idx.append(a+1)
            genre_idx.append(torch.cat((torch.tensor([0]), b), 0).tolist())
            director_idx.append(torch.cat((torch.tensor([0]), c), 0).tolist())
            actor_idx.append(torch.cat((torch.tensor([0]), d), 0).tolist())
        else:
            rate_idx.append(0)
            genre_idx.append(torch.cat((torch.tensor([1]), torch.zeros_like(b)), 0))
            director_idx.append(torch.cat((torch.tensor([1]), torch.zeros_like(c)), 0))
            actor_idx.append(torch.cat((torch.tensor([1]), torch.zeros_like(d)), 0))
    rate_idx = torch.tensor(rate_idx)
    genre_idx = torch.tensor(genre_idx)
    director_idx = torch.tensor(director_idx)
    actor_idx = torch.tensor(actor_idx)
    return rate_idx, genre_idx, director_idx, actor_idx

def get_movielens_user_idx(user_emb_length, user_dict):
        gender_idx = []
        age_idx = []
        occupation_idx = []
        area_idx = []
        for i in range(user_emb_length):
            user_id = i + 1
            if user_id in user_dict:

                a, b, c, d = tuple(user_dict[user_id][0])
                gender_idx.append(a+1)
                age_idx.append(b+1)
                occupation_idx.append(c+1)
                area_idx.append(d+1)
            else:
                gender_idx.append(0)
                age_idx.append(0)
                occupation_idx.append(0)
                area_idx.append(0)
        return torch.tensor(gender_idx), torch.tensor(age_idx), torch.tensor(occupation_idx), torch.tensor(area_idx)

def get_dbook_item_idx(item_emb_length, item_dict):
    pub_idx = []
    for i in range(item_emb_length):
        item_id = i + 1
        if item_id in item_dict:
            a = item_dict[item_id][0][0]
            pub_idx.append(a+1)
        else:
            pub_idx.append(0)
    return torch.tensor(pub_idx)

def get_dbook_user_idx(user_emb_length, user_dict):
    location_idx = []
    for i in range(user_emb_length):
        user_id = i + 1
        if user_id in user_dict:
            a = user_dict[user_id][0]
            location_idx.append(a+1)
        else:
            location_idx.append(0)
    return torch.tensor(location_idx)


# latent relation GCN
class TailGNN_movielens(nn.Module):
    def __init__(self, args, layer_num, nclass, params, device, ver, idx):
        super(TailGNN_movielens, self).__init__()

        self.nhid = params.hidden
        self.layer_num = layer_num
        self.dropout = params.dropout
        self.nclass = nclass
        

        self.rel_user = TransGCN(128, self.nclass, g_sigma=params.g_sigma, device=device, \
                            ver=ver, ablation=params.ablation).to(device)
        self.rel_item = TransGCN(128, self.nclass, g_sigma=params.g_sigma, device=device, \
                            ver=ver, ablation=params.ablation).to(device)
        #user
        self.embedding_gender = torch.nn.Embedding(
            num_embeddings=args.num_gender,
            embedding_dim=args.embedding_dim
        ).to(device)

        self.embedding_age = torch.nn.Embedding(
            num_embeddings=args.num_age,
            embedding_dim=args.embedding_dim
        ).to(device)

        self.embedding_occupation = torch.nn.Embedding(
            num_embeddings=args.num_occupation,
            embedding_dim=args.embedding_dim
        ).to(device)

        self.embedding_area = torch.nn.Embedding(
            num_embeddings=args.num_zipcode,
            embedding_dim=args.embedding_dim
        ).to(device)
        #item
        self.embedding_rate = torch.nn.Embedding(
            num_embeddings=args.num_rate, 
            embedding_dim=args.embedding_dim
        ).to(device)
        
        self.embedding_genre = torch.nn.Linear(
            in_features=args.num_genre,
            out_features=args.embedding_dim,
            bias=False
        ).to(device)
        
        self.embedding_director = torch.nn.Linear(
            in_features=args.num_director,
            out_features=args.embedding_dim,
            bias=False
        ).to(device)
        
        self.embedding_actor = torch.nn.Linear(
            in_features=args.num_actor,
            out_features=args.embedding_dim,
            bias=False
        ).to(device)
        self.gender_idx, self.age_idx, self.occupation_idx, self.area_idx, self.rate_idx, self.genre_idx, self.director_idx, self.actor_idx = idx

    
    def get_user_embedding(self, gender_idx, age_idx, occupation_idx, area_idx):
        
        gender_embed =  self.embedding_gender(gender_idx.to(device))
        age_embed = self.embedding_age(age_idx.to(device))
        occupation_embed = self.embedding_occupation(occupation_idx.to(device))
        aread_embed = self.embedding_area(area_idx.to(device))
        return torch.cat((gender_embed, age_embed, occupation_embed, aread_embed), 1).to(device)

    def get_item_embedding(self, rate_idx, genre_idx, director_idx, actor_idx):
    
        rate_embed = self.embedding_rate(rate_idx.to(device))
        genre_embed = self.embedding_genre(genre_idx.to(device).float()) / torch.sum(genre_idx.to(device).float(), 1).view(-1, 1)
        director_embed = self.embedding_director(director_idx.to(device).float()) / torch.sum(director_idx.to(device).float(), 1).view(-1, 1)
        actor_embed = self.embedding_actor(actor_idx.to(device).float()) / torch.sum(actor_idx.to(device).float(), 1).view(-1, 1)
        
        return torch.cat((rate_embed, genre_embed, director_embed, actor_embed), 1).to(device)
    
    def forward(self, user_emb_length, item_emb_length, user_item_adj, head, user_dict, item_dict, test= False):
        user_emb = self.get_user_embedding(self.gender_idx, self.age_idx, self.occupation_idx, self.area_idx)
        item_emb = self.get_item_embedding(self.rate_idx, self.genre_idx, self.director_idx, self.actor_idx)

        x1, out1 = self.rel_user(user_emb, item_emb, user_item_adj, head, test=test)
        x2, out2 = self.rel_item(item_emb, user_emb, user_item_adj.T, head, test=test)
        x = torch.cat((x1, x2), 0)
        out = torch.cat((out1, out2), 0)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x_user, x_item = torch.split(x,[user_emb_length, item_emb_length],dim=0)
        return x_user, x_item, F.log_softmax(x, dim=1), [out]


# latent relation GCN
class TailGNN_dbook(nn.Module):
    def __init__(self, args, layer_num, nclass, params, device, ver, idx):
        super(TailGNN_dbook, self).__init__()

        self.nhid = params.hidden
        self.layer_num = layer_num
        self.dropout = params.dropout
        self.nclass = nclass
        

        self.rel_user = TransGCN(64, self.nclass, g_sigma=params.g_sigma, device=device, \
                            ver=ver, ablation=params.ablation).to(device)
        self.rel_item = TransGCN(64, self.nclass, g_sigma=params.g_sigma, device=device, \
                            ver=ver, ablation=params.ablation).to(device)
        #user
        self.embedding_location = torch.nn.Embedding(
            num_embeddings=args.num_location,
            embedding_dim=args.embedding_dim * 2
        ).to(device)
        #item
        self.embedding_publisher = torch.nn.Embedding(
            num_embeddings=args.num_publisher, 
            embedding_dim=args.embedding_dim * 2
        ).to(device)
        self.pub_idx, self.location_idx = idx

    
    def get_user_embedding(self, loc_idx):
        loc_embed =  self.embedding_location(loc_idx.to(device))
        return loc_embed

    def get_item_embedding(self, pub_idx):
    
        pub_embed = self.embedding_publisher(pub_idx.to(device))
        
        
        return pub_embed
    
    def forward(self, user_emb_length, item_emb_length, user_item_adj, head, user_dict, item_dict, test= False):

        user_emb = self.get_user_embedding(self.location_idx)
        item_emb = self.get_item_embedding(self.pub_idx)

        x1, out1 = self.rel_user(user_emb, item_emb, user_item_adj, head, test=test)
        x2, out2 = self.rel_item(item_emb, user_emb, user_item_adj.T, head, test=test)
        x = torch.cat((x1, x2), 0)
        out = torch.cat((out1, out2), 0)
        x = F.elu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x_user, x_item = torch.split(x,[user_emb_length, item_emb_length],dim=0)
        return x_user, x_item, F.log_softmax(x, dim=1), [out]

def normalize_output(out_feat, idx):
    sum_m = 0
    for m in out_feat:
        sum_m += torch.mean(torch.norm(m[idx], dim=1))

    return sum_m 

 
def train_embed(embed_model, disc, optimizer, u_id, spt_pos, qry_pos, all_neg, criterion, args, user_item_head_adj, user_item_tail_adj, user_emb_length, item_emb_length, user_dict, item_dict):
    embed_model.train()
    optimizer.zero_grad()

    embed_h_user, embed_h_item, output_h, support_h  = embed_model(user_emb_length, item_emb_length, user_item_head_adj, True, user_dict, item_dict)
    embed_t_user, embed_t_item, output_t, support_t  = embed_model(user_emb_length, item_emb_length, user_item_tail_adj, False, user_dict, item_dict)
    x_pos = np.hstack((spt_pos, qry_pos))

    # weight regularizer
    m_h = normalize_output(support_h, x_pos + user_emb_length - 1)
    prob_t = disc(embed_t_item)
    t_labels = torch.full((len(x_pos), 1), 0.0, device=device)

    errorG = criterion(prob_t[x_pos-1], t_labels)
    L_d = errorG
    L_all = args.mu * m_h- (args.eta * L_d)

    L_all.backward()
    optimizer.step()
    return L_all, None, L_d

def train_disc(disc, optimizer_D,  embed_model, criterion, u_id, spt_pos, qry_pos, user_item_head_adj, user_item_tail_adj, user_emb_length, item_emb_length, user_dict, item_dict):
    disc.train()
    optimizer_D.zero_grad()

    embed_h_user, embed_h_item, _, _ = embed_model(user_emb_length, item_emb_length, user_item_head_adj, True, user_dict, item_dict)
    embed_t_user, embed_t_item, _, _ = embed_model(user_emb_length, item_emb_length, user_item_tail_adj, False, user_dict, item_dict)
    
    prob_h = disc(embed_h_item)
    prob_t = disc(embed_t_item)

    x_pos = np.hstack((spt_pos, qry_pos))

    t_labels = torch.full((len(x_pos-1), 1), 0.0, device=device)
    h_labels = torch.full((len(x_pos-1), 1), 1.0, device=device)

    errorD = criterion(prob_h[x_pos-1], h_labels)
    errorG = criterion(prob_t[x_pos-1], t_labels)
    L_d = (errorD + errorG)/2 

    L_d.backward()
    optimizer_D.step()
    return L_d

def gen_neg(u_id, spt_pos, qry_pos, item_emb_length):
    neg_nodes = []
    for i, node in enumerate(u_id):
        neighbor = []
        length = len(spt_pos[i]) + len(qry_pos[i])
        while length > 0:
            neg = random.randint(1, item_emb_length)
            if neg not in spt_pos[i] and neg not in qry_pos[i]:
                neighbor.append(neg)
                length -= 1
        neg_nodes.append(np.asarray(neighbor).reshape(-1))  
    return np.asarray(neg_nodes)

def gen_pos(adj, u_id, x_support):
    pos_nodes = [int(s[0]) for s in x_support]
    train_pos_nodes = np.asarray(pos_nodes).reshape(-1)
    return train_pos_nodes

def get_user(i_id, adj, user_emb_length):
    pos_user_adj = adj[i_id[0] + user_emb_length -1, :]
    pos_user_adj = pos_user_adj.cpu().numpy()

    pos_user = []
    for i, p in enumerate(pos_user_adj):
        if p!=0:
            pos_user.append(i)
    neg_user = []
    length = len(pos_user)
    while length > 0:
        neg = random.randint(0, user_emb_length-1)
        if neg not in pos_user:
            neg_user.append(neg)
            length -= 1
    return np.asarray(pos_user), np.asarray(neg_user)


def run(args, num_workers=1, log_interval=100, verbose=True, save_path=None):
    start_time = time.time()
    utils.set_seed(args.seed)


    # ---------------------------------------------------------
    # -------------------- training ---------------------------
    user_item_test_adj, user_item_head_adj, user_item_tail_adj, user_dict, item_dict, user_emb_length, item_emb_length = DatasetLoad(args).load()

    user_item_test_adj = torch.FloatTensor(user_item_test_adj.todense()).to(device)
    user_item_head_adj = torch.FloatTensor(user_item_head_adj.todense()).to(device)
    user_item_tail_adj = torch.FloatTensor(user_item_tail_adj.todense()).to(device)

    # initialise model
    n_class = 128
    if args.dataset == "movielens":
        gender_idx, age_idx, occupation_idx, area_idx = get_movielens_user_idx(user_emb_length, user_dict)
        rate_idx, genre_idx, director_idx, actor_idx = get_movielens_item_idx(item_emb_length, item_dict)
        
        embed_model = TailGNN_movielens(
                args,
                layer_num = 2,
                nclass=n_class,
                params=args,
                device=device,
                ver=2,
                idx = (gender_idx, age_idx, occupation_idx, area_idx, rate_idx, genre_idx, director_idx, actor_idx))
    elif args.dataset == "dbook":
        location_idx = get_dbook_user_idx(user_emb_length, user_dict)
        pub_idx = get_dbook_item_idx(item_emb_length, item_dict)
        
        embed_model = TailGNN_dbook(
                args,
                layer_num = 2,
                nclass=n_class,
                params=args,
                device=device,
                ver=2,
                idx = (pub_idx, location_idx))
        
    optimizer_tailgnn = torch.optim.Adam(embed_model.parameters(),
                        lr=args.lr_tail, weight_decay=args.lamda)

    
    disc = Discriminator(n_class).to(device)
    optimizer_D = torch.optim.Adam(disc.parameters(),
                        lr=args.lr_tail, weight_decay=args.lamda)
    criterion = nn.BCELoss()
    if args.dataset == "movielens":
        model = user_preference_estimator_movielens(args).to(device)
    elif args.dataset == "dbook":
        model = user_preference_estimator_dbook(args).to(device)

    model.train()
    print(sum([param.nelement() for param in model.parameters()]))
    para = list(model.parameters())
    para.extend(list(embed_model.parameters()))
    meta_optimiser = torch.optim.Adam(para, args.lr_meta)

    # initialise logger
    logger = Logger()
    logger.args = args
    # initialise the starting point for the meta gradient (it's faster to copy this than to create new object)
    meta_grad_init = [0 for _ in range(len(model.state_dict()) + len(embed_model.state_dict()))]
    dataloader_train = DataLoader(Data(args),
                                     batch_size=1,num_workers=args.num_workers)
    dataloader_test_newuser = DataLoader(Data(args,partition='test',test_way='new_user'),#old, new_user, new_item, new_item_user
                                batch_size=1,num_workers=args.num_workers)
    dataloader_test_newitem = DataLoader(Data(args,partition='test',test_way='new_item'),#old, new_user, new_item, new_item_user
                                batch_size=1,num_workers=args.num_workers)
    dataloader_test_newall = DataLoader(Data(args,partition='test',test_way='new_item_user'),#old, new_user, new_item, new_item_user
                                batch_size=1,num_workers=args.num_workers)
    dataloader_test_old = DataLoader(Data(args,partition='test',test_way='old'),#old, new_user, new_item, new_item_user
                                batch_size=1,num_workers=args.num_workers)    
    min_valid_mae = 10000
    for epoch in range(args.num_epoch):

        x_spt, y_spt, x_qry, y_qry, u_id, spt_pos, qry_pos = [],[],[],[],[],[],[]
        iter_counter = 0
        for step, batch in enumerate(dataloader_train):
            # if iter_counter >0:
            #     break
            if len(x_spt)<args.tasks_per_metaupdate:
                x_spt.append(batch[0][0].to(device))
                y_spt.append(batch[1][0].to(device))
                x_qry.append(batch[2][0].to(device))
                y_qry.append(batch[3][0].to(device))
                u_id.append(batch[4][0].data.numpy())
                spt_pos.append(np.array(batch[5]).reshape(-1))
                qry_pos.append(np.array(batch[6]).reshape(-1))
                if not len(x_spt)==args.tasks_per_metaupdate:
                    continue

            if len(x_spt) != args.tasks_per_metaupdate:
                continue

            u_id = np.array(u_id)

            all_neg = gen_neg(u_id, spt_pos, qry_pos, item_emb_length)
            u_id = torch.tensor(u_id).to(device)
            # initialise meta-gradient
            meta_grad = copy.deepcopy(meta_grad_init)
            loss_pre = []
            loss_after = []
            for i in range(args.tasks_per_metaupdate): 
                loss_d = train_disc(disc, optimizer_D, embed_model, criterion, u_id[i], spt_pos[i], qry_pos[i], user_item_head_adj, user_item_tail_adj, user_emb_length, item_emb_length, user_dict, item_dict)
                L_all, loss_link, L_d = train_embed(embed_model, disc, optimizer_tailgnn, u_id[i], spt_pos[i], qry_pos[i], all_neg[i], criterion, args, user_item_head_adj, user_item_tail_adj, user_emb_length, item_emb_length, user_dict, item_dict)

                x_item, x_user =  model(x_qry[i])
                # with torch.no_grad():
                user_emb, item_emb, _, _ = embed_model(user_emb_length, item_emb_length, user_item_head_adj, True, user_dict, item_dict)
                
                user_e = user_emb[np.array([u_id[i].data.cpu().numpy()-1 for _ in range(len(qry_pos[i]))])]
               
                item_e = item_emb[qry_pos[i]-1]
                
                user = torch.cat((x_user, user_e), 1)
                item = torch.cat((x_item, item_e), 1)
               
                x_qr = torch.cat((item, user), 1)

                x_q = model.final_part(x_qr)
                loss_pre.append(F.mse_loss(x_q, y_qry[i]).item())
                fast_parameters = model.final_part.parameters()
                for weight in model.final_part.parameters():
                    weight.fast = None
             
                user_emb, item_emb, _, _ = embed_model(user_emb_length, item_emb_length, user_item_head_adj, True, user_dict, item_dict)
                for k in range(args.num_grad_steps_inner):
                   
                    x_item, x_user =  model(x_spt[i])
                    
                    item_e = item_emb[spt_pos[i]-1]
                    user_e = user_emb[np.array([u_id[i].data.cpu().numpy()-1 for _ in range(len(spt_pos[i]))])]
                   
                    
                    user = torch.cat((x_user, user_e), 1)
                    item = torch.cat((x_item, item_e), 1)
                    x_s = torch.cat((item, user), 1)
                    x_s = model.final_part(x_s)
                    loss = F.mse_loss(x_s, y_spt[i])
                    grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                    fast_parameters = []
                    for k, weight in enumerate(model.final_part.parameters()):
                        if weight.fast is None:
                            weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                        else:
                            weight.fast = weight.fast - args.lr_inner * grad[k]  
                        fast_parameters.append(weight.fast)         

                x_item, x_user =  model(x_qry[i])
                
                user_emb, item_emb, _, _ = embed_model(user_emb_length, item_emb_length, user_item_head_adj, True, user_dict, item_dict)
                
                user_e = user_emb[np.array([u_id[i].data.cpu().numpy()-1 for _ in range(len(qry_pos[i]))])]
                item_e = item_emb[qry_pos[i]-1]
                user = torch.cat((x_user, user_e), 1)
                item = torch.cat((x_item, item_e), 1)
                x_qr = torch.cat((item, user), 1)
                x_q = model.final_part(x_qr)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.mse_loss(x_q, y_qry[i])
                loss_after.append(loss_q.item())

                a = list(embed_model.parameters())
                b = list(model.parameters())
                a.extend(b)

                task_grad_test = torch.autograd.grad(loss_q, a, allow_unused=True)
                
                for g in range(len(task_grad_test)):
                    if task_grad_test[g] is None:
                        continue
                    meta_grad[g] += task_grad_test[g].detach()
                    
            # -------------- meta update --------------
            
            meta_optimiser.zero_grad()

            # set gradients of parameters manually
            for c, param in enumerate(a):
                if task_grad_test[c] is None:
                    continue
                param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                param.grad.data.clamp_(-10, 10)

            # the meta-optimiser only operates on the shared parameters, not the context parameters
            meta_optimiser.step()
            
            x_spt, y_spt, x_qry, y_qry, u_id, spt_pos, qry_pos = [],[],[],[],[],[],[]
            
            loss_pre = np.array(loss_pre)
            loss_after = np.array(loss_after)
            logger.train_loss.append(np.mean(loss_pre))
            logger.valid_loss.append(np.mean(loss_after))
            logger.train_conf.append(1.96*np.std(loss_pre, ddof=0)/np.sqrt(len(loss_pre)))
            logger.valid_conf.append(1.96*np.std(loss_after, ddof=0)/np.sqrt(len(loss_after)))
            logger.test_loss.append(0)
            logger.test_conf.append(0)
    
            logger.print_info(epoch, iter_counter, start_time)
            start_time = time.time()
            
            iter_counter += 1
        
        dataloader_valid = DataLoader(Data(args, partition='valid'),
                                     batch_size=1, num_workers=args.num_workers)
        mae = evaluate_test(args.dataset, "valid", epoch, args, model, disc, embed_model, dataloader_valid, user_item_test_adj, user_emb_length, item_emb_length, user_dict, item_dict)
        if min_valid_mae > mae:
            min_valid_mae = mae
            _ = evaluate_test(args.dataset, "new_user", epoch, args, model, disc, embed_model, dataloader_test_newuser, user_item_test_adj, user_emb_length, item_emb_length, user_dict, item_dict)
            _ = evaluate_test(args.dataset, "new_item", epoch, args, model, disc, embed_model, dataloader_test_newitem, user_item_test_adj, user_emb_length, item_emb_length, user_dict, item_dict)
            _ = evaluate_test(args.dataset, "new_user_item", epoch, args, model, disc, embed_model, dataloader_test_newall, user_item_test_adj, user_emb_length, item_emb_length, user_dict, item_dict)
            _ = evaluate_test(args.dataset, "old", epoch, args, model, disc, embed_model, dataloader_test_old, user_item_test_adj, user_emb_length, item_emb_length, user_dict, item_dict)
            
    return logger, model


def evaluate_test(dataset, state, epoch, args, model, disc, tailgnn,  dataloader, user_item_test_adj, user_emb_length, item_emb_length, user_dict, item_dict):
    model.eval()
    disc.eval()
    tailgnn.eval()
    loss_all = []
    ndcg_all_1 = []
    ndcg_all_3 = []
    ndcg_all_11 = []
    ndcg_all_33 = []
    ap3_all = []
    pre3_all = []
    for c, batch in tqdm(enumerate(dataloader)):
        x_spt = batch[0][0].to(device)
        y_spt = batch[1][0].to(device)
        x_qry = batch[2][0].to(device)
        y_qry = batch[3][0].to(device)
        u_id = batch[4][0].data.numpy()
        spt_pos = np.array(batch[5]).reshape(-1)
        qry_pos = np.array(batch[6]).reshape(-1)
        u_id = np.array(u_id)
        u_id = torch.tensor(u_id).to(device)
        fast_parameters = model.final_part.parameters()
        for weight in model.final_part.parameters():
            weight.fast = None
        for k in range(args.num_grad_steps_inner):
            x_item, x_user =  model(x_spt)
            with torch.no_grad():
                user_emb, item_emb, _, _ = tailgnn(user_emb_length, item_emb_length, user_item_test_adj, False, user_dict, item_dict, True)
            
            user_e = user_emb[np.array([u_id.data.cpu().numpy()-1 for _ in range(len(spt_pos))])]
            
            item_e = item_emb[spt_pos-1]
            user = torch.cat((x_user, user_e), 1)
            item = torch.cat((x_item, item_e), 1)
            x_qr = torch.cat((item, user), 1)
            logits = model.final_part(x_qr)
            loss = F.mse_loss(logits, y_spt)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            for k, weight in enumerate(model.final_part.parameters()):
                if weight.fast is None:
                    weight.fast = weight - args.lr_inner * grad[k] #create weight.fast 
                else:
                    weight.fast = weight.fast - args.lr_inner * grad[k]  
                fast_parameters.append(weight.fast)
        x_item, x_user =  model(x_qry)
        with torch.no_grad():
            
            user_emb, item_emb, _, _ = tailgnn(user_emb_length, item_emb_length, user_item_test_adj, False, user_dict, item_dict, test=True)
            

        user_e = user_emb[np.array([u_id.data.cpu().numpy()-1 for _ in range(len(qry_pos))])]
        
        item_e = item_emb[qry_pos-1]
        user = torch.cat((x_user, user_e), 1)
        item = torch.cat((x_item, item_e), 1)
        x_qr = torch.cat((item, user), 1)
        x_qr = model.final_part(x_qr)

        x = x_qr.cpu().detach()
        y = y_qry.cpu().detach().numpy().reshape(-1)
        NDCG_1 = ndcg_score([y], [x.numpy().reshape(-1)], k=1)
        NDCG_3 = ndcg_score([y], [x.numpy().reshape(-1)], k=3)
        output_list, x = x.view(-1).sort(descending=True)
        NDCG_11 = nDCG(x, y, 1)
        NDCG_33 = nDCG(x, y, 3)
        ap_3 = AP(x, y, 3)
        pre_3 = precision(x, y, 3)
        ndcg_all_1.append(NDCG_1)
        ndcg_all_3.append(NDCG_3)
        ndcg_all_11.append(NDCG_11)
        ndcg_all_33.append(NDCG_33)
        ap3_all.append(ap_3)
        pre3_all.append(pre_3)
        loss_all.append(F.l1_loss(y_qry, x_qr).item())
    loss_all = np.array(loss_all)
    ndcg_all_1 = np.array(ndcg_all_1)
    ndcg_all_3 = np.array(ndcg_all_3)
    ndcg_all_11 = np.array(ndcg_all_11)
    ndcg_all_33 = np.array(ndcg_all_33)
    ap3_all = np.array(ap3_all)
    pre3_all = np.array(pre3_all)
    with open("results/exp12.txt", 'a') as f:
        f.write("Epoch:{}\n".format(epoch))
        f.write("State:  "+state+"\n")
        f.write('{}+/-{}\n'.format(np.mean(loss_all), 1.96*np.std(loss_all,0)/np.sqrt(len(loss_all))))
        f.write("NDCG@1:{}\n".format(np.mean(ndcg_all_1)))
        f.write("NDCG@3:{}\n".format(np.mean(ndcg_all_3)))
        f.write("NDCG@11:{}\n".format(np.mean(ndcg_all_11)))
        f.write("NDCG@33:{}\n".format(np.mean(ndcg_all_33)))
        f.write("MAP@3:{}\n".format(np.mean(ap3_all)))
        f.write("P@3:{}\n".format(np.mean(pre3_all)))
    print('{}+/-{}'.format(np.mean(loss_all), 1.96*np.std(loss_all,0)/np.sqrt(len(loss_all))))
    print("NDCG@1:{}".format(np.mean(ndcg_all_1)))
    print("NDCG@3:{}".format(np.mean(ndcg_all_3)))
    return np.mean(loss_all)

            

if __name__ == '__main__':
    args = parse_args()
    run(args, num_workers=1, log_interval=100, verbose=True, save_path=None)
