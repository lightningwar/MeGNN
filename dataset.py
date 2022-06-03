import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import random
import  re
import datetime
import pandas as pd
import json
import scipy.sparse as sp
from tqdm import tqdm

class DatasetLoad(object):
    def __init__(self, args):
        self.dataset = args.dataset
        self.path = args.data_root + self.dataset + "/original"
        self.dataset_path = args.data_root + args.dataset
        # self.user_data, self.item_data, self.score_data, self.head_adj, self.tail_adj, self.tail_adj1, self.user_emb_length, self.item_emb_length = self.load()

    def split_nodes(self, adj, k=5):
        num_links = np.sum(adj, axis=1)
        head = np.where(num_links > k)[0]
        tail = np.where(num_links <= k)[0]
        return head, tail

    def link_dropout(self, adj, idx, k=30):
        tail_adj = adj.copy()
        num_links = np.random.randint(k, size=idx.shape[0])
        num_links += 1
        for i in range(idx.shape[0]):
            index = tail_adj[idx[i]].nonzero()[1]
            new_idx = np.random.choice(index, num_links[i], replace=False)
            tail_adj[idx[i]] = 0.0
            for j in new_idx:
                tail_adj[idx[i], j] = 1.0
        return tail_adj

    def load(self):

        if self.dataset == "movielens":
            score_data_path = "{}/ratings.dat".format(self.path)
            score_data = pd.read_csv(
            score_data_path, names=['user_id', 'movie_id', 'rating', 'timestamp'],
            sep="::", engine='python')
            user_id = list(score_data['user_id'])
            item_id = list(score_data['movie_id'])
            user_id = np.array(list(map(int, user_id)))
            user_emb_length = max(user_id)
            
            item_id = list(map(int, item_id))
            item_emb_length = max(item_id)
            temp_user_id = user_id - 1
            temp_item_id = np.array(item_id) - 1
            user_item_total_adj = sp.coo_matrix((np.ones(temp_user_id.shape[0]), (temp_user_id, temp_item_id)),
                                    shape=(user_emb_length, item_emb_length),
                                    dtype=np.float32)
            user_item_total_adj = user_item_total_adj.tolil()


            state = 'new_meta_training'
            with open("{}/{}.json".format(self.dataset_path, state), encoding="utf-8") as f:
                self.dataset_split = json.loads(f.read())
            user_list = []
            item_list = []
            for k, v in self.dataset_split.items():
                user = [int(k)] * len(v)
                item = v
                user_list.extend(user)
                item_list.extend(item)

            temp_item_id = np.array(item_list) - 1
            temp_user_id = np.array(user_list) - 1
            user_item_head_adj = sp.coo_matrix((np.ones(temp_user_id.shape[0]), (temp_user_id, temp_item_id)),
                                    shape=(user_emb_length, item_emb_length),
                                    dtype=np.float32)
            user_item_head_adj = user_item_head_adj.tolil()

            head_nodes, tail_nodes = self.split_nodes(user_item_head_adj, k=20)#movielens:50
            user_item_tail_adj = self.link_dropout(user_item_head_adj, head_nodes, k=20)#movielens:50

            user_dict = np.load(self.dataset_path  + "/user_feature.npy", allow_pickle=True).item()
            item_dict = np.load(self.dataset_path + "/movie_feature_homo.npy", allow_pickle=True).item()
    
        elif self.dataset == "dbook":
            score_data_path = "{}/user_book.dat".format(self.path)
            score_data = pd.read_csv(
            score_data_path, names=['user_id', 'book_id', 'rating'],
            sep="\t", engine='python')
            user_id = list(score_data['user_id'])
            item_id = list(score_data['book_id'])
            user_id = np.array(list(map(int, user_id)))
            user_emb_length = max(user_id)
            
            item_id = list(map(int, item_id))
            item_emb_length = max(item_id)
            temp_user_id = user_id - 1
            temp_item_id = np.array(item_id) - 1
            user_item_total_adj = sp.coo_matrix((np.ones(temp_user_id.shape[0]), (temp_user_id, temp_item_id)),
                                    shape=(user_emb_length, item_emb_length),
                                    dtype=np.float32)
            user_item_total_adj = user_item_total_adj.tolil()


            state = 'new_meta_training'
            with open("{}/{}.json".format(self.dataset_path, state), encoding="utf-8") as f:
                self.dataset_split = json.loads(f.read())
            user_list = []
            item_list = []
            for k, v in self.dataset_split.items():
                user = [int(k)] * len(v)
                item = v
                user_list.extend(user)
                item_list.extend(item)

            temp_item_id = np.array(item_list) - 1
            temp_user_id = np.array(user_list) - 1
            user_item_head_adj = sp.coo_matrix((np.ones(temp_user_id.shape[0]), (temp_user_id, temp_item_id)),
                                    shape=(user_emb_length, item_emb_length),
                                    dtype=np.float32)
            user_item_head_adj = user_item_head_adj.tolil()

            head_nodes, tail_nodes = self.split_nodes(user_item_head_adj, k=20)#movielens:50
            user_item_tail_adj = self.link_dropout(user_item_head_adj, head_nodes, k=20)#movielens:50
            user_dict = np.load(self.dataset_path  + "/user_feature.npy", allow_pickle=True).item()
            item_dict = np.load(self.dataset_path + "/item_feature_homo.npy", allow_pickle=True).item()

        return user_item_total_adj, user_item_head_adj, user_item_tail_adj, user_dict, item_dict, user_emb_length, item_emb_length

class Data(Dataset):
    def __init__(self, args, partition='train', test_way=None, path=None):
        super(Data, self).__init__()
        self.partition = partition
        self.dataset = args.dataset
        
        self.dataset_path = args.data_root + args.dataset
        dataset_path = self.dataset_path



        self.user_dict = np.load(self.dataset_path  + "/user_feature.npy", allow_pickle=True).item()
        if args.dataset == "movielens":
            self.item_dict = np.load(self.dataset_path + "/movie_feature_homo.npy", allow_pickle=True).item()
        elif args.dataset == "lastfm" or args.dataset == "dbook":
            self.item_dict = np.load(self.dataset_path + "/item_feature_homo.npy", allow_pickle=True).item()

        if partition == 'valid' or test_way == 'old':
            self.query_size = 4
        else:
            self.query_size = 10

        if partition == 'train':
            self.state = 'new_meta_training'
        elif partition == 'valid':
            self.state = 'valid'
        else:
            if test_way is not None:
                if test_way == 'old':
                    self.state = 'warm_up'
                elif test_way == 'new_user':
                    self.state = 'user_cold_testing'
                elif test_way == 'new_item':
                    self.state = 'item_cold_testing'
                else:
                    self.state = 'user_and_item_cold_testing'
        print(self.state)
        with open("{}/{}.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split = json.loads(f.read())
        with open("{}/{}_y.json".format(dataset_path, self.state), encoding="utf-8") as f:
            self.dataset_split_y = json.loads(f.read())            
        length = len(self.dataset_split.keys())
        self.final_index = []
        count = 0
        for _, user_id in tqdm(enumerate(list(self.dataset_split.keys()))):
            u_id = int(user_id)
            seen_movie_len = len(self.dataset_split[str(u_id)])
            if partition == 'valid' or test_way == 'old':
                if (seen_movie_len < 5 or seen_movie_len > 100):
                    continue
                else:
                    self.final_index.append(user_id)
            else:
                if (seen_movie_len < 13 or seen_movie_len > 100):
                    continue
                else:
                    self.final_index.append(user_id)
         
    def __getitem__(self, item):
        user_id = self.final_index[item]
        u_id = int(user_id)
        seen_movie_len = len(self.dataset_split[str(u_id)])
        indices = list(range(seen_movie_len))
        random.shuffle(indices)
        tmp_x = np.array(self.dataset_split[str(u_id)])
        tmp_y = np.array(self.dataset_split_y[str(u_id)])
        support_x_app = None
        support_pos = []
        for m_id in tmp_x[indices[:-self.query_size]]:
            m_id = int(m_id)
            support_pos.append(m_id)
            if self.dataset=="lastfm":
                item_d = torch.tensor([[m_id]])
                user_d = torch.tensor([[u_id]])
                tmp_x_converted = torch.cat((item_d, user_d), 1)
            else:
                tmp_x_converted = torch.cat((self.item_dict[m_id], self.user_dict[u_id]), 1)
            try:
                support_x_app = torch.cat((support_x_app, tmp_x_converted), 0)
            except:
                support_x_app = tmp_x_converted
        query_x_app = None
        query_pos = []
        for m_id in tmp_x[indices[-self.query_size:]]:
            m_id = int(m_id)
            query_pos.append(m_id)
            u_id = int(user_id)
            if self.dataset=="lastfm":
                item_d = torch.tensor([[m_id]])
                user_d = torch.tensor([[u_id]])
                tmp_x_converted = torch.cat((item_d, user_d), 1)
            else:
                tmp_x_converted = torch.cat((self.item_dict[m_id], self.user_dict[u_id]), 1)
            try:
                query_x_app = torch.cat((query_x_app, tmp_x_converted), 0)
            except:
                query_x_app = tmp_x_converted
        support_y_app = torch.FloatTensor(tmp_y[indices[:-self.query_size]])
        query_y_app = torch.FloatTensor(tmp_y[indices[-self.query_size:]])
        return support_x_app, support_y_app.view(-1,1), query_x_app, query_y_app.view(-1,1), u_id, np.array(support_pos), np.array(query_pos)
        
    def __len__(self):
        return len(self.final_index)