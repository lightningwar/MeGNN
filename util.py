import hashlib
import os
import pickle
import random

import numpy as np
import torch

import math

def AP(ranked_list, ground_truth, topn):
    hits, sum_precs = 0, 0.0
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t=t[:topn]
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            hits += 1
            sum_precs += hits / (i+1.0)
            t.remove(ground_truth[id])
    if hits > 0:
        return sum_precs / topn
    else:
        return 0.0

def RR(ranked_list, ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            return 1 / (i + 1.0)
    return 0

def precision(ranked_list,ground_truth,topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    t = t[:topn]
    hits = 0
    for i in range(topn):
        id = ranked_list[i]
        if ground_truth[id] in t:
            t.remove(ground_truth[id])
            hits += 1
    pre = hits/topn
    return pre


def nDCG(ranked_list, ground_truth, topn):
    dcg = 0
    idcg = IDCG(ground_truth, topn)
    # print(ranked_list)
    # input()
    for i in range(topn):
        id = ranked_list[i]
        dcg += ((2 ** ground_truth[id]) -1)/ math.log(i+2, 2)
    # print('dcg is ', dcg, " n is ", topn)
    # print('idcg is ', idcg, " n is ", topn)
    return dcg / idcg

def IDCG(ground_truth, topn):
    t = [a for a in ground_truth]
    t.sort(reverse=True)
    idcg = 0
    for i in range(topn):
        idcg += ((2**t[i]) - 1) / math.log(i+2, 2)
    return idcg

def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    if (seed is not None) and cudnn:
        torch.backends.cudnn.deterministic = True


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_path_from_args(args):
    """ Returns a unique hash for an argparse object. """
    args_str = str(args)
    path = hashlib.md5(args_str.encode()).hexdigest()
    return path


def get_base_path():
    p = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(p):
        return p
    raise RuntimeError('I dont know where I am; please specify a path for saving results.')
