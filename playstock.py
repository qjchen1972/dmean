# -*- coding:utf-8 -*-
import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import pandas as pd
import os
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import random
import argparse

from kmean.model import Kmodel, KmodelConfig 
from kmean.trainer import KmeanTrainer, KmeanConfig
from kmean.util import set_seed, data2cluster

# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,        
        #filename= 'dat.log',
)
logger = logging.getLogger(__name__)

tr_path = 'data/tr.npy'
te_path = 'data/te.npy' 



class StockDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        #数据格式是 n*1*6
        self.data = np.load(path,allow_pickle=True)
        print(self.data.shape)
      
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        # 读出来的数据是strt类型, 需要转换
        return torch.from_numpy(self.data[idx,0,2:].astype(np.float32))
    
    def getinfo(self, idx):
        return self.data[idx,0,:2]    

def test_dataset():
    train_dataset = StockDataset(tr_path)
    print(len(train_dataset))
    print(train_dataset.getinfo(0))
    print(train_dataset[0])
    logger.info('ok')


def train(dl_mode=True, last_model=None):
    
    train_dataset = StockDataset(tr_path)
    mconf = KmodelConfig(dataset=train_dataset,
                         ncluster=2048,
                         ndim=4,
                         dl_mode = dl_mode,) 
                         
    model = Kmodel(mconf)
    
    kconf = KmeanConfig(max_epochs=30,
                        warmup_tokens=800,
                        dataset=train_dataset, 
                        batch_size=512, 
                        dead_cluster=10,                    
                        last_model=last_model,)


    model = Kmodel(mconf)
    trainer = KmeanTrainer(kconf, model)
    trainer.train()
    
    x = [[0.032, 0.049, -0.025, 0.015]]
    model.eval()
    idx, loss = model(x)
    print(idx, loss, model.cluster[idx])
    

def test(path='model/best.pt'):

    mconf = KmodelConfig(ncluster=2048,
                         ndim=4,) 
                         
    model = Kmodel(mconf)                      
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)

    model.eval()
    test_dataset = StockDataset(te_path)
    x = [test_dataset[i] for i in range(10000)]
    x = torch.stack(x, dim=0)
    idx, loss = model(x)
    print(loss)


def checkCluster(path='model/best.pt', cluster_num=2048):
    
    mconf = KmodelConfig(ncluster=2048,
                         ndim=4,) 
                         
    model = Kmodel(mconf)                      
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint)
    model.eval()
    
    train_dataset = StockDataset(tr_path)
    loader = DataLoader(train_dataset, shuffle=True, pin_memory=True,
                        batch_size=512, num_workers=0)
                        
    new_cluster_num = torch.zeros(cluster_num)                     
    for it, x in enumerate(loader):
        idx, loss = model(x)
        for k in range(cluster_num):
            t = x[idx==k]
            new_cluster_num[k] += t.shape[0]
    
    #torch.set_printoptions(profile='full')
    print(new_cluster_num, sum(new_cluster_num!=0))
    
    a, b = new_cluster_num.sort(descending=True)
    print(a, b)
    
    print(model.cluster[new_cluster_num!=0])
    return model.cluster[new_cluster_num!=0]

if __name__ == '__main__':    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', type=int, default=0, metavar='mode', help='input run mode (default: 1)')    
    args = parser.parse_args()
    
    set_seed(42)

    if args.m == 0:
        test_dataset()
    elif args.m == 1:
        #train(dl_mode=True, last_model='cluser_27.pt')
        train(dl_mode=False, last_model=None)                
    elif args.m == 2:
        test(path='model/best.pt')
    else:
         checkCluster(cluster_num=2048)    