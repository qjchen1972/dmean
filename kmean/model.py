# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from kmean.util import data2cluster

class KmodelConfig:
    dataset = None
    dl_mode = True
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class Kmodel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        
        if self.cfg.dataset is None:
            self.cluster = nn.Parameter(torch.zeros(self.cfg.ncluster, self.cfg.ndim)) 
        else:
            self.cluster = nn.Parameter(self.getRndValue(self.cfg.ncluster))       
               
        if not self.cfg.dl_mode:
            self.new_cluster = torch.zeros(self.cluster.shape)
            self.new_cluster_num = torch.zeros(self.cfg.ncluster)
        
        #for mn, m in self.named_modules():
        #    for pn, p in m.named_parameters():
        #        print(pn, p)

    def update(self, dead_cluster):
        if self.cfg.dl_mode: return 0                 
        para = self.new_cluster / self.new_cluster_num[:, None]        
        nanix = self.new_cluster_num <= dead_cluster
        ndead = nanix.sum().item()            
        print('re-initialized %d dead clusters' % (ndead))
        if ndead > 0:            
            para[nanix] = self.getRndValue(ndead).to(self.cluster.device)
        self.cluster = nn.Parameter(para)  
        self.new_cluster = torch.zeros(self.cluster.shape)
        self.new_cluster_num = torch.zeros(self.cfg.ncluster)            
        return ndead    
          
    def getRndValue(self, num):
        loader = DataLoader(self.cfg.dataset, shuffle=True, pin_memory=True,
                            batch_size=num, num_workers=0)        
        return next(iter(loader))           
    
        
    def configure_optimizers(self, cfg):
        if not self.cfg.dl_mode: return None 
        #optimizer = torch.optim.SGD(params=self.parameters(), lr=1e-3)
        optimizer = torch.optim.AdamW(params=self.parameters(), 
                                      lr=cfg.learning_rate,
                                      weight_decay=0.0)
        return optimizer
        
        
    def forward(self, x):
        idx, loss = data2cluster(x, self.cluster)        
        if not self.cfg.dl_mode and self.training:            
            self.new_cluster = self.new_cluster.to(x.device)
            self.new_cluster_num = self.new_cluster_num.to(x.device)            
            for k in range(self.cfg.ncluster):
                t = x[idx==k]
                self.new_cluster_num[k] += t.shape[0]
                self.new_cluster[k] += t.sum(dim=0)
        return idx, loss    
       