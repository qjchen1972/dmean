"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""
import math
import logging
from tqdm import tqdm
import numpy as np
import os
import torch
from torch.utils.data.dataloader import DataLoader
from kmean.util import ScheduledOptim

logger = logging.getLogger(__name__)

class KmeanConfig:
    max_epochs = 10
    warmup_tokens=800
    learning_rate = 1e-3
    batch_size = 64
    last_model = None
    model_path = 'model/'
    #grad_norm_clip = 1.0
    dead_cluster = 10
    
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)


class KmeanTrainer:

    def __init__(self, config, model):
        self.cfg = config
        self.model = model        
        #cpu模式更快
        self.device = 'cpu' #torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        self.best_loss = float('inf')
        
        if config.last_model is not None:
            self.load_model(os.path.join(self.cfg.model_path, config.last_model))
        else:
            self.start_epoch = 0  
        self.model = self.model.to(self.device)   
        #self.model = torch.nn.DataParallel(self.model).to(self.device)        
            
    def save_model(self, epoch, loss):
        torch.save({'epoch':epoch, 
                    'loss':loss, 
                    'state_dict':self.model.state_dict(),},
                    os.path.join(self.cfg.model_path, 'cluser_%d.pt' %(epoch + 1)),
                   )                    
    
    def load_model(self, path):
        state_dict = torch.load(path)        
        self.model.load_state_dict(state_dict['state_dict'])
        self.start_epoch = state_dict['epoch'] + 1
        self.best_loss = state_dict['loss']
        print(self.start_epoch, self.best_loss)
        
    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", os.path.join(self.cfg.model_path, 'best.pt'))
        torch.save(raw_model.state_dict(), os.path.join(self.cfg.model_path, 'best.pt')) 
    
    def train(self):
        
        model, cfg, device = self.model, self.cfg, self.device       
        raw_model = model.module if hasattr(model, "module") else model
        optimizer = raw_model.configure_optimizers(cfg)
        init_tokens = self.start_epoch * (len(cfg.dataset) // cfg.batch_size)
        final_tokens = self.cfg.max_epochs * (len(cfg.dataset) // cfg.batch_size)
        optim_schedule = ScheduledOptim(optimizer, 
                                        init_lr=cfg.learning_rate, 
                                        n_warmup_steps=cfg.warmup_tokens, 
                                        n_current_steps=init_tokens,
                                        final_steps=final_tokens)
        def run_epoch():
            
            loader = DataLoader(cfg.dataset, shuffle=True, pin_memory=True,
                            batch_size=cfg.batch_size, num_workers=0)
                            
            pbar = tqdm(enumerate(loader), total=len(loader))       
            total_loss = []
            for it, x in pbar:
                x = x.to(device)
                with torch.set_grad_enabled(optimizer is not None):
                    idx, loss = model(x)
                    total_loss.append(loss)
                if optimizer is not None:
                    model.zero_grad(set_to_none=True)
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_norm_clip)
                    optim_schedule.step_and_update_lr()                    
                    #optimizer.zero_grad()
                    #loss.backward()
                    #optimizer.step()
                    
                pbar.set_description(f"epoch {epoch} iter {it}: train loss {loss.item():.5f}. lr {optim_schedule.lr:e}")                
            if optimizer is None:
                model.update(cfg.dead_cluster)                    
            logger.info('train loss = %.3f' %(sum(total_loss))) 
            return sum(total_loss)
            
        for epoch in range(self.start_epoch, self.cfg.max_epochs):      
            loss = run_epoch()
            if loss < self.best_loss:
                self.best_loss = loss
                self.save_checkpoint()
            self.save_model(epoch, self.best_loss)
            
            
            