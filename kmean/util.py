import math
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def data2cluster(x, c):
    if isinstance(x, list):
        x = torch.from_numpy(np.array(x))
    elif isinstance(x, np.ndarray):    
        x = torch.from_numpy(x) 
    x = x.to(c.device)
    return data2cluster_euc(x, c)
    
def data2cluster_euc(x, c):
    temp = x[:, None, :] - c[None, :, :]
    temp = temp ** 2
    temp = temp.sum(-1)
    value, idx = temp.min(dim=1)
    return idx, value.sum()
        
        
class ScheduledOptim:
    
    def __init__(self, optimizer, init_lr, n_warmup_steps=0, 
                 n_current_steps=0, final_steps=0):
        self._optimizer = optimizer
        self.init_lr = init_lr
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = n_current_steps
        self.final_steps = final_steps
        self.lr = 0

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad(set_to_none=True)
        
    def _update_learning_rate(self):        
        if  self.n_warmup_steps == 0: return
        
        self.n_current_steps += 1
        if self.n_current_steps < self.n_warmup_steps:
            lr_mult = float(self.n_current_steps) / float(max(1, self.n_warmup_steps))
        else:
            progress = float(self.n_current_steps - self.n_warmup_steps) / float(max(1, self.final_steps - self.n_warmup_steps))
            lr_mult = max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        self.lr = self.init_lr * lr_mult      
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = self.lr #param_group['lr'] * lr_mult
            


