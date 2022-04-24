import logging
from torch.nn.modules import Module
import torch
class Bay_Loss(Module):
    def __init__(self, use_background, device):
        super(Bay_Loss, self).__init__()
        self.device = device
        self.l = 0.1
        self.c = 64
        self.use_bg = use_background
    # pre_density:D^est, targets, problist:
    def forward(self, prob_list,ot_list, target_list, pre_density):
        loss = 0
        # enumerate(p)-> (1,p1),(2,p2),(3,p3)
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points,ot = None
                # pre_count是这一列的求和
                pre_count = torch.sum(pre_density[idx])
  
                # target = 0
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                
                N = len(prob) # 一张图有n个像素(或者n-1和bg)
                if self.use_bg:
                    
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    # 除了最后一个
                    target[:-1] = target_list[idx]
                else:
                    target = target_list[idx]
                # \sum density*prob
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector
                loss += self.l*torch.sum(torch.matmul(pre_density[idx].view((1, -1)),ot_list[idx])) 
            #loss = |total-pred|+lambda* density* [...k,0,...]
            # target = 0/1, precount 看来是对行求和了
            loss += torch.sum(torch.abs(target - pre_count)) 

        loss = loss / len(prob_list)
        logging.info(loss)
        return loss



