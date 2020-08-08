#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:01:53 2020

@author: dengbo
"""
import torch
import matplotlib.pyplot as plt
from torch.optim import *
import torch.nn as nn
import math

class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.fc = nn.Linear(1,10)
    def forward(self,x):
        return self.fc(x)

class args:
    milestones=[20,70,80]
    warm_up_epochs=10
    epochs=40
    
import numpy as np 
lr_list = []
model = net()
LR = 0.01
optimizer = Adam(model.parameters(),lr = LR)
######warm_up_cosine_lr#######
warm_up_with_cosine_lr = lambda epoch: epoch / args.warm_up_epochs if epoch <= args.warm_up_epochs else 0.5 * ( math.cos((epoch - args.warm_up_epochs) /(args.epochs - args.warm_up_epochs) * math.pi) + 1)
scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)
#scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=40)

for epoch in range(100):
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
plt.plot(range(100),lr_list,color = 'r')

