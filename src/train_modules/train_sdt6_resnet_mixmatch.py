# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:58:01 2019

@author: Austin Hsu
"""

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import random

use_cuda = torch.cuda.is_available()

def train_resnet_4loss_mixmatch(input_t, target_Var, decoders, dec_opts, 
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] 

    input_time_step = input_t.size()[3]

    onDecOpt.zero_grad()
    
    onLoss  = 0

    window_size = 2*k+1

    nn_softmax = nn.Softmax(dim=1)
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            input_Var = Variable(torch.stack([ input_t[0, :, :, BATCH_SIZE*step+i-k:BATCH_SIZE*step+i-k+window_size]\
                           for i in range(BATCH_SIZE)], dim=0))
            #input_Var shape: (10,3,522,19)
            onDecOut6 = onDec(input_Var)
            onDecOut1 = nn_softmax(onDecOut6[:, :2])
            onDecOut2 = nn_softmax(onDecOut6[:, 2:4])
            onDecOut3 = nn_softmax(onDecOut6[:, 4:])

            temp_t = torch.max(onDecOut2[:, 1], onDecOut3[:, 1]).view(-1,1)
            onDecOut4 = torch.cat((onDecOut1, temp_t), dim=1)
            #print(onDecOut4.shape)

            for i in range(BATCH_SIZE):
                onLoss += onLossFunc(onDecOut1[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, :2].contiguous().view(1, 2))
                onLoss += onLossFunc(onDecOut2[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 2:4].contiguous().view(1, 2))
                onLoss += onLossFunc(onDecOut3[i].view(1, 2), target_Var[:,BATCH_SIZE*step+i, 4:].contiguous().view(1, 2))
                target_T = torch.max(target_Var[:,BATCH_SIZE*step+i, 3], target_Var[:,BATCH_SIZE*step+i, 5])
                onLoss += onLossFunc(onDecOut4[i].view(1, 3), torch.cat((target_Var[:,BATCH_SIZE*step+i, :2].contiguous().view(1, 2), target_T.contiguous().view(1, 1)), 1))
                            
    onLoss.backward()
    onDecOpt.step()
    
    # why not update in for loop?
    
    return onLoss.item() / input_time_step

def Mixmatch(labeled_data, labeled_label,
             unlabeled_data,
             curr_model,
             sharpening_temp=2, augment_time=2, beta_dist_alpha=0.75):
    # labeled_data   shape: (10, 3, 522, 19)
    # labeled_label  shape: (10, 6)
    # unlabeled_data shape: (10, 3, 522, 19)
    
    curr_model = curr_model.eval() # avoid influencing gradient calculations
    
    aug_x = Augment_data(labeled_data)
    aug_u = []
    label_guess = None
    for k in range(augment_time):
        aug_u_k = Augment_data(unlabeled_data)
        aug_u.append(aug_u_k)
        if label_guess == None:
            label_guess = curr_model(aug_u_k)
        else:
            label_guess += curr_model(aug_u_k)
    label_guess /= k
    # label_guess shape: (10, 6)
    label = Sharpen(label_guess, sharpening_temp)
    
    stack_data  = torch.cat((aug_x, *aug_u), dim=0)
    
    shuffle = torch.randperm(stack_data.size(0))
    x_mix   = Mixup(aug_x, labeled_label, stack_data[shuffle[:aug_x.size(0)]])
    u_mix   = Mixup(torch.cat(aug_u, dim=0), [label for i in range(k)], stack_data[shuffle[aug_x.size(0):]])
    
    return x_mix, u_mix
    
def Augment_data(orig_data):
    
    # perform some audio data augmentation tips, or cutout etc
    aug_data = orig_data
    
    return aug_data

def Sharpen(dist, T):
    sharpen_dist = dist
    for i in range(sharpen_dist.size(0)):
        sharpen_dist[i] = ( sharpen_dist[i]**(1./T) )/torch.sum( sharpen_dist[i]**(1./T) )
    return sharpen_dist

def Mixup(data, label,
          unlabel_data):
    return 
    