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
import math
import copy
import torch.nn.functional as F

from train_modules.audio_augment import transform_method

def train_resnet_4loss_DataAug(input_t, target_Var, decoders, dec_opts, device,
                               loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k):
    
    # input_t    shape: (1,3,522,data_length)
    # target_Var shape: (1,data_length,6)
    # unlabel_t  shape: (1,3,522,unlabel_data_length)
    
    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] 

    input_time_step = input_t.size()[3]

    window_size = 2*k+1
    
    totLoss = 0

    nn_softmax = nn.Softmax(dim=1)
    
    for step in range(k, input_time_step - k - BATCH_SIZE + 1, BATCH_SIZE):  
        onLoss  = 0 
        
        x_data  = torch.stack([ input_t[0, :, :, step+i-k:step+i-k+window_size] for i in range(BATCH_SIZE)], dim=0)
        x_label = target_Var[:, step:step+BATCH_SIZE]
        
        x_data, x_label = DataAug(x_data, x_label, device)
        
        x_data = Variable(x_data)
        x_label = Variable(x_label.unsqueeze(0))
        
        # === Labeled ===
        #input_Var shape: (10,3,522,19)
        onDecOut6   = onDec(x_data)
        onDecOut1   = nn_softmax(onDecOut6[:, :2])
        onDecOut2   = nn_softmax(onDecOut6[:, 2:4])
        onDecOut3   = nn_softmax(onDecOut6[:, 4:])
        
        temp_t = torch.max(onDecOut2[:, 1], onDecOut3[:, 1]).view(-1,1)
        onDecOut4 = torch.cat((onDecOut1, temp_t), dim=1)

        for i in range(BATCH_SIZE):
            
            # === Labeled ===
            onLoss += onLossFunc(onDecOut1[i].view(1, 2), x_label[:,i, :2].contiguous().view(1, 2))
            onLoss += onLossFunc(onDecOut2[i].view(1, 2), x_label[:,i, 2:4].contiguous().view(1, 2))
            onLoss += onLossFunc(onDecOut3[i].view(1, 2), x_label[:,i, 4:].contiguous().view(1, 2))
            target_T = torch.max(x_label[:,i, 3], x_label[:,i, 5])
            onLoss += onLossFunc(onDecOut4[i].view(1, 3), torch.cat((x_label[:,i, :2].contiguous().view(1, 2), 
                                 target_T.contiguous().view(1, 1)), 1))
            
        onDecOpt.zero_grad()
        onLoss.backward()
        onDecOpt.step()
        totLoss += onLoss.item()
    
    return totLoss / input_time_step

def DataAug(labeled_data, labeled_label, device, 
            transform_dict={'cutout'    :{'n_holes':1, 'height':50, 'width':5}, 
                            'freq_mask' :False, #{'freq_mask_param':100},
                            'time_mask' :False, #{'time_mask_param':5},
                            'pitchshift':False, #{'shift_range':48},
                            'addnoise'  :False, #{'noise_type':'pink', 'noise_size':0.01}, 
                            },
            MixUp_bool=True, beta_dist_alpha=0.75, # MixUp
            ):
    
    # labeled_data   shape: (10, 9, 174, 19)
    # labeled_label  shape: (10, 6)
    
    # --- Setup Augmentation Methods ---
    transform  = transform_method(transform_dict)
    
    # --- Normalization ---
    labeled_data   = Normalize(labeled_data)
    
    # --- Augmentation ---
    labeled_data  = transform(labeled_data)
    labeled_label = labeled_label[0]
    
    # --- MixUp ---
    if MixUp_bool:
        shuffle = torch.randperm(labeled_data.size(0))
        labeled_data, labeled_label = Mixup(labeled_data, labeled_label,
                                            labeled_data[shuffle], labeled_label[shuffle],
                                            beta_dist_alpha)
    
    # --- CUDA ---    
    labeled_data  = labeled_data.to(device)
    labeled_label = labeled_label.to(device, non_blocking=True)
    
    return labeled_data, labeled_label

def Normalize(data):
    return (data-torch.mean(data))/torch.std(data)

def Mixup(data, label,
          unlabel_data, unlabel_label,
          alpha):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.-lam)
    mixed_data  = lam*data  + (1.-lam)*unlabel_data
    mixed_label = lam*label + (1.-lam)*unlabel_label
    return mixed_data, mixed_label