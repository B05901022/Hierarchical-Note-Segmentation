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
from train_modules.VAT import VATLoss

def train_resnet_4loss_VAT(input_t, target_Var, decoders, dec_opts, device,
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k,
    unlabel_t, unlabel_lambda,
    ):
    
    # input_t    shape: (1,3,522,data_length)
    # target_Var shape: (1,data_length,6)
    # unlabel_t  shape: (1,3,522,unlabel_data_length)
    
    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] #CrossEntropyLoss_for_MixMatch()
    smLossFunc  = VATLoss()
    enLossFunc  = EntropyLoss()

    input_time_step   = input_t.size()[3]
    unlabel_time_step = unlabel_t.size()[3]

    window_size = 2*k+1
    
    totLoss = 0

    nn_softmax = nn.Softmax(dim=1)

    for step in range(k, input_time_step - k - BATCH_SIZE + 1, BATCH_SIZE): 
        
        # --- Loss ---
        super_Loss = 0
        smsup_Loss = 0
        en_Loss    = 0
        onLoss     = 0 
    
        # --- Data Collection ---        
        x_unmix_data = torch.stack([ input_t[0, :, :, step+i-k:step+i-k+window_size] for i in range(BATCH_SIZE)], dim=0)
        random_position = torch.randperm(unlabel_time_step-1-window_size)[:BATCH_SIZE]
        if step+BATCH_SIZE-k+window_size < unlabel_time_step:
            u_unmix_data = torch.stack([ unlabel_t[0, :, :, step+i-k:step+i-k+window_size] for i in range(BATCH_SIZE)], dim=0)
        else:
            u_unmix_data = torch.stack([ unlabel_t[0, :, :, random_position[i]:random_position[i]+window_size] for i in range(BATCH_SIZE)], dim=0)
        
        # ---Data Preprocessing ---
        x_mix_data, u_mix_data, x_mix_label = DataPreprocess(labeled_data=x_unmix_data,
                                                             labeled_label=target_Var[:, step:step+BATCH_SIZE],
                                                             unlabeled_data=u_unmix_data,
                                                             device=device
                                                             )
        
        # --- Pseudo Label ---
        # mix_label = torch.cat((x_mix_label, u_mix_label), dim=0)
        
        # --- Variable ---
        x_mix_data = Variable(x_mix_data)
        u_mix_data = Variable(u_mix_data)
        x_mix_label = Variable(x_mix_label) ###
        
        # --- Run Model ---
        onDecOut_mix = onDec(x_mix_data) 
        onDecOut6    = onDecOut_mix
        
        # === labeled ===
        onDecOut1   = nn_softmax(onDecOut6[:, :2])
        onDecOut2   = nn_softmax(onDecOut6[:, 2:4])
        onDecOut3   = nn_softmax(onDecOut6[:, 4:])
        
        temp_t = torch.max(onDecOut2[:, 1], onDecOut3[:, 1]).view(-1,1)
        onDecOut4 = torch.cat((onDecOut1, temp_t), dim=1)
        
        # --- Loss ---        
        # === Supervised Loss ===
        super_Loss += onLossFunc(onDecOut1.view(-1, 2), x_mix_label[:,  :2].contiguous().view(-1, 2))
        super_Loss += onLossFunc(onDecOut2.view(-1, 2), x_mix_label[:, 2:4].contiguous().view(-1, 2))
        super_Loss += onLossFunc(onDecOut3.view(-1, 2), x_mix_label[:, 4: ].contiguous().view(-1, 2))
        target_T = torch.max(x_mix_label[:, 3], x_mix_label[:, 5])
        super_Loss += onLossFunc(onDecOut4.view(-1, 3), torch.cat((x_mix_label[:, :2].contiguous().view(-1, 2), 
                                                                  target_T.contiguous().view(-1, 1)), 1))     
        
        # === Entropy Minimization ===
        en_Loss += enLossFunc(onDecOut1.view(-1, 2))
        en_Loss += enLossFunc(onDecOut2.view(-1, 2))
        en_Loss += enLossFunc(onDecOut3.view(-1, 2))
        
        # === VAT Loss ===
        #smsup_Loss += smLossFunc(onDec, u_mix_data)
        
        print('total_Loss: %.10f' % (super_Loss.item() / input_time_step), 'entropy_Loss: %.10f' % (en_Loss.item() / input_time_step)) # 'semi-supervised_Loss: %.10f' % (unlabel_lambda * smsup_Loss.item() / input_time_step)
        onLoss = super_Loss + en_Loss #+ unlabel_lambda * smsup_Loss 
        onDecOpt.zero_grad()
        onLoss.backward()
        onDecOpt.step()
        totLoss += onLoss.item()
    
    return totLoss / input_time_step 

def DataPreprocess(labeled_data, labeled_label,
                   unlabeled_data,
                   device,
                   transform_dict={'cutout'    :False, #{'n_holes':1, 'height':50, 'width':5}, 
                                   'freq_mask' :{'freq_mask_param':100},
                                   'time_mask' :False, #{'time_mask_param':5},
                                   'pitchshift':{'shift_range':48}, 
                                   'addnoise'  :False, #{'noise_type':'pink', 'noise_size':0.01}, 
                                   },
                   augment_time=1,
                   ):
    
    # labeled_data   shape: (batchsize, 9, 174, 19)
    # labeled_label  shape: (batchsize, 6)
    # unlabeled_data shape: (batchsize, 9, 174, 19)
    
    # --- Setup Augmentation Methods ---
    transform  = transform_method(transform_dict)
    
    # --- Normalization ---
    labeled_data   = Normalize(labeled_data)
    unlabeled_data = Normalize(unlabeled_data)
    
    # --- Labeled Augmentation ---
    aug_x   = transform(labeled_data)
    label_x = labeled_label[0]
    
    # --- Unlabeled Augmentation ---
    aug_u = transform(unlabeled_data)
    
    # --- Shuffle ---
    shuffle  = torch.randperm(aug_x.size(0))
    shuffle2 = torch.randperm(aug_u.size(0))
    aug_x    = aug_x[shuffle]
    aug_u    = aug_u[shuffle2]
    label_x  = label_x[shuffle]
    
    # --- CUDA ---
    aug_x  = aug_x.to(device)
    aug_u  = aug_u.to(device)
    labeled_label = labeled_label.to(device, non_blocking=True)
    
    return aug_x, aug_u, label_x

def Normalize(data):
    # Batchwise normalization (test)
    return (data-torch.mean(data))/torch.std(data)
    
class EntropyLoss(nn.Module):
    def __init__(self, entmin_weight=1.0):
        super(EntropyLoss, self).__init__()
        self.entmin_weight = entmin_weight
    def forward(self, softmax_x):
        return -self.entmin_weight * torch.mean(softmax_x * torch.log(softmax_x))  
