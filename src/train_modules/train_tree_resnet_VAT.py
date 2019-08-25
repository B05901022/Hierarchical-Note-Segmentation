# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:43:16 2019

@author: Austin Hsu
"""

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
from train_modules.VAT import VATLoss_tree

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
    onLossFunc  = nn.CrossEntropyLoss() #LabelSmoothingLoss()
    smLossFunc  = VATLoss_tree() # can try ip=1
    enLossFunc  = EntropyLoss()
    
    target_Var  = ToLabel(ToOneHot(target_Var[0])) # to index label

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
                                                             labeled_label=target_Var[step:step+BATCH_SIZE],
                                                             unlabeled_data=u_unmix_data,
                                                             device=device
                                                             )
        
        # --- Pseudo Label ---
        # mix_label = torch.cat((x_mix_label, u_mix_label), dim=0)
        
        # --- Variable ---
        x_mix_data = Variable(x_mix_data)
        u_mix_data = Variable(u_mix_data)
        x_mix_label = Variable(x_mix_label)
        
        # --- Run Model ---
        onDecOut_mix = onDec(torch.cat((x_mix_data, u_mix_data),dim=0)) #onDec(x_mix_data) 
        onDecOut6    = onDecOut_mix[:BATCH_SIZE]
        onDecOut6    = nn_softmax(onDecOut6.view(3,-1,2), dim=2).view(-1,6)
        
        # === labeled ===
        onDecOut6    = ToOneHot(onDecOut6)
        
        # --- Loss ---        
        # === Supervised Loss ===
        super_Loss += onLossFunc(onDecOut6, x_mix_label.contiguous())   
        
        # === Entropy Minimization ===
        # --- labeled ---
        en_Loss    += enLossFunc(onDecOut6)
        # --- unlabeled ---
        onDecOut6_u = onDecOut_mix[BATCH_SIZE:]
        onDecOut6_u = nn_softmax(onDecOut6_u.view(3,-1,2), dim=2).view(-1,6)
        en_Loss    += enLossFunc(onDecOut6_u)
        
        # === VAT Loss ===
        smsup_Loss += smLossFunc(onDec, u_mix_data)
        
        print('supervised_Loss: %.10f' % (super_Loss.item() / input_time_step), 'semi-supervised_Loss: %.10f' % (unlabel_lambda * smsup_Loss.item() / input_time_step)) #'entropy_Loss: %.10f' % (en_Loss.item() / input_time_step)
        onLoss = super_Loss + unlabel_lambda * smsup_Loss + en_Loss
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
    def forward(self, x):
        return -self.entmin_weight * torch.mean(x * torch.log(x)) 

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smooth_eps=0.1, num_class):
        super(LabelSmoothingLoss, self).__init__()
        self.smooth_eps = smooth_eps
        self.num_class  = num_class
    def forward(self, x, target):
        # x : log_softmax output
        # === One-hot ===
        target = target.unsqueeze(dim=1)
        target = torch.zeros(target.size(0), self.num_class).scatter_(1, target, 1)
        # ===============
        smooth_target = (1.-self.smooth_eps) * target + self.smooth_eps * torch.Tensor(target.size()).fill_(1./self.num_class)
        return F.nll_loss(x, smooth_target)

def ToOneHot(input_label):
    p_SOnXn = input_label[:,0].unsqueeze(1)
    p_DOnXn = (input_label[:,1]*input_label[:,2]*input_label[:,4]).unsqueeze(1)
    p_DOnX  = (input_label[:,1]*input_label[:,2]*input_label[:,5]).unsqueeze(1)
    p_DOXn  = (input_label[:,1]*input_label[:,3]*input_label[:,4]).unsqueeze(1)
    p_DOX   = (input_label[:,1]*input_label[:,3]*input_label[:,5]).unsqueeze(1)
    return torch.cat([p_SOnXn, p_DOnXn, p_DOnX, p_DOXn, p_DOX], dim=1)

def ToLabel(input_onehot):
    return input_onehot.argmax(dim=1)