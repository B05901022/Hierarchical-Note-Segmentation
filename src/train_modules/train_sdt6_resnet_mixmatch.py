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

def train_resnet_4loss_mixmatch(input_t, target_Var, decoders, dec_opts, device,
    loss_funcs, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, k,
    unlabel_t, unlabel_lambda=1.0,
    ):
    
    # input_t    shape: (1,3,522,data_length)
    # target_Var shape: (1,data_length,6)
    # unlabel_t  shape: (1,3,522,unlabel_data_length)
    
    # encoder: Encoder
    # decoder: AttentionClassifier
    onDec       = decoders[0]
    onDecOpt    = dec_opts[0]
    onLossFunc  = loss_funcs[0] #CrossEntropyLoss_for_MixMatch()
    smLossFunc  = VATLoss() #nn.MSELoss()
    enLossFunc  = EntropyLoss()

    input_time_step   = input_t.size()[3]
    unlabel_time_step = unlabel_t.size()[3]
    unlabel_aug_time  = 2
    total_time_step   = input_time_step #+ unlabel_time_step * unlabel_aug_time

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
        try:
            print('1')
            u_unmix_data = torch.stack([ unlabel_t[0, :, :, step+i-k:step+i-k+window_size] for i in range(BATCH_SIZE)], dim=0)
        except:
            print('2')
            u_unmix_data = torch.stack([ unlabel_t[0, :, :, random_position[i]:random_position[i]+window_size] for i in range(BATCH_SIZE)], dim=0)
        
        print(u_unmix_data.shape)
        
        # --- MixMatch/MixUp ---
        mix_data, x_mix_label, u_mix_label = Mixmatch(labeled_data=x_unmix_data,
                                                      labeled_label=target_Var[:, step:step+BATCH_SIZE],
                                                      unlabeled_data=u_unmix_data,
                                                      curr_model=onDec,
                                                      device=device
                                                      )
        
        """
        # --- Variable ---
        #x_mix_data = Variable(x_mix_data)
        #u_mix_data = Variable(u_mix_data)
        x_mix_label = Variable(x_mix_label) ###
        u_mix_label = Variable(u_mix_label) ###
        """
        
        """
        # === Labeled ===
        #input_Var shape: (10,3,522,19)
        onDecOut6   = onDec(x_mix_data)
        onDecOut1   = nn_softmax(onDecOut6[:, :2])
        onDecOut2   = nn_softmax(onDecOut6[:, 2:4])
        onDecOut3   = nn_softmax(onDecOut6[:, 4:])
        
        temp_t = torch.max(onDecOut2[:, 1], onDecOut3[:, 1]).view(-1,1)
        onDecOut4 = torch.cat((onDecOut1, temp_t), dim=1)
        
        # === Unlabeled ===
        
        onDecOut6_u = onDec(u_mix_data)
        onDecOut1_u = nn_softmax(onDecOut6_u[:, :2])
        onDecOut2_u = nn_softmax(onDecOut6_u[:, 2:4])
        onDecOut3_u = nn_softmax(onDecOut6_u[:, 4:])
        temp_t2 = torch.max(onDecOut2_u[:, 1], onDecOut3_u[:, 1]).view(-1,1)
        onDecOut4_u = torch.cat((onDecOut1_u, temp_t2), dim=1)

        #onDecOutT_u = torch.cat((onDecOut1_u, onDecOut2_u, onDecOut3_u), dim=1)
        """
        
        # --- Run Model ---
        onDecOut_mix = onDec(mix_data[:BATCH_SIZE]) ###
        onDecOut6    = onDecOut_mix#[:BATCH_SIZE] # labeled
        #onDecOut6_u  = onDecOut_mix[BATCH_SIZE:] # unlabeled
        
        # === labeled ===
        onDecOut1   = nn_softmax(onDecOut6[:, :2])
        onDecOut2   = nn_softmax(onDecOut6[:, 2:4])
        onDecOut3   = nn_softmax(onDecOut6[:, 4:])
        
        temp_t = torch.max(onDecOut2[:, 1], onDecOut3[:, 1]).view(-1,1)
        onDecOut4 = torch.cat((onDecOut1, temp_t), dim=1)
        
        # === unlabeled ===
        #onDecOut1_u = nn_softmax(onDecOut6_u[:, :2])
        #onDecOut2_u = nn_softmax(onDecOut6_u[:, 2:4])
        #onDecOut3_u = nn_softmax(onDecOut6_u[:, 4:])
        #temp_t2 = torch.max(onDecOut2_u[:, 1], onDecOut3_u[:, 1]).view(-1,1)
        #onDecOut4_u = torch.cat((onDecOut1_u, temp_t2), dim=1)
        
        # --- Loss ---
        """
        # === Labeled ===
        for i in range(BATCH_SIZE):
            super_Loss += onLossFunc(onDecOut1[i].view(1, 2), x_mix_label[:,i, :2].contiguous().view(1, 2))
            super_Loss += onLossFunc(onDecOut2[i].view(1, 2), x_mix_label[:,i, 2:4].contiguous().view(1, 2))
            super_Loss += onLossFunc(onDecOut3[i].view(1, 2), x_mix_label[:,i, 4:].contiguous().view(1, 2))
            target_T = torch.max(x_mix_label[:,i, 3], x_mix_label[:,i, 5])
            super_Loss += onLossFunc(onDecOut4[i].view(1, 3), torch.cat((x_mix_label[:,i, :2].contiguous().view(1, 2), 
                                     target_T.contiguous().view(1, 1)), 1))
            
        # === Unlabeled ===
        for i in range(BATCH_SIZE*unlabel_aug_time):
            # Add L2 loss for unlabeled data (Hierachical)
            smsup_Loss += smLossFunc(onDecOut1_u[i].view(1, 2), u_mix_label[:,i, :2].contiguous().view(1, 2))
            smsup_Loss += smLossFunc(onDecOut2_u[i].view(1, 2), u_mix_label[:,i, 2:4].contiguous().view(1, 2))
            smsup_Loss += smLossFunc(onDecOut3_u[i].view(1, 2), u_mix_label[:,i, 4:].contiguous().view(1, 2))
            target_T2 = torch.max(u_mix_label[:,i, 3], u_mix_label[:,i, 5])
            smsup_Loss += smLossFunc(onDecOut4_u[i].view(1, 3), torch.cat((u_mix_label[:,i, :2].contiguous().view(1, 2), 
                                                      target_T2.contiguous().view(1, 1)), 1))
            
            # Add L2 loss for unlabeled data
            #smsup_Loss += smLossFunc(onDecOutT_u[i].view(1, 6), u_mix_label[:,i].contiguous().view(1, 6))
        """
        
        # === Labeled ===
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
        smsup_Loss += smLossFunc(onDec, mix_data[BATCH_SIZE:])
        
        # === MixMatch ===
        # Add L2 loss for unlabeled data (Hierachical)
        #smsup_Loss += smLossFunc(onDecOut1_u.view(-1, 2), u_mix_label[:,  :2].contiguous().view(-1, 2))
        #smsup_Loss += smLossFunc(onDecOut2_u.view(-1, 2), u_mix_label[:, 2:4].contiguous().view(-1, 2))
        #smsup_Loss += smLossFunc(onDecOut3_u.view(-1, 2), u_mix_label[:, 4: ].contiguous().view(-1, 2))
        #target_T2 = torch.max(u_mix_label[:, 3], u_mix_label[:, 5])
        #smsup_Loss += smLossFunc(onDecOut4_u.view(-1, 3), torch.cat((u_mix_label[:, :2].contiguous().view(-1, 2), 
        #                                                            target_T2.contiguous().view(-1, 1)), 1))
            
        print('supervised_Loss: %.10f' % (super_Loss.item() / input_time_step), 'entropy_Loss: %.10f' % (en_Loss.item() / input_time_step), 'semi-supervised_Loss: %.10f' % (smsup_Loss.item() / input_time_step))
        onLoss = super_Loss + en_Loss + unlabel_lambda * smsup_Loss
        onDecOpt.zero_grad()
        onLoss.backward()
        onDecOpt.step()
        totLoss += onLoss.item()
    
    return totLoss / input_time_step #total_time_step

def Mixmatch(labeled_data, labeled_label,
             unlabeled_data,
             curr_model,
             device,
             TSA_bool=False, curr_timestep=0, total_timestep=0, TSA_k=6, TSA_schedule='exp', 
             transform_dict={'cutout'    :False, #{'n_holes':1, 'height':50, 'width':5}, 
                             'freq_mask' :{'freq_mask_param':100},
                             'time_mask' :False, #{'time_mask_param':5},
                             'pitchshift':{'shift_range':48}, 
                             'addnoise'  :False, #{'noise_type':'pink', 'noise_size':0.01}, 
                             }, # Cut-out, Frequency/Time Masking, Pitch shift 
             sharpening_temp=0.5, augment_time=1, beta_dist_alpha=0.75):
    
    # labeled_data   shape: (10, 9, 174, 19)
    # labeled_label  shape: (10, 6)
    # unlabeled_data shape: (10, 9, 174, 19)
    
    # --- Setup Augmentation Methods ---
    transform  = transform_method(transform_dict)
    
    # --- Normalization ---
    labeled_data   = Normalize(labeled_data)
    unlabeled_data = Normalize(unlabeled_data)
    
    # --- Labeled Augmentation ---
    aug_x = transform(labeled_data)
    labeled_label = labeled_label[0]
    
    # --- Unlabeled Augmentation ---
    aug_u = []
    label = None
    with torch.no_grad():
        for k in range(augment_time):
            aug_u_k = transform(unlabeled_data).to(device)
            aug_u.append(aug_u_k)
            label_guess = curr_model(aug_u_k)
            # --- Hierachical Label Guess ---
            if len(aug_u) == 1:
                label = torch.cat((F.softmax(label_guess[:, :2], dim=1),
                                   F.softmax(label_guess[:,2:4], dim=1),
                                   F.softmax(label_guess[:,4: ], dim=1)), dim=1)
            else:
                label += torch.cat((F.softmax(label_guess[:, :2], dim=1),
                                    F.softmax(label_guess[:,2:4], dim=1),
                                    F.softmax(label_guess[:,4: ], dim=1)), dim=1)
            # --------------------------------
        label /= augment_time
        label = Sharpen(label, sharpening_temp)
    
    # --- TSA ---
    if TSA_bool:
        tsa_detect = TSA(total_timestep, TSA_k, TSA_schedule)
        accept_label = tsa_detect(label, curr_timestep)
        label = label[accept_label]
        aug_u = aug_u[torch.stack([accept_label*(i+1) for i in range(augment_time)])]   
        
    # --- CUDA ---
    aug_x  = aug_x.to(device)
    labeled_label = labeled_label.to(device, non_blocking=True)
    
    # --- Mixing Data ---
    stack_data  = torch.cat((aug_x, *aug_u), dim=0)
    stack_label = torch.cat((labeled_label, *augment_time*[label]), dim=0)
    
    # --- MixUp ---
    shuffle = torch.randperm(stack_data.size(0)) #torch.arange(stack_data.size(0))
    x_mix_data, x_mix_label   = Mixup(aug_x, labeled_label, 
                                      stack_data[shuffle[:aug_x.size(0)]], stack_label[shuffle[:aug_x.size(0)]],
                                      beta_dist_alpha)
    u_mix_data, u_mix_label   = Mixup(torch.cat(aug_u, dim=0), torch.cat([label for i in range(augment_time)], dim=0), 
                                      stack_data[shuffle[aug_x.size(0):]], stack_label[shuffle[aug_x.size(0):]],
                                      beta_dist_alpha)
    
    # --- Concatenate X,U data ---
    mix_data = torch.cat((x_mix_data, u_mix_data), dim=0)
    
    return mix_data, x_mix_label, u_mix_label

def Normalize(data):
    # Batchwise normalization (test)
    return (data-torch.mean(data))/torch.std(data)

def Sharpen(dist, T):
    sharpen_dist = dist ** (1./T)
    #sharpen_dist = sharpen_dist / sharpen_dist.sum(dim=1, keepdim=True)
    sharpen_dist[:, :2] = sharpen_dist[:, :2] / sharpen_dist[:, :2].sum(dim=1, keepdim=True)
    sharpen_dist[:,2:4] = sharpen_dist[:,2:4] / sharpen_dist[:,2:4].sum(dim=1, keepdim=True)
    sharpen_dist[:,4: ] = sharpen_dist[:,4: ] / sharpen_dist[:,4: ].sum(dim=1, keepdim=True)
    return sharpen_dist

def Mixup(data, label,
          unlabel_data, unlabel_label,
          alpha):
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.-lam)
    mixed_data  = lam*data  + (1.-lam)*unlabel_data
    mixed_label = lam*label + (1.-lam)*unlabel_label
    return mixed_data, mixed_label

class TSA(object):
    def __init__(self, total_timestep, k, schedule):
        self.T = total_timestep
        self.K = k # total catagories
        if schedule == 'log':
            self.schedule = lambda t: (1-math.e**(-5*t/self.T)) * (1-1./self.K) + 1./self.K
        elif schedule == 'linear':
            self.schedule = lambda t: t/self.T * (1-1./self.K) + 1./self.K
        elif schedule == 'exp':
            self.schedule = lambda t: math.e**(5*(t/self.T-1)) * (1-1./self.K) + 1./self.K
        else:
            raise ValueError("Only \'log\', \'linear\', \'exp\' are valid TSA type")
    def __call__(self, label, t):
        threshold = self.schedule(t)
        false_tag = torch.gt(torch.max(label, dim=1)[0], threshold)
        right_tag = [i for i in range(false_tag.size(0)) if false_tag[i] == 0]
        return torch.Tensor(right_tag).long()
    
class EntropyLoss(nn.Module):
    def __init__(self, entmin_weight=1.0):
        super(EntropyLoss, self).__init__()
        self.entmin_weight = entmin_weight
    def forward(self, softmax_x):
        return -self.entmin_weight * torch.mean(softmax_x * torch.log(softmax_x))  
    
class CrossEntropyLoss_for_MixMatch(object):
    def __call__(self, softmax_x, target_x):
        return -torch.mean(torch.sum(torch.log(softmax_x)*target_x,dim=1))