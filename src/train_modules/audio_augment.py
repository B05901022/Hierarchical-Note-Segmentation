# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:19:11 2019

@author: Austin Hsu
"""

import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import scipy.ndimage
import math

def transform_method(transform_dict):
    
    # Cut-out, Frequency Masking, Pitch shift ... etc
    
    transform_list = []
    
    try:
        if transform_dict['pitchshift'] != False:
            transform_list.append(PitchShifting(**transform_dict['pitchshift']))
        if transform_dict['cutout'] != False:
            transform_list.append(CutOut(**transform_dict['cutout']))
        if transform_dict['freq_mask'] != False:
            transform_list.append(FrequencyMasking(**transform_dict['freq_mask']))
        if transform_dict['time_mask'] != False:
            transform_list.append(TimeMasking(**transform_dict['time_mask']))
    except:
        raise ValueError("""
            transform_method() should contain a full dictionary with: 
            transform_dict={\'cutout\'    :{\'n_holes\':[cutout holes], \'height\':[cutout height], \'width\':[cutout width]}, 
                            \'freq_mask\' :{\'freq_mask_param\':[F parameter in SpecAugment]},
                            \'time_mask\' :{\'time_mask_param\':[T parameter in SpecAugment]},
                            \'pitchshift\':{\'shift_range\':2},
                             }
            for transforms unused, simply give a bool \'False\' for the dictionary key
            """)
    
    return transforms.Compose(transform_list)

class CutOut(object):
    '''
    Better with normalized input
    Switched to rectangular due to the input shape of (522,19)
    '''
    def __init__(self, n_holes, height, width):
        self.n_holes = n_holes
        self.height  = height
        self.width   = width
    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h,w), np.float32)
        
        for holes in range(self.n_holes):
            centre_y = np.random.randint(h)
            centre_x = np.random.randint(w)
            
            y1 = np.clip(centre_y - self.height // 2, 0, h)
            y2 = np.clip(centre_y + self.height // 2, 0, h)
            x1 = np.clip(centre_x - self.width  // 2, 0, w)
            x2 = np.clip(centre_x + self.width  // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0.
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img  = img * mask
        
        return img
    
class FrequencyMasking(object):
    '''
    Better with normalized input
    '''
    def __init__(self, freq_mask_param):
        self.F = freq_mask_param
    def __call__(self, img):
        v    = img.size(1)
        f    = np.random.randint(0, self.F)
        f0   = np.random.randint(0, v-f)
        img[:,f0:f0+f,:].fill_(0)        
        return img

class TimeMasking(object):
    '''
    Better with normalized input
    '''
    def __init__(self, time_mask_param):
        self.T = time_mask_param
    def __call__(self, img):
        tau  = img.size(2)
        t    = np.random.randint(0, self.T)
        t0   = np.random.randint(0, tau-t)
        img[:,:,t0:t0+t].fill_(0)        
        return img

class PitchShifting(object):
    def __init__(self, shift_range):
        self.shift_range = shift_range
    def __call__(self, img):
        f_range   = img.size(1)
        shift     = np.random.uniform(0, self.shift_range)
        shift_img = torch.zeros(img.size())
        up_bound  = math.ceil(f_range * shift)
        for f in range(f_range):
            shift_to = int(f/shift)
            if shift_to < f_range:
                shift_img[:,f] = img[:,shift_to]
        shift_img = shift_img.float()
        shift_img = (shift_img-torch.mean(shift_img))/torch.std(shift_img)
        shift_img[:,up_bound:,:] = 0.0
        return shift_img
    
################################################################################
# Resources: 
#   https://www.kaggle.com/huseinzol05/sound-augmentation-librosa
        