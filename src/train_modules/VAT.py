# -*- coding: utf-8 -*-
# --- Reference ---
# https://github.com/lyakaap/VAT-pytorch/blob/master/vat.py
# -----------------

from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F

@contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)
    
def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= (torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8)
    return d

class VATLoss(nn.Module):

    def __init__(self, xi=1e-6, eps=40.0, ip=2):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            #pred = F.softmax(model(x), dim=1)
            pred = F.softmax(model(x).view(3,-1,2), dim=2).view(-1,6)
            
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                #logp_hat = F.softmax(pred_hat, dim=1)
                logp_hat = F.log_softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
    
class VATLoss_5class(nn.Module):

    def __init__(self, xi=1e-6, eps=40.0, ip=2):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss_5class, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)
            
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
    
class VATLoss_tree(nn.Module):

    def __init__(self, xi=1e-6, eps=40.0, ip=2):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss_tree, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x).view(3,-1,2), dim=2).view(-1,6)
            pred = ToOneHot(pred)
            
        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                pred_hat = F.softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)
                pred_hat = ToOneHot(pred_hat)
                logp_hat = torch.log(torch.clamp(pred_hat,1e-8))
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            pred_hat = F.softmax(pred_hat.view(3,-1,2), dim=2).view(-1,6)
            pred_hat = ToOneHot(pred_hat)
            logp_hat = torch.log(pred_hat)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds
    
def ToOneHot(input_label):
    p_SOnXn = input_label[:,0].unsqueeze(1)
    p_DOnXn = (input_label[:,1]*input_label[:,2]*input_label[:,4]).unsqueeze(1)
    p_DOnX  = (input_label[:,1]*input_label[:,2]*input_label[:,5]).unsqueeze(1)
    p_DOXn  = (input_label[:,1]*input_label[:,3]*input_label[:,4]).unsqueeze(1)
    p_DOX   = (input_label[:,1]*input_label[:,3]*input_label[:,5]).unsqueeze(1)
    return torch.cat([p_SOnXn, p_DOnXn, p_DOnX, p_DOXn, p_DOX], dim=1)