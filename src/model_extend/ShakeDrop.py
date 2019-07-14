# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:26:13 2019

@author: Austin Hsu
"""

import torch
import torch.nn as nn
import math

#######################################################################################
# ShakeDrop Reference
# https://github.com/owruby/shake-drop_pytorch/
#

class ShakeDropFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpharange=[-1,1]):
        if training:
            gate = torch.cuda.FloatTensor([0]).bernoulli_(1-p_drop)
            ctx.save_for_backward(gate)
            if gate.item() == 0:
                alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpharange)
                alpha = alpha.view(alpha.size(0),1,1,1).expand_as(x)
                return alpha * x
            else:
                return x
        else:
            return (1-p_drop) * x
    
    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        if gate.item() == 0:
            beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_([0,1])
            beta = beta.view(beta.size(0),1,1,1).expand_as(grad_output)
            beta = torch.autograd.Variable(beta)
            return beta * grad_output, None, None, None
        else:
            return grad_output, None, None, None
            
class ShakeDrop(nn.Module):
    
    def __init__(self, p_drop, alpha=[-1,1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha  = alpha
    
    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, p_drop=self.p_drop, 
                                       alpharange=self.alpha)

#######################################################################################

class BasicBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, 
                 stride=1, shakedrop=False, p_shakedrop=1.0):
        super(BasicBlock, self).__init__()
        self.branch = nn.Sequential(nn.BatchNorm2d(in_channel),
                                    nn.Conv2d(in_channel, out_channel, kernel_size=3, 
                                              padding=1, stride=stride, bias=False),
                                    nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channel, out_channel, kernel_size=3, 
                                              padding=1, stride=stride, bias=False),
                                    nn.BatchNorm2d(out_channel))
        self.downsample = stride == 2
        self.shortcut   = nn.AvgPool2d(2) if self.downsample else None
        
        self.shakedrop  = shakedrop
        if self.shakedrop:
            self.shakedrop_layer = ShakeDrop(p_shakedrop, [-1,1])
        
    def forward(self, x):
        print('input', x.shape)
        h0 = self.shortcut(x) if self.downsample else x
        print('h0',h0.shape)
        h  = self.branch(x)
        print('h', h.shape)
        h  = self.shakedrop_layer(h) if self.shakedrop else h
        print('h', h.shape)
        
        #padding zero is enough for pyramidNet
        pad_zero = torch.autograd.Variable(torch.zeros(h0.size(0),
                                                       h.size(1)-h0.size(1),
                                                       h0.size(2),
                                                       h0.size(3)).float()).cuda()
        print('pad_zero', pad_zero.shape)
        h0  = torch.cat([h0, pad_zero], dim=1)
        print('h0', h0.shape)
        return h+h0        

class PyramidNet_ShakeDrop(nn.Module):
    
    def __init__(self, conv1_in_channel=3, depth=20, alpha=270, num_class=6,
                 shakedrop=False, block=BasicBlock):
        super(PyramidNet_ShakeDrop, self).__init__()
        
        if (depth-2) % 6 != 0:
            raise ValueError('depth should be one of 20, 32, 44, 56, 110, 1202')
        
        self.in_ch     = 16
        self.shakedrop = shakedrop
        block          = BasicBlock
        
        # PyramidNet
        n_units = (depth - 2) // 6
        channel = lambda x: math.ceil( alpha * (x+1) / (3 * n_units) )
        self.in_chs  = [self.in_ch] + [self.in_ch + channel(i) for i in range (n_units * 3)]
        print(self.in_chs)
        
        # Stochastic Depth
        self.p_L = 0.5
        linear_decay = lambda x: (1-self.p_L) * x / (3*n_units)
        self.ps_shakedrop = [linear_decay(i) for i in range(3*n_units)]
        
        self.u_idx   = 0
        
        ### Model ###
        
        # input shape (batch, 3, 174, 19)
        
        self.conv1   = nn.Conv2d(conv1_in_channel, self.in_chs[0], kernel_size=(7,7),
                               stride=(2,2), padding=(3,3), bias=False)
        self.bn1     = nn.BatchNorm2d(self.in_chs[0])
        self.relu1   = nn.ReLU(inplace=True)
        
        self.layer1  = self._make_layer(n_units, block, 1)
        self.layer2  = self._make_layer(n_units, block, 2)
        self.layer3  = self._make_layer(n_units, block, 2)
        
        self.bn_out  = nn.BatchNorm2d(self.in_chs[-1])
        self.relu_out= nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=(3,3), stride=(2,2), padding=1)
        self.fc_out  = nn.Linear(self.in_chs[-1], num_class)
        
        #output shape (batch, 6)
        
    def forward(self, x):
        print('1',x.shape)
        h = self.relu1(self.bn1(self.conv1(x)))
        print('2',h.shape)
        h = self.layer1(h)
        print('3',h.shape)
        print('==================================================')
        h = self.layer2(h)
        print('4',h.shape)
        h = self.layer3(h)
        print('5',h.shape)
        h = self.relu_out(self.bn_out(h))
        print('6',h.shape)
        h = self.avgpool(h)
        print('7',h.shape)
        h = h.view(h.size(0), -1)
        print('8',h.shape)
        h = self.fc_out(h)
        print('9',h.shape)
        return h
    
    def _make_layer(self, n_units, block, stride=1):
        layers = []
        for i in range(int(n_units)):
            layers.append(block(self.in_chs[self.u_idx], self.in_chs[self.u_idx+1],
                                stride, self.shakedrop, self.ps_shakedrop[self.u_idx]))
            self.u_idx += 1
            stride = 1
        return nn.Sequential(*layers)
            
if __name__ == '__main__':
    from torchsummaryX import summary
    model = PyramidNet_ShakeDrop().cuda()
    summary(model, torch.zeros(1,3,174,19).cuda())