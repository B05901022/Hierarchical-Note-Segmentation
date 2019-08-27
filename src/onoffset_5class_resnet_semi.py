# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 11:13:53 2019

@author: Austin Hsu
"""

# onoffset_sdt.py
# onset & offset detection using Seq2seq AE
# Input: feat, Output: vector(note classification)

import torch
from onoffset_modules import ConcatDataset
import torchvision.models as models
from torch.autograd import Variable
import torch.utils.data as data_utils
import numpy as np
from argparse import ArgumentParser

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

from model_extend.PyramidNet_ShakeDrop import PyramidNet_ShakeDrop_MaxPool_9
from model_extend.ResNet_ShakeDrop import ResNet_ShakeDrop_9
from model_extend.AdamW import AdamW
from train_modules.train_5class_resnet_VAT import train_resnet_4loss_VAT
from train_modules.train_sdt6_resnet_mixmatch import train_resnet_4loss_mixmatch
from train_modules.train_sdt6_resnet_DataAug import train_resnet_4loss_DataAug
import os

#----------------------------
# Parser
#----------------------------
parser = ArgumentParser()
parser.add_argument("-d1", help="data file 1 position", dest="d1file", default="data.npy", type=str)
parser.add_argument("-a1", help="label file 1 position", dest="a1file", default="ans1.npy", type=str)
parser.add_argument("-dm1", help="decoder model 1 destination", dest="dm1file", default="model/onset_v4_dec", type=str)
parser.add_argument("-dmt1", help="train decoder model 1 destination", dest="dmt1file", default="model/onset_v4_tdec", type=str)
parser.add_argument("-p", help="present file number", dest="present_file", default=0, type=int)
parser.add_argument("-e", help="present epoch", dest="present_epoch", default=0, type=int)
parser.add_argument("-l", help="learning rate", dest="lr", default=0.001, type=float)
parser.add_argument("--window-size", help="input window size", dest="window_size", default=3, type=int)
parser.add_argument("--single-epoch", help="single turn training epoch", dest="single_epoch", default=5, type=int)
parser.add_argument("--batch-size", help="training batch size (frames)", dest="batch_size", default=10, type=int)
parser.add_argument("--feat1", help="feature cascaded", dest="feat_num1", default=1, type=int)
parser.add_argument("--loss-record", help="loss record file position", dest="lfile", default="loss.npy", type=str)

parser.add_argument("-u1", help="unlabeled data file 1 position", dest="u1dir", default="udata.npy", type=str)
parser.add_argument("-pretrain_model", help="use pretrained model of 10 epochs", dest="premod", default=False, type=bool)
parser.add_argument("-pretrain_dest", help="destination of pretrained model", dest="predest", default="baseline_models/PyramidNet_FreqMask_PitchShift_Baseline", type=str)

args = parser.parse_args()

#----------------------------
# Parameters
#----------------------------
on_data_file = args.d1file # Z file
on_ans_file = args.a1file  # marked onset/offset/pitch matrix file
on_udata_dir = args.u1dir
on_dec_model_file = args.dm1file # e.g. model_file = "model/onset_v3_bi_k3"
on_dec_model_train_file = args.dmt1file # e.g. model_file = "model/onset_v3_bi_k3"
loss_file = args.lfile
INPUT_SIZE1 = 174*args.feat_num1
OUTPUT_SIZE = 6
LR = args.lr
EPOCH = args.single_epoch
DATA_BATCH_SIZE = 1
BATCH_SIZE = args.batch_size
WINDOW_SIZE = args.window_size
PATIENCE = 700

PRESENT_FILE = args.present_file
PRESENT_EPOCH = args.present_epoch

PRETRAIN_BOOL = args.premod
PRETRAIN_DEST = args.predest

#----------------------------
# Data Collection
#----------------------------
print("Training File ", PRESENT_FILE, "| Turn: ", PRESENT_EPOCH)
try:
    myfile = open(on_data_file, 'r')
except IOError:
    print("Could not open file ", on_data_file)
    exit()

# === Answer from 6-class to 5-class ===
# 6-class : [S, A, O', O, X', X]
# 5-class : SO'X' : [1,0,1,0,1,0] >>> class 0
#           DO'X' : [0,1,1,0,1,0] >>> class 1
#           DO'X  : [0,1,1,0,0,1] >>> class 2
#           DOX'  : [0,1,0,1,1,0] >>> class 3
#           DOX   : [0,1,0,1,0,1] >>> class 4
# =======================================

with open(on_data_file, 'r') as fd1:
    with open(on_ans_file, 'r') as fa1:
        on_data_np = np.loadtxt(fd1)
        on_data_np = np.transpose(on_data_np)
        on_ans_np = np.loadtxt(fa1, delimiter=',')
        min_row = on_ans_np.shape[0] if (on_ans_np.shape[0] < on_data_np.shape[0]) else on_data_np.shape[0]
        on_data_np = on_data_np[:min_row].reshape((1, -1, int(args.feat_num1), 174)).transpose((0,2,3,1)) #
        on_ans_np = on_ans_np[:min_row].reshape((1,-1))
        on_data = torch.from_numpy(on_data_np).type(torch.FloatTensor)
        on_ans = torch.from_numpy(on_ans_np).type(torch.FloatTensor)

#----------------------------
# Unlabeled Data Collection
#----------------------------
unlabel_list = os.listdir(on_udata_dir)
on_udata_file = unlabel_list[np.random.randint(len(unlabel_list))]
on_udata_file = on_udata_dir + on_udata_file

try:
    myfile = open(on_udata_file, 'r')
except IOError:
    print("Could not open file ", on_udata_file)
    exit()

with open(on_udata_file, 'r') as fu1:
    on_udata_np = np.loadtxt(fu1)
    on_udata_np = np.transpose(on_udata_np)
    on_udata_np = on_udata_np.reshape((1, -1, int(args.feat_num1), 174)).transpose((0,2,3,1)) #
    on_udata = torch.from_numpy(on_udata_np).type(torch.FloatTensor)#.to(device)       

#train data
train_loader = data_utils.DataLoader(
    ConcatDataset(on_data, on_ans, on_udata), 
    batch_size=BATCH_SIZE,
    shuffle=False)

# load resnet50
"""
resnet18 = models.resnet18(pretrained=False)
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, OUTPUT_SIZE)
num_fout = resnet18.conv1.out_channels
resnet18.conv1 = nn.Conv2d(int(args.feat_num1//3), num_fout, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
resnet18.avgpool = nn.AvgPool2d(kernel_size=(17,1), stride=1, padding=0)
"""
#resnet18 = ResNet_ShakeDrop_9(depth=18, shakedrop=False)
resnet18 = PyramidNet_ShakeDrop_MaxPool_9(depth=110, shakedrop=True, num_class=5, alpha=270)

#----------------------------
# Model Initialize
#----------------------------
if PRESENT_FILE == 1 and PRESENT_EPOCH == 0:
    print("Re-initialize PyramidNet Module...")
    on_note_decoder = resnet18
    on_note_decoder.to(device)
    on_dec_optimizer = AdamW(on_note_decoder.parameters(), lr=LR)#AdamW(on_note_decoder.parameters(), lr=LR)
elif PRESENT_FILE == 1 and PRESENT_EPOCH == 10 and PRETRAIN_BOOL:
    print("Using pretrain PyramidNet model...")
    on_note_decoder = resnet18
    on_note_decoder.load_state_dict(torch.load(PRETRAIN_DEST))
    on_note_decoder.to(device)
    on_dec_optimizer = AdamW(on_note_decoder.parameters(), lr=LR)
    on_dec_optimizer.load_state_dict(torch.load(PRETRAIN_DEST+'.optim'))
else:
    on_note_decoder = resnet18
    on_note_decoder.load_state_dict(torch.load(on_dec_model_train_file))
    on_note_decoder.to(device)
    on_dec_optimizer = AdamW(on_note_decoder.parameters(), lr=LR)#AdamW(on_note_decoder.parameters(), lr=LR)#
    on_dec_optimizer.load_state_dict(torch.load(on_dec_model_train_file+'.optim'))

note_decoders = [on_note_decoder]
dec_optimizers = [on_dec_optimizer]
on_loss_func = torch.nn.CrossEntropyLoss()
loss_funcs = [on_loss_func]

#----------------------------
# Train Model
#----------------------------
min_loss = 10000.0
stop_count = 0
loss_list = []

for epoch in range(EPOCH):
    total_loss = 0
    loss_count = 0
    for step, xys in enumerate(train_loader):                 # gives batch data
        b_x1 = xys[0].contiguous() # reshape x to (batch, C, feat_size, time_frame)
        b_y1 = Variable(xys[1].contiguous().view(DATA_BATCH_SIZE, -1, OUTPUT_SIZE)) #Variable(xys[1].contiguous().view(DATA_BATCH_SIZE, -1, OUTPUT_SIZE)).to(device)
        b_u1 = xys[2].contiguous()
        
        # b_x1 shape: (1,9,174,songlength)
        # b_y1 shape: (1,songlength,5)
        
        loss = train_resnet_4loss_VAT(b_x1, b_y1, note_decoders, dec_optimizers, device,
                                      loss_funcs, INPUT_SIZE1, OUTPUT_SIZE, 
                                      BATCH_SIZE, k=WINDOW_SIZE,
                                      unlabel_t=b_u1, unlabel_lambda=1.0)
        
        """
        loss = train_resnet_4loss_mixmatch(b_x1, b_y1, note_decoders, dec_optimizers, device,
                                           loss_funcs, INPUT_SIZE1, OUTPUT_SIZE, 
                                           BATCH_SIZE, k=WINDOW_SIZE,
                                           unlabel_t=b_u1, unlabel_lambda=1.0)
        
        if PRESENT_EPOCH > 9:
            loss = train_resnet_4loss_mixmatch(b_x1, b_y1, note_decoders, dec_optimizers, device,
                                               loss_funcs, INPUT_SIZE1, OUTPUT_SIZE, 
                                               BATCH_SIZE, k=WINDOW_SIZE,
                                               unlabel_t=b_u1, unlabel_lambda=1.0)
        else:
            loss = train_resnet_4loss_DataAug(b_x1, b_y1, note_decoders, dec_optimizers, device,
                                              loss_funcs, INPUT_SIZE1, OUTPUT_SIZE,
                                              BATCH_SIZE, k=WINDOW_SIZE)
        """
        
        total_loss += loss
        loss_count += 1

        avg_loss = total_loss / loss_count

        if(step%10 == 0):
            print('Epoch: ', epoch, '| Step: ', step, '| Loss: %.4f' % loss)

        if avg_loss < min_loss:
            print("Renewing best model ...")
            min_loss = avg_loss
            torch.save(note_decoders[0].state_dict(), on_dec_model_file)
            torch.save(dec_optimizers[0].state_dict(), on_dec_model_file+'.optim')
            stop_count = 0
        else:
            stop_count += 1

        if stop_count > PATIENCE:
            print("Early stopping...")
            exit()
    
    loss_list.append(avg_loss)

torch.save(note_decoders[0].state_dict(), on_dec_model_train_file)
torch.save(dec_optimizers[0].state_dict(), on_dec_model_train_file+'.optim')

if PRESENT_EPOCH == 9 and PRESENT_FILE == 82:
    torch.save(note_decoders[0].state_dict(), on_dec_model_train_file+'_10')
    torch.save(dec_optimizers[0].state_dict(), on_dec_model_train_file+'_10.optim')

if PRESENT_EPOCH == 0 and PRESENT_FILE == 1:
    print("Re-initialize Loss Record File ...")
    with open(loss_file, 'w') as flo:
        for i in range(len(loss_list)):
            flo.write("{:.5f}\n".format(loss_list[i]))
else:
    print("Writing Loss to Loss Record File ...")
    with open(loss_file, 'a') as flo:
        for i in range(len(loss_list)):
            flo.write("{:.5f}\n".format(loss_list[i]))
