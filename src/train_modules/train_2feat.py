import torch
from torch import nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import numpy as np
import random

use_cuda = torch.cuda.is_available()

def train_2feat(input_Var1, input_Var2, target_Var1, target_Var2, encoders, decoders, enc_opts, dec_opts, 
    loss_funcs, INPUT_SIZE1, INPUT_SIZE2, OUTPUT_SIZE, BATCH_SIZE, k=3):

    # encoder: Encoder
    # decoder: AttentionClassifier
    onEnc       = encoders[0]
    offEnc      = encoders[1]
    onDec       = decoders[0]
    offDec      = decoders[1]
    onEncOpt    = enc_opts[0]
    offEncOpt   = enc_opts[1]
    onDecOpt    = dec_opts[0]
    offDecOpt   = dec_opts[1]
    onLossFunc  = loss_funcs[0] 
    offLossFunc = loss_funcs[1] 
    
    input_batch = input_Var1.size()[0]
    input_time_step = input_Var1.size()[1]

    onEncOpt.zero_grad()
    onDecOpt.zero_grad()
    offEncOpt.zero_grad()
    offDecOpt.zero_grad()

    onLoss  = 0
    offLoss = 0

    window_size = 2*k+1
    
    for step in range((input_time_step//BATCH_SIZE)+1):
        if BATCH_SIZE*step > k and BATCH_SIZE*step < input_time_step - (k+1) - BATCH_SIZE:

            onEncHidden = onEnc.initHidden(BATCH_SIZE)
            offEncHidden = offEnc.initHidden(BATCH_SIZE)
            
            onEncOuts = torch.zeros(2*k+1, BATCH_SIZE, onEnc.hidden_size*2) if onEnc.bidir else torch.zeros(2*k+1, BATCH_SIZE, onEnc.hidden_size)
            offEncOuts = torch.zeros(2*k+1, BATCH_SIZE, offEnc.hidden_size*2) if offEnc.bidir else torch.zeros(2*k+1, BATCH_SIZE, offEnc.hidden_size)

            # Onset Encode Step
            for ei in range(window_size):
                enc_out, onEncHidden = onEnc(input_Var1[:, BATCH_SIZE*step-k+ei:BATCH_SIZE*step-k+ei+BATCH_SIZE, :].contiguous().view(BATCH_SIZE, 1, INPUT_SIZE1), onEncHidden)
                onEncOuts[ei] = enc_out.squeeze(1).data

            # To Onset Decoder
            onEncOuts = onEncOuts.transpose(0, 1)

            onDecAttnHidden = torch.cat((onEncHidden[0][2*onEnc.hidden_layer-1], onEncHidden[0][2*onEnc.hidden_layer-2]), 1) if onEnc.bidir else onEncHidden[0][onEnc.hidden_layer-1]
            onEncOuts = Variable(onEncOuts).cuda()

            # 1 step input (cause target only 1 time step)
            onDecOut, onDecAttn = onDec(onDecAttnHidden, onEncOuts)

            # To Offset Encoder
            for ei in range(window_size):
                enc_out, offEncHidden = offEnc(input_Var2[:, BATCH_SIZE*step-k+ei:BATCH_SIZE*step-k+ei+BATCH_SIZE, :].contiguous().view(BATCH_SIZE, 1, INPUT_SIZE2), offEncHidden)
                offEncOuts[ei] = enc_out.squeeze(1).data

            # To Offset Decoder
            offEncOuts = offEncOuts.transpose(0, 1)
            offDecAttnHidden = torch.cat((offEncHidden[0][2*offEnc.hidden_layer-1], offEncHidden[0][2*offEnc.hidden_layer-2]), 1) if offEnc.bidir else offEncHidden[0][offEnc.hidden_layer-1]
            offEncOuts = Variable(offEncOuts).cuda()

            offDecOut, offDecAttn = offDec(offDecAttnHidden, offEncOuts, onDecOut)
            
            for i in range(BATCH_SIZE):
                onLoss += onLossFunc(onDecOut[i].view(1, OUTPUT_SIZE), torch.max(target_Var1[:,BATCH_SIZE*step+i].contiguous().view(input_batch, 1, OUTPUT_SIZE), dim=2)[1].view(input_batch))
            
            for i in range(BATCH_SIZE):
                offLoss += offLossFunc(offDecOut[i].view(1, OUTPUT_SIZE), torch.max(target_Var2[:,BATCH_SIZE*step+i].contiguous().view(input_batch, 1, OUTPUT_SIZE), dim=2)[1].view(input_batch))
            
    #for i in range(input_batch):
    #    loss += loss_func(note_out_prob[i], torch.max(target_Var.contiguous().view(input_batch, -1, OUTPUT_SIZE), dim=2)[1].view(note_out_prob.shape[1]))
    
    Loss = onLoss + offLoss
    Loss.backward()

    onEncOpt.step()
    onDecOpt.step()
    offEncOpt.step()
    offDecOpt.step()

    return onLoss.item() / input_time_step , offLoss.item() / input_time_step