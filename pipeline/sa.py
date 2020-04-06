import os
import math

import argparse

import torch
from torch.autograd import Variable

import model

import numpy as np
import seaborn as sns

def get_neuron_and_polarity(sd, neuron):
    """return a +/- 1 indicating the polarity of the specified neuron in the module"""
    if neuron == -1:
        neuron = None
    if 'classifier' in sd:
        sd = sd['classifier']
        if 'weight' in sd:
            weight = sd['weight']
        else:
            return neuron, 1
    else:
        return neuron, 1
    if neuron is None:
        val, neuron = torch.max(torch.abs(weight[0].float()), 0)
        # IndexError: invalid index of a 0-dim tensor. Use tensor.item() to convert a 0-dim tensor to a Python number
        #neuron = neuron.item[0]
        neuron = neuron.item()
    val = weight[0][neuron]
    if val >= 0:
        polarity = 1
    else:
        polarity = -1
    return neuron, polarity

def process_hidden(cell, hidden, neuron, mask=False, mask_value=1, polarity=1):
    feat = cell.data[:, neuron]
    rtn_feat = feat.clone()
    if mask:
#        feat.fill_(mask_value*polarity)
        hidden.data[:, neuron].fill_(mask_value*polarity)
    return rtn_feat[0]

def model_step(model, input, neuron=None, mask=False, mask_value=1, polarity=1):
    out, _ = model(input)
    if neuron is not None:
        hidden = model.rnn.rnns[-1].hidden
        if len(hidden) > 1:
            hidden, cell = hidden
        else:
            hidden = cell = hidden
        feat = process_hidden(cell, hidden, neuron, mask, mask_value, polarity)
        return out, feat
    return out

def sample(out, temperature):
    if temperature == 0:
        char_idx = torch.max(out.squeeze().data, 0)[1][0]
    else:
        word_weights = out.float().squeeze().data.div(temperature).exp().cpu()
        char_idx = torch.multinomial(word_weights, 1)[0]
    return char_idx

def process_text(text, model, input, temperature, neuron=None, mask=False, overwrite=1, polarity=1):
    chrs = []
    vals = []
    for c in text:
        input.data.fill_(int(ord(c)))
        if neuron:
            ch, val = model_step(model, input, neuron, mask, overwrite, polarity)
            vals.append(val)
        else:
            ch = model_step(model, input, neuron, mask, overwrite, polarity)
#        ch = sample(ch, temperature)
    input.data.fill_(sample(ch, temperature))
    chrs = list(text)
#    chrs.append(chr(ch))
    return chrs, vals

class Sentiment:
    def __init__(self, load_model, visualize):
        self.data_size = 256
        self.seed = -1
        self.model_nn = 'mLSTM'
        self.emsize = 64
        self.nhid = 4096
        self.nlayers = 1
        self.dropout = 0.0
        self.tied = False
        self.neuron = -1
        self.overwrite = None
        self.temperature = 1
        self.visualize = visualize
        self.gen_length = 0
        # Load the model
        self.load_model = load_model

    def initialize(self):
        
        cuda = torch.cuda.is_available()

        self.model_test = model.RNNModel(self.model_nn, self.data_size, self.emsize, self.nhid, self.nlayers, self.dropout, self.tied)
        
        if cuda:
            self.model_test.cuda()

        with open(self.load_model, 'rb') as f:
            self.sd = torch.load(f)

        self.model_test.load_state_dict(self.sd)
        print('Model loaded state dict')
        
        # Get the neuron and polarity
        self.neuron, self.polarity = get_neuron_and_polarity(self.sd, self.neuron)
        self.neuron = self.neuron if self.visualize or self.overwrite is not None else None
        self.mask = self.overwrite is not None

        # model_test train ?   
        self.model_test.eval()
        
        # Computing

        self.hidden = self.model_test.rnn.init_hidden(1)
        self.input = Variable(torch.LongTensor([int(ord('\n'))]))

        if cuda:
            self.input = self.input.cuda()

        self.input = self.input.view(1,1).contiguous()
        model_step(self.model_test, self.input, self.neuron, self.mask, self.overwrite, self.polarity)
        self.input.data.fill_(int(ord(' ')))
        out = model_step(self.model_test, self.input, self.neuron, self.mask, self.overwrite, self.polarity)
        if self.neuron is not None:
            out = out[0]
        self.input.data.fill_(sample(out, self.temperature))
        
    def process(self, text):
        outchrs = []
        outvals = []

        with torch.no_grad():
            
            chrs, vals = process_text(text, self.model_test, self.input, self.temperature, self.neuron, self.mask, self.overwrite, self.polarity)
            outchrs += chrs
            outvals += vals


            outstr = ''.join(outchrs)
            # Get each val
            outvals_list = [round(val.item() if type(val) != int else val, 3) for val in outvals]

            #make_heatmap(outstr, outvals, 'test')

        return outvals_list                                                            