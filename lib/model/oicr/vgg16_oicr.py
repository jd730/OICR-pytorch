# --------------------------------------------------------
# OICR
# Written by Jaedong Hwang
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
from model.oicr.oicr import _OICR
import pdb

class vgg16_oicr(_OICR):
  def __init__(self, classes, pretrained=False, class_agnostic=False, summary=None):
    self.model_path = 'data/pretrained_model/vgg16_caffe.pth'
    self.dout_base_model = 512
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic

    _OICR.__init__(self, classes, class_agnostic, tb=summary)

  def _init_modules(self):
    vgg = models.vgg16()
    if self.pretrained:
        print("Loading pretrained weights from %s" %(self.model_path))
        state_dict = torch.load(self.model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})

    vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
    # not using the last maxpool layer
    self.OICR_base = nn.Sequential(*list(vgg.features._modules.values())[:-1]) #  this is relu_5_3
    
    # Fix the layers before conv3:
    for layer in range(10):
      for p in self.OICR_base[layer].parameters(): p.requires_grad = False
    
    for i in range(24, len(self.OICR_base)):
        self.OICR_base[i-1] = self.OICR_base[i]
    self.OICR_base = self.OICR_base[:-1]

    for i in range(3) :
        self.OICR_base[23 + 2*i].dilation=(2,2)
        self.OICR_base[23 + 2*i].padding=(2,2)

    self.OICR_top = vgg.classifier

    # not using the last maxpool layer
    self.midn_score0 = nn.Linear(4096, self.n_classes)
    self.midn_score1 = nn.Linear(4096, self.n_classes)
    self.ic_score = nn.Linear(4096, self.n_classes+1)
    self.ic_score1 = nn.Linear(4096, self.n_classes+1)
    self.ic_score2 = nn.Linear(4096, self.n_classes+1)
    self.groups = self.get_parameter_groups()

  def _head_to_tail(self, pool5):
    pool5_flat = pool5.view(pool5.size(0), -1)
    fc7 = self.OICR_top(pool5_flat)
    return fc7

