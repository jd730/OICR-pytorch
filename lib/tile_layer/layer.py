# --------------------------------------------------------
# Online Instance Classifier Refinement
# Copyright (c) 2016 HUST MCLAB
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng Tang
# Modified by Jaedong Hwang
# --------------------------------------------------------


"""The layer used during training to get proposal labels for classifier refinement.

OICRLayer implements a Caffe Python layer.
"""

import numpy as np
from model.utils.config import cfg
from model.utils.cython_bbox import bbox_overlaps
import torch
import torch.nn as nn
from torch.autograd import Function
import pdb

class TileLayer(nn.Module):
    """get proposal labels used for online instance classifier refinement."""

    def __init__(self) :
        super(TileLayer, self).__init__()


    @staticmethod
    def forward(cls,featsize) :
        n = featsize[0]
        if len(featsize) > 2 :
            w = featsize[2]
            h = featsize[3]
            ret = cls.unsqueeze(-1).unsqueeze(-1).repeat(n,1,w,h)
        else : 
            ret = cls.repeat(n,1)
        return ret

    @staticmethod
    def backward(grad_output):
        #output = torch.sum(torch.sum(torch.sum(grad_output, dim=0), dim=2), dim=3)
        output = (torch.sum(grad_output, dim=0))
        return output

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


