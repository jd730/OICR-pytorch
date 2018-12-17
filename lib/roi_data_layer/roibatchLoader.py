# OICR
# Licensed under The MIT License [see LICENSE for details]
# Written by Jaedong Hwang
# ----------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from model.utils.config import cfg
from roi_data_layer.minibatch import get_minibatch, get_minibatch

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):
    def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.TRAIN.PROPOSAL_LIMIT #cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.max_image_size = cfg.TRAIN.MAX_IMAGE_SIZE
        self.max_rois_size = cfg.TRAIN.MAX_ROIS_SIZE


    def __getitem__(self, index):
    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
        minibatch_db =  [self._roidb[index]] # [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        np.random.shuffle(blobs['rois'])
        rois = torch.from_numpy(blobs['rois'][:self.max_rois_size])
        data = torch.from_numpy(blobs['data'])
        labels = torch.from_numpy(blobs['labels'])
        data_height, data_width = data.size(1), data.size(2)
        
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)

        info = torch.Tensor([rois.size(0), data_height, data_width])
    
        return data, rois, labels, info


    def __len__(self):
        return len(self._roidb)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
        We should build custom collate_fn rather than using default collate_fn, 
        because merging caption (including padding) is not supported in default.
        Args:
        data: list of tuple (image, rois, labels, info). 
        - image: torch tensor of shape (3, axis1, axis2).
        - rois: (N,5)
        - labels: torch tensor of shape (?); variable length.
        - info: the information of N, axis1, axis2
        Returns:
        - im_data: torch tensor of shape (batch_size, 3, max_axis1, max_axis2).
        - im_rois: torch tensor of shape (batch_size, max_N, 5).
        - labels: torch tensor of shape (batch_size, ?)
        - info: torch tensor of shape (batch_size,5)
    """
    # Sort a data list by caption length (descending order).
    img, rois, labels, info = zip(*data)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    labels = torch.stack(labels, 0)
    info = torch.stack(info, 0)
    
    max_rois_len = min(int(torch.max(info[:,0])), cfg.TRAIN.MAX_ROIS_SIZE)
    
    axis1 = int(torch.max(info[:,1]))
    axis2 = int(torch.max(info[:,2]))
    im_rois = rois[0].new().new_zeros(len(rois),max_rois_len,5)
    im_data = img[0].new().new_zeros(len(rois), 3, axis1, axis2)
    
    for i, cap in enumerate(data):
        im_rois[i, :int(info[i,0].item())] = rois[i]
        im_data[i, :, :int(info[i,1].item()), :int(info[i,2].item())] = img[i]
    return im_data, im_rois, labels, info



