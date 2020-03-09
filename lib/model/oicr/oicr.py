# --------------------------------------------------------
# OICR
# Written by Jaedong Hwang
# Licnesed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from oicr_layer.layer import OICRLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

def multi_class_cross_entropy_loss(preds, labels, eps=1e-6):
    cls_loss = labels * torch.log(preds +eps) + (1-labels) * torch.log(1-preds +eps)
    summed_cls_loss = torch.sum(cls_loss, dim=1)
    loss = -torch.mean(summed_cls_loss, dim=0)
    if torch.isnan(loss.sum()) :
        pdb.set_trace()
    return loss

def check_nan (var) :
    return torch.isnan(var) > 0 


def WeightedSoftmaxWithLoss(prob, labels_ic, cls_loss_weights, eps = 1e-6):
    loss = (labels_ic * torch.log(prob + eps))
    loss = loss.sum(dim=2)
    loss = -cls_loss_weights * loss 
    ret = loss.sum() / loss.numel()
    if torch.isnan(ret.sum()) :
        pdb.set_trace()
    return ret


class _OICR(nn.Module):
    """ OICR """
    def __init__(self, classes, class_agnostic, tb=None):
        super(_OICR, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
            
        self.param_groups = [[], [], [], []]
        self.OICR_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/8.0)
        self.OICR_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/8.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.OICR_roi_crop = _RoICrop()
        self.ic_layers = []
        self.tb = tb

    def forward(self, im_data, rois, labels=None, num_boxes=None):
        batch_size = im_data.size(0)
        num_rois = rois.size(0)
        init_rois = rois.data
        if self.training : # for multi-GPU
            try :
                nb = int(num_boxes[:,0].item())
            except :
                nb=int(num_boxes.item())
            num_boxes = num_boxes.data
            ret_prob = rois.new().new_zeros(1,rois.size(1),21)
            rois = rois[:,:nb]
            axis1 = int(num_boxes[:,1].item())
            axis2 = int(num_boxes[:,2].item())
            im_data = im_data[:,:,:axis1, :axis2]
            num_boxes = nb
            # feed image data to base model to obtain base feature map
        else :
            num_boxes = num_rois
        base_feat = self.OICR_base(im_data)
        rois = Variable(rois)
        # do roi pooling based on predicted rois

        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.OICR_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':
            pooled_feat = self.OICR_roi_align(base_feat, rois.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.OICR_roi_pool(base_feat, rois.view(-1,5))
        
        # feed pooled features to top model
        fc7= self._head_to_tail(pooled_feat) # fc7
        
        ic_score = self.ic_score(fc7).view(batch_size,num_boxes,self.n_classes+1)
        ic_score1 = self.ic_score1(fc7).view(batch_size,num_boxes,self.n_classes+1)
        ic_score2 = self.ic_score2(fc7).view(batch_size,num_boxes,self.n_classes+1)
        self.ic_prob = F.softmax(ic_score, dim=2)
        self.ic_prob1 = F.softmax(ic_score1, dim=2)
        self.ic_prob2 = F.softmax(ic_score2, dim=2)
        loss_midn=loss_oicr=loss_oicr1=loss_oicr2=0
        
        self.midn_prob0 = self.midn_score0(fc7).view(batch_size, num_boxes, self.n_classes)
        self.midn_prob1 = self.midn_score1(fc7).view(batch_size, num_boxes, self.n_classes)
        self.midn_prob0 = F.softmax(self.midn_prob0,dim=1) # rois
        self.midn_prob1 = F.softmax(self.midn_prob1,dim=2) # class
        self.midn_prob = self.midn_prob0 * self.midn_prob1
        if self.training:
            labels = labels.data
            if torch.isnan(fc7).sum() > 0 or torch.isnan(self.midn_score0.weight.data).sum() > 0 :
                pdb.set_trace()
            
            self.global_pool = self.midn_prob.sum(dim=1, keepdim=True)
            self.global_pool = self.global_pool.view(batch_size, self.n_classes)
            loss_midn = multi_class_cross_entropy_loss(self.global_pool, labels)
            
            label_ic, cls_loss_weights = OICRLayer(rois, self.midn_prob.clone(), labels)
            label_ic1, cls_loss_weights1 = OICRLayer(rois, self.ic_prob.clone(), labels)
            label_ic2, cls_loss_weights2 = OICRLayer(rois, self.ic_prob1.clone(), labels)
            
            if torch.isnan(self.ic_prob).sum().data > 0 or torch.isnan(self.ic_prob1).sum().data > 0 or torch.isnan(self.ic_prob2).sum().data >0 :
                pdb.set_trace()

            label_ic = torch.FloatTensor(label_ic).cuda().detach()
            label_ic1 = torch.FloatTensor(label_ic1).cuda().detach()
            label_ic2 = torch.FloatTensor(label_ic2).cuda().detach()
            cls_loss_weights = torch.tensor(cls_loss_weights).cuda().detach()
            cls_loss_weights1 = torch.tensor(cls_loss_weights1).cuda().detach()
            cls_loss_weights2 = torch.tensor(cls_loss_weights2).cuda().detach()
            
            loss_oicr = WeightedSoftmaxWithLoss(self.ic_prob, label_ic, cls_loss_weights)
            loss_oicr1 = WeightedSoftmaxWithLoss(self.ic_prob1, label_ic1, cls_loss_weights1)
            loss_oicr2 = WeightedSoftmaxWithLoss(self.ic_prob2, label_ic2, cls_loss_weights2)
           
            oicr_loss = loss_oicr + loss_oicr1 + loss_oicr2
            ret_prob[:,:nb] = (self.ic_prob + self.ic_prob1 + self.ic_prob2) / 3
            return init_rois, loss_midn.view(1), loss_oicr.view(1), loss_oicr1.view(1), loss_oicr2.view(1), ret_prob
        else :
            return self.ic_prob, self.ic_prob1, self.ic_prob2

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        self.from_scratch_layers = []
        
        for name, parameter in self.named_parameters() :
            if parameter.requires_grad :
                if 'weight' in name :
                    if 'ic' in name :
                        groups[2].append(parameter)
                    else :
                        groups[0].append(parameter)

                else : # bias
                    if 'ic' in name :
                        groups[3].append(parameter)
                    else :
                        groups[1].append(parameter)
        return groups

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.midn_score0, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.midn_score1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.ic_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.ic_score1, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.ic_score2, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
