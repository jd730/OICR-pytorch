# ---------------------------------------------------------
# OICR
# Written by Jaedong Hwang
# Licensed under The MIT License [see LICENSE for details]
# ---------------------------------------------------------

"""The layer used during training to get proposal labels for classifier refinement.

OICRLayer implements a Caffe Python layer.
"""
import numpy as np
import torch
import pdb
from model.utils.cython_bbox import bbox_overlaps

def OICRLayer(boxes, cls_prob, im_labels, cfg_TRAIN_FG_THRESH = 0.5):
    boxes = boxes[...,1:]
    proposals = _get_highest_score_proposals(boxes, cls_prob, im_labels)
    labels, rois, cls_loss_weights = _sample_rois(boxes, proposals, 21)
    return labels, cls_loss_weights

def _get_highest_score_proposals(boxes, cls_prob, im_labels):
    """Get proposals with highest score."""

    num_images, num_classes = im_labels.shape
    assert num_images == 1, 'batch size shoud be equal to 1'
    im_labels_tmp = im_labels[0, :]
    gt_boxes = np.zeros((0, 4), dtype=np.float32)
    gt_classes = np.zeros((0, 1), dtype=np.int32)
    gt_scores = np.zeros((0, 1), dtype=np.float32)
    
    if 21 == cls_prob.shape[2] : # added 1016
        cls_prob = cls_prob[:,:,1:]

    for i in range(num_classes):
        if im_labels_tmp[i] == 1:
            cls_prob_tmp = cls_prob[:,:, i].data
            max_index = np.argmax(cls_prob_tmp)
            gt_boxes = np.vstack((gt_boxes, boxes[:,max_index, :].reshape(1, -1))) 
            gt_classes = np.vstack((gt_classes, (i + 1) * np.ones((1, 1), dtype=np.int32))) # for pushing ground
            gt_scores = np.vstack((gt_scores, 
                cls_prob_tmp[:, max_index] ))  # * np.ones((1, 1), dtype=np.float32)))
            cls_prob[:, max_index, :] = 0 #in-place operation <- OICR code but I do not agree

    proposals = {'gt_boxes' : gt_boxes,
                 'gt_classes': gt_classes,
                 'gt_scores': gt_scores}

    return proposals

def _sample_rois(all_rois, proposals, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    gt_boxes = proposals['gt_boxes']
    gt_labels = proposals['gt_classes']
    gt_scores = proposals['gt_scores']
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[0], dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    try :
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
    except :
        pdb.set_trace()

    labels = gt_labels[gt_assignment, 0]
    cls_loss_weights = gt_scores[gt_assignment, 0]
    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= 0.5)[0]

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < 0.5)[0]

    labels[bg_inds] = 0
    real_labels = np.zeros((labels.shape[0], 21))
    for i in range(labels.shape[0]) :
        real_labels[i, labels[i]] = 1
    rois = all_rois
    return real_labels, rois, cls_loss_weights
