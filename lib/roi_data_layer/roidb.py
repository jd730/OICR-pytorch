# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
#
# Modified by Peng Tang for Online Instance Classifier Refinement
# Modified by Jaedong Hwang for OICR
# --------------------------------------------------------

"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""

import numpy as np
from model.utils.config import cfg
import model.utils.cython_bbox
from datasets.factory import get_imdb
import datasets


def prepare_roidb(imdb):
    """Enrich the imdb's roidb by adding some derived quantities that
    are useful for training. This function precomputes the maximum
    overlap, taken over ground-truth boxes, between each ROI and
    each ground-truth box. The class with maximum overlap is also
    recorded.
    """
    roidb = imdb.roidb
    for i in range(len(imdb.image_index)):
        roidb[i]['image'] = imdb.image_path_at(i)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb



def combined_roidb(imdb_names, training=True):
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')
    print('Preparing training data...')

    prepare_roidb(imdb)
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb
  
  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)
    return roidb
  
  roidbs = [get_roidb(s) for s in imdb_names.split('+')]
  roidb = roidbs[0]

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)

  if training:
    roidb = filter_roidb(roidb)

  #ratio_list, ratio_index = rank_roidb_ratio(roidb)
  
  return imdb, roidb, 0,0#ratio_list, ratio_index
