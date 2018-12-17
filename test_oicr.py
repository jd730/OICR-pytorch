# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# Modified by Jaedong Hwang for implementing OICR
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.nms.nms_wrapper import nms
from model.utils.net_utils import save_net, load_net, vis_detections, vis_gts
from model.oicr.vgg16_oicr import vgg16_oicr
#from model.oicr.test import test_net

# test.py
from model.utils.timer import Timer
import scipy.io as sio
#from model.utils.cython_nms import nms
import heapq
from model.utils.blob import im_list_to_blob

from scipy.misc import imread

try:
    range          # Python 2
except NameError:
    range = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='vgg16', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='directory to load models', default="test",
                      type=str)
    parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
    parser.add_argument('--model', default='oicr', type=str)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--threshold',type=float, default=0.5)
    args = parser.parse_args()
    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image in BGR order
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
        im_shapes: the list of image shapes
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_size_max = np.max(im_shape[0:2])
    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im_list_to_blob([im]))

    blob = processed_ims
    return blob, np.array(im_scale_factors)

def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob
    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid
    """
    rois_blob_real = []

    for i in range(len(im_scale_factors)):
        rois, levels = _project_im_rois(im_rois, np.array([im_scale_factors[i]]))
        rois_blob = np.hstack((levels, rois))
        rois_blob_real.append(rois_blob.astype(np.float32, copy=False))

    return rois_blob_real

def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.
    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob
    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (list): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1
        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels

def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {'data' : None, 'rois' : None}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    
    return blobs, im_scale_factors

def im_detect(net, im, boxes):
    """Detect object classes in an image given object proposals.

    Arguments:
        net (caffe.Net): Fast R-CNN network to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals

    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """
    blobs, unused_im_scale_factors = _get_blobs(im, boxes)
    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    for i in range(len(blobs['data'])):
        if cfg.DEDUP_BOXES > 0:
            v = np.array([1, 1e3, 1e6, 1e9, 1e12])
            hashes = np.round(blobs['rois'][i] * cfg.DEDUP_BOXES).dot(v)
            _, index, inv_index = np.unique(hashes, return_index=True,
                                            return_inverse=True)
            blobs['rois'][i] = blobs['rois'][i][index, :]
            boxes_tmp = boxes[index, :].copy()
        else:
            boxes_tmp = boxes.copy()
        t_data = blobs['data'][i].astype(np.float32, copy=False)
        #t_data = t_data.reshape((1, t_data.shape[0], t_data.shape[1], t_data.shape[2], t_data.shape[3]))
        data_height, data_width = t_data.shape[1], t_data.shape[2]
        im_data = torch.FloatTensor(t_data).cuda()
        im_data = im_data.permute(0, 3, 1, 2).contiguous() #.view(3, data_height, data_width)
        LIM = 2000 # split ROIs due to memory issue
        if cfg.TEST.USE_FLIPPED :
            blobs['data'][i] = blobs['data'][i][:, :,  ::-1, :]
            width = blobs['data'][i].shape[3]
            t_data = blobs['data'][i].astype(np.float32, copy=False)
            data_height, data_width = t_data.shape[1], t_data.shape[2]
            #im_data = torch.FloatTensor(t_data).cuda()
            im_data_flip = torch.from_numpy(t_data.copy()).cuda()
            im_data_flip = im_data_flip.permute(0, 3, 1, 2).contiguous()#.view(3, data_height, data_width)
            #im_data = im_data[...,::-1]
        for j in range (int(np.ceil(blobs['rois'][i].shape[0] / LIM))) :
            t_rois = blobs['rois'][i][j*LIM:(j+1)*LIM].astype(np.float32, copy=False)
            im_rois = torch.FloatTensor(t_rois).cuda()
            ic_prob, ic_prob1, ic_prob2 = net(im_data, im_rois)
            scores_tmp = ic_prob + ic_prob1 + ic_prob2
            pred_boxes_small = np.tile(boxes_tmp[j*LIM : (j+1)*LIM], (1, scores_tmp.shape[2]))

            if cfg.TEST.USE_FLIPPED:
                #pdb.set_trace()
                oldx1 = blobs['rois'][i][j*LIM:(j+1)*LIM, 1].copy()
                oldx2 = blobs['rois'][i][j*LIM:(j+1)*LIM, 3].copy()
                blobs['rois'][i][j*LIM:(j+1)*LIM, 1] = width - oldx2 - 1
                blobs['rois'][i][j*LIM:(j+1)*LIM, 3] = width - oldx1 - 1
                assert (blobs['rois'][i][j*LIM:(j+1)*LIM, 3] >= blobs['rois'][i][j*LIM:(j+1)*LIM, 1]).all()
                t_rois = blobs['rois'][i][j*LIM:(j+1)*LIM].astype(np.float32, copy=False)
                im_rois = torch.FloatTensor(t_rois).cuda()
                ic_prob, ic_prob1, ic_prob2 = net(im_data_flip, im_rois)
                scores_tmp += ic_prob + ic_prob1 + ic_prob2
                del im_rois

            if j is 0 :
                scores_tmp_real = scores_tmp
                pred_boxes = pred_boxes_small
            else :
                scores_tmp_real = torch.cat((scores_tmp_real, scores_tmp), dim=1)
                pred_boxes = np.vstack((pred_boxes, pred_boxes_small))


        if cfg.DEDUP_BOXES > 0:
            # Map scores and predictions back to the original set of boxes
            scores_tmp = scores_tmp_real[:,inv_index, :]
            pred_boxes = pred_boxes[inv_index, :]
        
        if i == 0:     
            scores = np.copy(scores_tmp.data).squeeze()
            if len(scores.shape) == 1 :
                scores = scores[np.newaxis, :]
        else:
            scores += scores_tmp[0].data

    scores /= len(blobs['data']) * (1. + cfg.TEST.USE_FLIPPED)
    return scores[:,1:], pred_boxes[:, 4:]

def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            keep = nms(torch.FloatTensor(dets).cuda(), thresh)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes

if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    
    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    #cfg.TRAIN.USE_FLIPPED = False
    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))
    
    output_dir = os.path.join(args.load_dir, args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_dir_map = os.path.join(output_dir, 'map')
    output_dir_corloc = os.path.join(output_dir, 'corloc')
    output_dir_vis = os.path.join(output_dir, 'images')
    print(output_dir)
    if not os.path.exists(output_dir_map):
        os.makedirs(output_dir_map)
    if not os.path.exists(output_dir_corloc):
        os.makedirs(output_dir_corloc)
    if args.vis and not os.path.exists(output_dir_vis):
        os.makedirs(output_dir_vis)
    
    if args.restore :
        print('Evaluating detections')
        imdb.evaluate_detections(None, output_dir_map,args.restore)
        print('Evaluating CorLoc')
        imdb.evaluate_discovery(None, output_dir_corloc,args.restore)
        exit()

    input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(input_dir):
        raise Exception('There is no input directory for loading network from ' + input_dir)
    load_name = os.path.join(input_dir, '{:06d}.pth'.format(args.checkpoint))
    # initilize the network here.



    if args.model == 'oicr' :
        OICR = vgg16_oicr(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.model == 'oicr_cbp_v2' :
        from model.oicr_cbp_v2.vgg16_oicr_cbp import vgg16_oicr_cbp
        OICR = vgg16_oicr_cbp(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.model == 'oicr_cbp' :
        from model.oicr_cbp.vgg16_oicr_cbp import vgg16_oicr_cbp
        OICR = vgg16_oicr_cbp(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
    else : 
        raise Exception("Model does not exist")

    OICR.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    OICR.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']


    print('load model successfully!')

    if args.cuda:
        cfg.CUDA = True

    if args.cuda:
        OICR.cuda()

    start = time.time()
    OICR.eval()
    
    """Test an OICR network on an image database."""
    num_images =  len(imdb.image_index)
    # heuristic: keep an average of 40 detections per class per images prior
    # to NMS
    max_per_set = 40 * num_images
    # heuristic: keep at most 100 detection per class per image prior to NMS
    max_per_image = 100
    # detection thresold for each class (this is adaptively set based on the
    # max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # thresh = 0.1 * np.ones(imdb.num_classes)
    # top_scores will hold one minheap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in range(imdb.num_classes)]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)] 
                 for _ in range(imdb.num_classes)]
    all_boxes_corloc = [[[] for _ in range(num_images)] 
                 for _ in range(imdb.num_classes)]

    #output_dir = get_output_dir(imdb, 'oicr')
    
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}
    

    images_real = np.zeros((num_images,), dtype=object)
    gt = np.zeros((num_images, ), dtype=object)
    roidb = imdb.roidb

    scores_all = []
    boxes_all = []
    error_flag = False

    if args.mGPUs :
        OICR = nn.DataParallel(OICR)
        OICR.cuda()
    
    LIM = 20000 # split ROIs due to memory issue.

    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        
        for j in range (int(np.ceil(roidb[i]['boxes'].shape[0] / LIM))) :
            roi_temp = roidb[i]['boxes'][j*LIM:(j+1)*LIM]
            scores_temp, boxes_temp = im_detect(OICR, im, roi_temp)
            if j is 0 :
                scores = scores_temp
                boxes = boxes_temp
            else :
                scores = np.vstack((scores, scores_temp))
                boxes = np.vstack((boxes, boxes_temp))


        _t['im_detect'].toc()

        #scores_all.append(scores)
        #boxes_all.append(boxes)

        _t['misc'].tic()
        # mAP
        for j in range(0, imdb.num_classes):
            inds = np.where((scores[:, j] > thresh[j]))[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            top_inds = np.argsort(-cls_scores)[:max_per_image]
            cls_scores = cls_scores[top_inds]
            cls_boxes = cls_boxes[top_inds, :]
            # push new scores onto the minheap
            for val in cls_scores:
                heapq.heappush(top_scores[j], val)
            # if we've collected more than the max number of detection,
            # then pop items off the minheap and update the class threshold
            if len(top_scores[j]) > max_per_set:
                while len(top_scores[j]) > max_per_set:
                    heapq.heappop(top_scores[j])
                thresh[j] = top_scores[j][0]
            all_boxes[j][i] = \
                    np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                    .astype(np.float32, copy=False)
            
            if args.vis and i < 100: # save first 100 images.
                try :
                    keep = nms(torch.FloatTensor(all_boxes[j][i]).cuda(), 0.3)
                    cls_det = torch.FloatTensor(all_boxes[j][i]).cuda()
                    cls_det = cls_det[keep.view(-1).long()]
                    if j==0 :
                        im2show = vis_gts (im, imdb.image_path_at(i))
                    im2show = vis_detections(im, imdb.classes[j], all_boxes[j][i][keep.view(-1).long()],0)
                except : 
                    error_flag = True
                    pdb.set_trace()

        # CorLoc
            index = np.argmax(scores[:, j])
            all_boxes_corloc[j][i] = \
                np.hstack((boxes[index, j*4:(j+1)*4].reshape(1, -1), 
                           np.array([[scores[index, j]]])))
        
        if args.vis and not error_flag and i < 100:
            path = '{}/images/{:06d}.png'.format(output_dir, i)
            if i % 100 == 0 :
                print(path)
            cv2.imwrite(path, im2show)
        error_flag = False

        gt_tmp = {'aeroplane' : np.empty((0, 4), dtype=np.float32), 
                'bicycle' : np.empty((0, 4), dtype=np.float32), 
                'bird' : np.empty((0, 4), dtype=np.float32), 
                'boat' : np.empty((0, 4), dtype=np.float32), 
                'bottle' : np.empty((0, 4), dtype=np.float32), 
                'bus' : np.empty((0, 4), dtype=np.float32), 
                'car' : np.empty((0, 4), dtype=np.float32), 
                'cat' : np.empty((0, 4), dtype=np.float32), 
                'chair' : np.empty((0, 4), dtype=np.float32), 
                'cow' : np.empty((0, 4), dtype=np.float32), 
                'diningtable' : np.empty((0, 4), dtype=np.float32), 
                'dog' : np.empty((0, 4), dtype=np.float32), 
                'horse' : np.empty((0, 4), dtype=np.float32), 
                'motorbike' : np.empty((0, 4), dtype=np.float32), 
                'person' : np.empty((0, 4), dtype=np.float32), 
                'pottedplant' : np.empty((0, 4), dtype=np.float32), 
                'sheep' : np.empty((0, 4), dtype=np.float32), 
                'sofa' : np.empty((0, 4), dtype=np.float32), 
                'train' : np.empty((0, 4), dtype=np.float32), 
                'tvmonitor':np.empty((0, 4), dtype=np.float32)}
        tmp_idx = np.where(roidb[i]['labels'][:imdb.num_classes])[0] 
        
        for j in range(len(tmp_idx)):
            idx_real = np.argmax(scores[:, tmp_idx[j]])
            gt_tmp[imdb.classes[tmp_idx[j]]] = np.array([boxes[idx_real, tmp_idx[j]*4], 
                                                    boxes[idx_real, tmp_idx[j]*4], 
                                                    boxes[idx_real, tmp_idx[j]*4+3],
                                                    boxes[idx_real, tmp_idx[j]*4+2]],
                                                    dtype=np.float32)
#            gt_tmp[imdb.classes[1+tmp_idx[j]]] += 1
            gt_tmp[imdb.classes[tmp_idx[j]]] += 1

        gt[i] = {'gt' : gt_tmp}

        images_real[i] = imdb.image_index[i]

        _t['misc'].toc()

        #sys.stdout.write
        if i % 500 == 0 :
            print ('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
              .format(i + 1, num_images, _t['im_detect'].average_time, 
                _t['misc'].average_time))
    
        else :
            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
              .format(i + 1, num_images, _t['im_detect'].average_time, 
                _t['misc'].average_time))
            sys.stdout.flush()

    # Rethresholding
    for j in range(imdb.num_classes):
        for i in range(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]
    
    model_save_gt = {'images' : images_real, 'gt' : gt}
    sio.savemat('{}_gt.mat'.format(imdb.name), model_save_gt)
    
    det_file = os.path.join(output_dir_map, 'detections.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        
    det_file = os.path.join(output_dir_corloc, 'discovery.pkl')
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes_corloc, f, pickle.HIGHEST_PROTOCOL)

    # due to memory issue
#    det_file_all = os.path.join(output_dir, 'detections_all.pkl')
#    results_all = {'scores_all' : scores_all, 'boxes_all' : boxes_all}
#    with open(det_file_all, 'wb') as f:
#        pickle.dump(results_all, f, pickle.HIGHEST_PROTOCOL)

    print('Applying NMS to all detections')
    nms_dets = apply_nms(all_boxes, cfg.TEST.NMS)
    
    print('Evaluating detections')
    imdb.evaluate_detections(nms_dets, output_dir_map)
    
    print('Evaluating CorLoc')
    imdb.evaluate_discovery(all_boxes_corloc, output_dir_corloc)
