# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader, collate_fn
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
       adjust_learning_rate, save_checkpoint, clip_gradient
from model.utils.net_utils import save_net, load_net, vis_detections
from model.oicr.vgg16_oicr import vgg16_oicr
import model.utils.logger as logger
#import tensorflow as tf
import cv2
from scipy.misc import imsave
from model.nms.nms_wrapper import nms
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                       help='training dataset',
                       default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                     help='vgg16, vggm',
                     default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                       help='starting epoch',
                       default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                       help='number of epochs to train',
                       default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                       help='number of iterations to display',
                       default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                       help='number of iterations to display',
                       default=1000, type=int)
    parser.add_argument('--save_dir', dest='save_dir',
                       help='directory to save models', default="models",
                       type=str)
    parser.add_argument('--nw', dest='num_workers',
                       help='number of worker to load data',
                       default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                       help='whether use CUDA',
                       action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                       help='whether use large imag scale',
                       action='store_true')                      
    parser.add_argument('--mGPUs', dest='mGPUs',
                       help='whether use multiple GPUs',
                       action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                       help='batch_size',
                       default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                       help='whether perform class_agnostic bbox regression',
                       action='store_true')

    parser.add_argument('--model', default='oicr', type=str)
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                       help='training optimizer',
                       default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                       help='starting learning rate',
                       default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                       help='step to do learning rate decay, unit is epoch',
                       default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                       help='learning rate decay ratio',
                       default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                       help='training session',
                       default=1, type=int)

    # resume trained model
    parser.add_argument('--checkpoint', dest='checkpoint',
                       help='checkpoint to load model',
                       default=-1, type=int)
    # log and diaplay
    parser.add_argument('--use_tb', dest='use_tb',
                       help='whether use tensorboard',
                       action='store_true')
    parser.add_argument('--load_dir', dest='load_dir', type=str, default=None)
    parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
    parser.add_argument('--threshold',type=float, default=0.01)
    args = parser.parse_args()
    return args

class Summary(object):
    def __init__(self, sess, path):
        self.sess = sess
        self.placeholders = {}
        self.kvs = {}
        self.merged= None
        self.writer = tf.summary.FileWriter(path)

    def add_value(self, key, val):
        if key not in self.kvs :
            var = tf.Variable(0.)
            self.placeholders[key] = tf.placeholder("float")
            summary_var = var.assign(self.placeholders[key])
            tf.summary.scalar(key, summary_var)
        self.kvs[key] = val
    
    def add_hist (self, name, grad) :
        tf.summary.histogram(name,grad)
        
    def add_sess(self,sess) :
        self.sess =sess
        
    def run(self, step) :
        if self.merged is None:
            self.merged = tf.summary.merge_all()
        s = {self.placeholders[k] : self.kvs[k] for k in self.kvs.keys() }
        ret = self.sess.run(self.merged,s)
        self.writer.add_summary(ret, step)


class sampler(Sampler):
    def __init__(self, train_size, batch_size):
        self.num_data = train_size
        self.num_per_batch = int(train_size / batch_size)
        self.batch_size = batch_size
        self.range = torch.arange(0,batch_size).view(1, batch_size).long()
        self.leftover_flag = False
        if train_size % batch_size:
            self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
            self.leftover_flag = True

    def __iter__(self):
        rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
        self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

        self.rand_num_view = self.rand_num.view(-1)

        if self.leftover_flag:
            self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

        return iter(self.rand_num_view)

    def __len__(self):
        return self.num_data

########### MAIN ######################
if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)
    if args.dataset == "pascal_voc":
        args.imdb_name = "voc_2007_trainval"
        args.imdbval_name = "voc_2007_trainval"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "pascal_voc_0712":
        args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
        args.imdbval_name = "voc_2007_test"
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2014_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
    elif args.dataset == "imagenet":
        args.imdb_name = "imagenet_train"
        args.imdbval_name = "imagenet_val"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
    elif args.dataset == "vg":
        # train sizes: train, smalltrain, minitrain
        # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
        args.imdb_name = "vg_150-50-50_minitrain"
        args.imdbval_name = "vg_150-50-50_minival"
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

    args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    vis = args.vis
    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    #torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda
    print(cfg.TRAIN.PROPOSAL_METHOD)

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
    train_size = len(roidb)
    ma = -1
    mi = 1e10
    for r in imdb.roidb :
        l = len(r['boxes'])
        if mi > l :
            mi = l
        if ma < l :
            ma = l
    print(ma,mi)
    print('{:d} roidb entries'.format(len(roidb)))

    output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if args.load_dir is None :
        load_dir = output_dir
    else :
        load_dir = args.load_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    sampler_batch = sampler(train_size, args.batch_size)
    dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size,\
                            imdb.num_classes, training=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size= args.batch_size,
                            sampler=sampler_batch, num_workers=args.num_workers,
                            collate_fn=collate_fn) # collate_fn is for multi-GPU

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_rois = torch.FloatTensor(1)
    labels = torch.FloatTensor(1)
    num_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_rois = im_rois.cuda()
        labels = labels.cuda()
        num_boxes = num_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_rois = Variable(im_rois)
    labels = Variable(labels)
    num_boxes = Variable(num_boxes)

    if args.cuda:
        cfg.CUDA = True
    
    tb = None
    summary=None
    log_dir = os.path.join(args.save_dir,'log')
#    if os.path.exists(log_dir) :
#        print('{} directory already exists'.format(log_dir))
#        for i in range(2,100) :
#            if not os.path.exists(log_dir + '_' + str(i)) :
#                log_dir = log_dir + '_' + str(i)
    logger.configure(dir=log_dir)

    if args.use_tb :
        num_cpu = 1
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=num_cpu,
                intra_op_parallelism_threads=num_cpu, gpu_options=tf.GPUOptions(
                    per_process_gpu_memory_fraction=0.005))
        sess = tf.Session(config=tf_config)
        summary = Summary(sess, log_dir)
        tb = logger.Logger(log_dir, output_formats=[logger.TensorBoardOutputFormat(log_dir)])
    

   # initilize the network here.
    print(args.model)
    if args.model == 'oicr' :
        OICR = vgg16_oicr(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic, summary=summary)
    else : 
        raise Exception("Model does not exist")

    OICR.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr
    #tr_momentum = cfg.TRAIN.MOMENTUM
    #tr_momentum = args.momentum

    weight_decay = cfg.TRAIN.WEIGHT_DECAY

    param_groups = OICR.groups
    if args.cuda:
        OICR.cuda()
    params = [{'params': param_groups[0], 'lr': lr, 'weight_decay': weight_decay},
        {'params': param_groups[1], 'lr': 2*lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*lr, 'weight_decay': weight_decay},
        {'params': param_groups[3], 'lr': 20*lr, 'weight_decay': 0}
    ]
    if args.optimizer == "adam":
        lr = lr * 0.1
        optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd": # 0.001 ~ 40k decrease to 0.0001 ~ + 30k
        optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

    if args.checkpoint >= 0:
        load_name = os.path.join(load_dir,'{:06d}.pth'.format(args.checkpoint))
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda  storage, loc:storage)
        args.session = checkpoint['session']
        args.start_epoch = checkpoint['epoch']
        OICR.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        print("loaded checkpoint %s" % (load_name))
        
        total_step = int(args.checkpoint)
    else : 
        total_step = 0

    print("Learning rate {}".format(lr))

    if args.mGPUs:
        OICR = nn.DataParallel(OICR)
        OICR.cuda()

    iters_per_epoch = int(train_size / args.batch_size)

    grads = {} # for recording gradients
    def save_grad(name):
        def hook(grad):
            grads[name] = grad
            return hook
    def cycle (it) :
        while True :
            for x in it :
                yield x

    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        OICR.train()
        loss_temp = 0
        loss_midn_temp = 0
        loss_oicr_temp = 0
        loss_oc1_temp = 0
        loss_oc2_temp = 0
        loss_oc3_temp = 0
        start = time.time()
        if args.mGPUs :
            groups = OICR.module.groups
        else :
            groups = OICR.groups
        
        if args.batch_size == 1 :
            digit = 2
        else :
            digit = 1
        #data_iter = iter(dataloader)
        data_iter = iter(cycle(dataloader))
        loading_time = 0 
        for step in range(iters_per_epoch):
            OICR.zero_grad()
            optimizer.zero_grad()
            loss = 0
            if total_step == 4e4:
                adjust_learning_rate(optimizer, args.lr_decay_gamma)
                lr *= args.lr_decay_gamma
            # for the case of sequential batch, the below should be for _ in range(args.batch_size)
            for _ in range(2): # to follow caffe work
                jaed = time.time()
                data = next(data_iter)
                loading_time += (time.time() - jaed)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_rois.data.resize_(data[1].size()).copy_(data[1])
                labels.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])
                labels = torch.squeeze(labels).view(im_data.size(0), labels.size(-1))
                rois, midn_loss, oc1, oc2, oc3, cls_prob= OICR(im_data, im_rois, labels, num_boxes)
                oicr_loss = oc1 + oc2 + oc3
                rois_label= cls_prob
                loss_total = midn_loss + oicr_loss
                midn_loss = midn_loss.mean()
                oicr_loss = oicr_loss.mean()
                oc1 = oc1.mean()
                oc2 = oc2.mean()
                oc3 = oc3.mean()
                loss = midn_loss + oicr_loss
                loss_temp += loss.item()
                loss_midn_temp += midn_loss.item()
                loss_oicr_temp += oicr_loss.item()
                loss_oc1_temp += oc1.item()
                loss_oc2_temp += oc2.item()
                loss_oc3_temp += oc3.item()
                # backward
                #loss /= 2 #digit #args.batch_size  # see https://discuss.pytorch.org/t/pytorch-gradients/884
                loss.backward(retain_graph=True)
            total_step +=1
            # batch end
            if args.net == "vgg16":
                clip_gradient(OICR, 10.)
            optimizer.step()

            if total_step % args.disp_interval == 0:
            
                end = time.time()
                if step > 0:
                    loss_temp /= (args.disp_interval + 1)
                    loss_midn_temp /= (args.disp_interval + 1)
                    loss_oicr_temp /= (args.disp_interval + 1)
                    loss_oc1_temp /= (args.disp_interval + 1)
                    loss_oc2_temp /= (args.disp_interval + 1)
                    loss_oc3_temp /= (args.disp_interval + 1)

                if args.mGPUs:
                    loss_midn = midn_loss.mean().item()
                    loss_oicr = oicr_loss.mean().item()
                    loss_oicr1 = oc1.mean().item()
                    loss_oicr2 = oc2.mean().item()
                    loss_oicr3 = oc3.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    record_module = OICR.module # parallel encapsulate OICR
                else:
                    loss_midn = midn_loss.item()
                    loss_oicr = oicr_loss.item()
                    loss_oicr1 = oc1.item()
                    loss_oicr2 = oc2.item()
                    loss_oicr3 = oc3.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    record_module = OICR
                
                a = torch.max(im_rois[:,:,1]).data
                b = torch.max(im_rois[:,:,2]).data
                c = torch.max(im_rois[:,:,3]).data
                d = torch.max(im_rois[:,:,4]).data

                logger.log("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
                                 % (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
                logger.log("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                #print("\t\t\tmidn : %.4f, oicr : %.4f" % (loss_midn, loss_oicr))
                logger.log("\t\t\tmidn : %.4f, oicr : %.4f" % (loss_midn_temp, loss_oicr_temp))
                logger.log("\t\t\tdata loading : %.4f" %(loading_time))
                loading_time = 0
                print(c,d, im_data.shape)
                logger.log("Logging to {}".format(log_dir))
                # end batch
                
                # logging 
                logger.record_tabular('loss', loss_temp)
                logger.record_tabular('midn_loss', loss_midn_temp)
                logger.record_tabular('oicr_loss', loss_oicr_temp)
                logger.record_tabular('oicr_loss1', loss_oc1_temp)
                logger.record_tabular('oicr_loss2', loss_oc2_temp)
                logger.record_tabular('oicr_loss3', loss_oc3_temp)
                logger.record_tabular('step',total_step)
                logger.record_tabular('layer_norm/ic_score',record_module.ic_score.weight.data.norm().item())
                logger.record_tabular('layer_norm/ic_score0',record_module.ic_score1.weight.data.norm().item())
                logger.record_tabular('layer_norm/ic_score1',record_module.ic_score2.weight.data.norm().item())
                logger.record_tabular('layer_norm/midn_score0',record_module.midn_score0.weight.data.norm().item())
                logger.record_tabular('layer_norm/midn_score1',record_module.midn_score1.weight.data.norm().item())
                logger.dump_tabular()

                if args.use_tb:
                    try : 
                        summary.add_value('loss/loss', loss_temp)
                        summary.add_value('loss/midn_loss', loss_midn_temp)
                        summary.add_value('loss/oicr_loss', loss_oicr_temp)
                        summary.add_value('step',total_step)
                        
                        summary.add_value('layer_norm/ic_score',record_module.ic_score.weight.data.norm())
                        summary.add_value('layer_norm/ic_score0',record_module.ic_score1.weight.data.norm())
                        summary.add_value('layer_norm/ic_score1',record_module.ic_score2.weight.data.norm())
                        summary.add_value('layer_norm/midn_score0',record_module.midn_score0.weight.data.norm())
                        summary.add_value('layer_norm/midn_score1',record_module.midn_score1.weight.data.norm())
                        
                        summary.add_hist('layer/ic_score',record_module.ic_score.weight.data)
                        summary.add_hist('layer/ic_score',record_module.ic_score1.weight.data)
                        summary.add_hist('layer/ic_score',record_module.ic_score2.weight.data)
                        summary.add_hist('layer/midn_score0',record_module.midn_score0.weight.data)
                        summary.add_hist('layer/midn_score1',record_module.midn_score1.weight.data)
                        
                        # gradients
                        targets = {'top','ic','midn','base'}
                        for name, parameter in OICR.named_parameters() :
                            if parameter.grad is None :
                                continue
                            for t in targets :
                                if t in name : 
                                    summary.add_hist('{}_grad/{}'.format(t,name), parameter.grad)
                                    summary.add_value('{}_grad/{}'.format(t,name), parameter.grad.data.norm())
                        summary.run(total_step)
                    except :
                        print("TensorBoard problem")
                        summary = Summary(sess, log_dir)
                loss_temp = 0
                loss_midn_temp = 0
                loss_oicr_temp = 0
                start = time.time()

                    

            if total_step % args.checkpoint_interval == 0:
                # Out of step
                save_name = os.path.join(output_dir, '{:06d}.pth'.format(total_step))
                save_checkpoint({
                    'session': args.session,
                    'epoch': epoch + 1,
                    'model': OICR.module.state_dict() if args.mGPUs else OICR.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'pooling_mode': cfg.POOLING_MODE,
                    'class_agnostic': args.class_agnostic,
                }, save_name)
                logger.log('save model: {}'.format(save_name))
                if vis :
                    im2show = np.copy(im_data[0]) + cfg.PIXEL_MEANS.reshape((3,1,1))
                    print(im2show.shape)
                    #im2show = np.ascontiguousarray(np.transpose(im2show,(1,2,0))[...,::-1])
                    im2show = np.ascontiguousarray(np.transpose(im2show,(1,2,0)))
                    scores = rois_label
                    scores = scores.data
                    boxes = im_rois.data[:, :, 1:5]
                    pred_boxes = np.tile(boxes , (1, scores.shape[2]))
                    pred_boxes = torch.from_numpy(pred_boxes).cuda()
                    scores = scores[0].squeeze()
                    pred_boxes = pred_boxes[0].squeeze()
                    thresh = args.threshold
                    for j in range(1,imdb.num_classes+1): # changed
                        inds = torch.nonzero(scores[:,j]>thresh).view(-1)
                        # if there is det
                        if inds.numel() > 0:
                            cls_scores = scores[:,j][inds]
                            _, order = torch.sort(cls_scores, 0, True)
                            if args.class_agnostic:
                                cls_boxes = pred_boxes[inds, :]
                            else:
                                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1) # ERROR cls_boxes is not tensor)
                            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                            cls_dets = cls_dets[order]
                            keep = nms(cls_dets, cfg.TEST.NMS)
                            cls_dets = cls_dets[keep.view(-1).long()]
                            im2show = vis_detections(im2show, imdb.classes[j-1], cls_dets.cpu().numpy(), thresh ) # changed
                            # this part makes the output label shift so I do not need to worry about the result
                    if not os.path.exists(os.path.join(args.save_dir, 'images')) :
                        os.mkdir(os.path.join(args.save_dir, 'images'))
                    img_name = os.path.join(args.save_dir,'images/{:06d}.png'.format(total_step))
                    re = cv2.imwrite(img_name, im2show)
                    logger.log('image save : {}'.format(img_name))
