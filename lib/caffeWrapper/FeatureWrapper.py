# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import cPickle
import scipy.io as sio
import numpy as np
import cv2

import caffe
from utils.timer import Timer
from nms.nms_wrapper import apply_nms, apply_nms_mask_single
from mnc_config import cfg, get_output_dir
from utils.blob import prep_im_for_blob, im_list_to_blob, prep_im_for_blob_cfm, pred_rois_for_blob
from transform.bbox_transform import clip_boxes, bbox_transform_inv, filter_small_boxes
from transform.mask_transform import cpu_mask_voting, gpu_mask_voting


class FeatureWrapper(object):
    """
    A simple wrapper around Caffe's test forward
    """
    def __init__(self, test_prototxt, test_model, task_name):
        # Pre-processing, test whether model stored in binary file or npy files
        self.net = caffe.Net(test_prototxt, test_model, caffe.TEST)
        self.net.name = os.path.splitext(os.path.basename(test_model))[0]
        self.task_name = task_name
        
        #self.imdb = imdb
        self.output_dir = os.path.join('output', 'cfm', 'cub_200_2011', 'CPMC_segms_sp_approx_'+self.net.name)
        list_name = 'data/CUB_200_2011/ImageSets/Segmentation/traintest.txt'
        with open(list_name) as f:
            file_list = f.read().splitlines()
        # We define some class variables here to avoid defining them many times in every method
        self.images = file_list
        self.num_images = len(file_list)
        self.image_path = 'data/CUB_200_2011/images_nodirs/'

        # heuristic: keep an average of 40 detections per class per images prior to nms
        #self.max_per_set = 40 * self.num_images
        # heuristic: keep at most 100 detection per class per image prior to NMS
        #self.max_per_image = 100

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_result(self):
        output_dir = self.output_dir
        #det_file = os.path.join(output_dir, 'res_boxes.pkl')
        #seg_file = os.path.join(output_dir, 'res_masks.pkl')
        if self.task_name == 'det':
            pass
        elif self.task_name == 'vis_seg':
            pass
        elif self.task_name == 'seg':
            pass
        elif self.task_name == 'cfm':
            if os.path.exists(output_dir):
                self.get_cfm_result()
                #cfm_boxes, cfm_masks = self.get_cfm_result()
                #with open(det_file, 'wb') as f:
                #    cPickle.dump(cfm_boxes, f, cPickle.HIGHEST_PROTOCOL)
                #with open(seg_file, 'wb') as f:
                #    cPickle.dump(cfm_masks, f, cPickle.HIGHEST_PROTOCOL)
            #print 'Evaluating segmentation using convolutional feature masking'
            #self.imdb.evaluate_segmentation(cfm_boxes, cfm_masks, output_dir)
            print 'Finishing feature extraction using convolutional feature masking'
        else:
            print 'task name only support \'det\', \'seg\', \'cfm\' and \'vis_seg\''
            raise NotImplementedError

    def get_cfm_result(self):
        # detection threshold for each class
        # (this is adaptively set based on the max_per_set constraint)
        #thresh = -np.inf * np.ones(self.num_classes)
        # top_scores will hold one min heap of scores per class (used to enforce
        # the max_per_set constraint)
        #top_scores = [[] for _ in xrange(self.num_classes)]
        # all detections and segmentation are collected into a list:
        # Since the number of dets/segs are of variable size
        #all_boxes = [[[] for _ in xrange(self.num_images)]
        #             for _ in xrange(self.num_classes)]
        #all_masks = [[[] for _ in xrange(self.num_images)]
        #             for _ in xrange(self.num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        for i in xrange(self.num_images):
            _t['im_detect'].tic()
            feats = self.cfm_network_forward(i)
            _t['im_detect'].toc()
            print 'process image %d/%d, forward average time %f' % (i, self.num_images,
                                                                    _t['im_detect'].average_time)

        return

    def cfm_network_forward(self, im_i):
        im = cv2.imread(os.path.join(self.image_path, self.images[im_i]+'.jpg'))
        roidb_cache = os.path.join('data/cache/cub_200_2011_cpmc_sp_approx_maskdb/', self.images[im_i] + '.mat')
        roidb = sio.loadmat(roidb_cache)
        boxes = roidb['boxes']
        #filter_keep = filter_small_boxes(boxes, min_size=16)
        #boxes = boxes[filter_keep, :]
        masks = roidb['masks']
        #masks = masks[filter_keep, :, :]
        assert boxes.shape[0] == masks.shape[0]
        output_dir = self.output_dir

        # Resize input mask, make it the same as CFM's input size
        mask_resize = np.zeros((masks.shape[0], cfg.TEST.CFM_INPUT_MASK_SIZE, cfg.TEST.CFM_INPUT_MASK_SIZE))
        for i in xrange(masks.shape[0]):
            mask_resize[i, :, :] = cv2.resize(masks[i, :, :].astype(np.float),
                                              (cfg.TEST.CFM_INPUT_MASK_SIZE, cfg.TEST.CFM_INPUT_MASK_SIZE))
        masks = mask_resize

        # Get top-k proposals from MCG
        if cfg.TEST.USE_TOP_K_MCG:
            num_keep = min(boxes.shape[0], cfg.TEST.USE_TOP_K_MCG)
            boxes = boxes[:num_keep, :]
            masks = masks[:num_keep, :, :]
            assert boxes.shape[0] == masks.shape[0]
        # deal with multi-scale test
        # we group several adjacent scales to do forward
        _, im_scale_factors = prep_im_for_blob_cfm(im, cfg.TEST.SCALES)
        orig_boxes = boxes.copy()
        boxes = pred_rois_for_blob(boxes, im_scale_factors)
        num_scale_iter = int(np.ceil(len(cfg.TEST.SCALES) / float(cfg.TEST.GROUP_SCALE)))
        LO_SCALE = 0
        MAX_ROIS_GPU = cfg.TEST.MAX_ROIS_GPU
        # set up return results
        #res_boxes = np.zeros((0, 4), dtype=np.float32)
        #res_masks = np.zeros((0, 1, cfg.MASK_SIZE, cfg.MASK_SIZE), dtype=np.float32)
        #res_seg_scores = np.zeros((0, self.num_classes), dtype=np.float32)
        res_feats = np.zeros((0, 4096+4096), dtype=np.float32)

        for scale_iter in xrange(num_scale_iter):
            HI_SCALE = min(LO_SCALE + cfg.TEST.GROUP_SCALE, len(cfg.TEST.SCALES))
            inds_this_scale = np.where((boxes[:, 0] >= LO_SCALE) & (boxes[:, 0] < HI_SCALE))[0]
            assert (inds_this_scale - np.arange(boxes.shape[0])).sum() == 0
            if len(inds_this_scale) == 0:
                LO_SCALE += cfg.TEST.GROUP_SCALE
                continue
            max_rois_this_scale = MAX_ROIS_GPU[scale_iter]
            boxes_this_scale = boxes[inds_this_scale, :]
            masks_this_scale = masks[inds_this_scale, :, :]
            num_iter_this_scale = int(np.ceil(boxes_this_scale.shape[0] / float(max_rois_this_scale)))
            # make the batch index of input box start from 0
            boxes_this_scale[:, 0] -= min(boxes_this_scale[:, 0])
            # re-prepare im blob for this_scale
            input_blobs = {}
            input_blobs['data'], _ = prep_im_for_blob_cfm(im, cfg.TEST.SCALES[LO_SCALE:HI_SCALE])
            input_blobs['data'] = input_blobs['data'].astype(np.float32, copy=False)
            input_start = 0
            for test_iter in xrange(num_iter_this_scale):
                input_end = min(input_start + max_rois_this_scale, boxes_this_scale.shape[0])
                input_box = boxes_this_scale[input_start:input_end, :]
                input_mask = masks_this_scale[input_start:input_end, :, :]
                input_blobs['rois'] = input_box.astype(np.float32, copy=False)
                input_blobs['masks'] = input_mask.reshape(input_box.shape[0], 1,
                                                    cfg.TEST.CFM_INPUT_MASK_SIZE, cfg.TEST.CFM_INPUT_MASK_SIZE
                                                    ).astype(np.float32, copy=False)
                input_blobs['masks'] = (input_blobs['masks'] >= cfg.BINARIZE_THRESH).astype(np.float32, copy=False)
                self.net.blobs['data'].reshape(*input_blobs['data'].shape)
                self.net.blobs['rois'].reshape(*input_blobs['rois'].shape)
                self.net.blobs['masks'].reshape(*input_blobs['masks'].shape)
                blobs_out = self.net.forward(**input_blobs)
                
                #output_mask = blobs_out['mask_prob'].copy()
                #output_score = blobs_out['seg_cls_prob'].copy()
                output_feat = blobs_out['join_box_mask'].copy()
                
                #res_masks = np.vstack((res_masks,
                #                       output_mask.reshape(
                #                           input_box.shape[0], 1, cfg.MASK_SIZE, cfg.MASK_SIZE
                #                       ).astype(np.float32, copy=False)))
                #res_seg_scores = np.vstack((res_seg_scores, output_score))
                res_feats = np.vstack((res_feats, output_feat))
                
                input_start += max_rois_this_scale
            
            #res_boxes = np.vstack((res_boxes, orig_boxes[inds_this_scale, :]))
            LO_SCALE += cfg.TEST.GROUP_SCALE

        res = {
            'D': np.transpose(res_feats)
        }
        sio.savemat(os.path.join(output_dir, self.images[im_i]+'.m'), res)

        return res_feats
