# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# System modules
import argparse
import os
import cPickle
import numpy as np
import scipy.io as sio
import cv2
from multiprocessing import Process
import time
import PIL
# User-defined module
import _init_paths
from mnc_config import cfg
from utils.cython_bbox import bbox_overlaps
from transform.mask_transform import mask_overlap, intersect_mask
from datasets.pascal_voc_seg import PascalVOCSeg


def parse_args():
    """ Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Prepare CPMC roidb')
    parser.add_argument('--input', dest='input_dir',
                        help='folder contain input mcg proposals',
                        default='data/CUB_200_2011/MySegmentsMat/ground_truth_sp_approx/', type=str)
    parser.add_argument('--output', dest='output_dir',
                        help='folder contain output roidb', required=True,
                        type=str)
    parser.add_argument('--gt_roi', dest='roidb', help='roidb',
                        default='data/cache/CUB_200_2011_train_gt_roidb.pkl', type=str)
    parser.add_argument('--gt_mask', dest='maskdb', help='maskdb',
                        default='data/cache/CUB_200_2011_train_gt_maskdb.pkl', type=str)
    parser.add_argument('-mask_sz', dest='mask_size',
                        help='compressed mask resolution',
                        default=21, type=int)
    parser.add_argument('--top_k', dest='top_k',
                        help='number of generated proposal',
                        default=-1, type=int)
    parser.add_argument('--db', dest='db_name',
                        help='train or validation',
                        default='train', type=str)
    parser.add_argument('--para_job', dest='para_job',
                        help='launch several process',
                        default='1', type=int)
    return parser.parse_args()


def process_roidb(file_start, file_end, db):

    for cnt in xrange(file_start, file_end):
        f = file_list[cnt]
        full_file = os.path.join(input_dir, f)
        if not os.path.exists(os.path.join(input_dir, f.split('.')[0] + '.mat')):
            print os.path.join(input_dir, f.split('.')[0] + '.mat')
            continue
        output_cache = os.path.join(output_dir, f.split('.')[0] + '.mat')
        timer_tic = time.time()
        if os.path.exists(output_cache):
            continue
        mcg_mat = sio.loadmat(full_file)
        #mcg_mask_label = mcg_mat['labels']
        #mcg_superpixels = mcg_mat['superpixels']
        #mcg_masks_ = mcg_mat['masks']     #im_height * im_width * num_proposal
        mcg_mask_label = mcg_mat['sp_app'] #num_sp * num_proposal
        mcg_superpixels = mcg_mat['sp']    #im_height * im_width
        num_proposal = mcg_mask_label.shape[1]

        num_proposal = 1
        mcg_boxes = np.zeros((num_proposal, 4))
        mcg_masks = np.zeros((num_proposal, mask_size, mask_size), dtype=np.bool)

        mcg_boxes[0, :] = np.array([0, 0, mcg_superpixels.shape[1]-1, mcg_superpixels.shape[0]-1])
        mcg_masks[0, :, :]= np.ones((mask_size, mask_size))

        if top_k != -1:
            mcg_boxes = mcg_boxes[:top_k, :]
            mcg_masks = mcg_masks[:top_k, :]

        if db == 'val':
            # if we prepare validation data, we only need its masks and boxes
            roidb = {
                'masks': (mcg_masks >= cfg.BINARIZE_THRESH).astype(bool),
                'boxes': mcg_boxes
            }
            sio.savemat(output_cache, roidb)
            use_time = time.time() - timer_tic
            print '%d/%d use time %f' % (cnt, len(file_list), use_time)

        else:

            continue

            # Otherwise we need to prepare other information like overlaps


if __name__ == '__main__':
    args = parse_args()
    input_dir = args.input_dir
    assert os.path.exists(input_dir), 'Path does not exist: {}'.format(input_dir)
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    mask_size = args.mask_size

    list_name = 'data/CUB_200_2011/ImageSets/Segmentation/train.txt' if args.db_name == 'train' else 'data/CUB_200_2011/ImageSets/Segmentation/traintest.txt'
    with open(list_name) as f:
        file_list = f.read().splitlines()

    # If we want to prepare training maskdb, first try to load gts
    #if args.db_name == 'train':
    #    if os.path.exists(args.roidb) and os.path.exists(args.maskdb):
    #        with open(args.roidb, 'rb') as f:
    #            gt_roidbs = cPickle.load(f)
    #        with open(args.maskdb, 'rb') as f:
    #            gt_maskdbs = cPickle.load(f)
    #    else:
    #        db = PascalVOCSeg('train', '2012', 'data/VOCdevkitSDS/')
    #        gt_roidbs = db.gt_roidb()
    #        gt_maskdbs = db.gt_maskdb()

    top_k = args.top_k
    num_process = args.para_job
    # Prepare train/val maskdb use multi-process
    processes = []
    file_start = 0
    file_offset = int(np.ceil(len(file_list) / float(num_process)))
    for process_id in xrange(num_process):
        file_end = min(file_start + file_offset, len(file_list))
        p = Process(target=process_roidb, args=(file_start, file_end, args.db_name))
        p.start()
        processes.append(p)
        file_start += file_offset

    for p in processes:
        p.join()


