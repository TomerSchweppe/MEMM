#!/usr/bin/env python

import numpy as np
from scipy import sparse
import time


def loss_function_no_for(v, spr_mats, tag_idx_tup, Lambda):
    """loss function for MEMM model (no for loop)"""
    v_times_f = np.reshape(spr_mats.dot(v), (len(tag_idx_tup), -1))
    loss_accum = np.sum(v_times_f[tuple(np.arange(len(tag_idx_tup))), tag_idx_tup])
    loss_accum -= np.sum(np.log(np.sum(np.exp(v_times_f), axis=1)))
    loss_accum -= (Lambda / 2) * np.linalg.norm(v)
    # print('no for loss', -loss_accum)
    return -loss_accum


def dloss_dv_no_for(v, spr_mats, tag_idx_tup, Lambda):
    """derivative function of MEMM loss (no for loop)"""
    tag_idx_fixed_list = [i * spr_mats.shape[0] / len(tag_idx_tup) + idx for i, idx in enumerate(tag_idx_tup)]
    deriv_accum = spr_mats[tag_idx_fixed_list].sum(axis=0).astype(np.float64)
    softmax = np.reshape(np.exp(spr_mats.dot(v)), (len(tag_idx_tup), -1))
    softmax /= np.sum(softmax, axis=1)[:, None]
    deriv_accum -= (spr_mats.transpose().dot(np.reshape(softmax, -1))).transpose()
    deriv_accum -= Lambda * v
    return -deriv_accum
