#!/usr/bin/env python

import numpy as np
from scipy import sparse
import time

LAMBDA = 10 ** (-7)


def loss_function(v, spr_mats):
    """
    loss function
    """
    linear_sum = 0
    log_sum = 0
    for (mat, tag_idx) in spr_mats:
        dense_dot = mat.dot(v)
        linear_sum += dense_dot[tag_idx]
        log_sum += np.log(np.sum(np.exp(dense_dot)))
    # print('for_loss', linear_sum - log_sum - (LAMBDA / 2) * np.linalg.norm(v))
    return -(linear_sum - log_sum - (LAMBDA / 2) * np.linalg.norm(v))


def loss_function_no_for(v, spr_mats, tag_idx_tup):
    """loss function for MEMM model (no for loop)"""
    v_times_f = np.reshape(spr_mats.dot(v), (len(tag_idx_tup), -1))
    loss_accum = np.sum(v_times_f[tuple(np.arange(len(tag_idx_tup))), tag_idx_tup])
    loss_accum -= np.sum(np.log(np.sum(np.exp(v_times_f), axis=1)))
    loss_accum -= (LAMBDA / 2) * np.linalg.norm(v)
    # print('no for loss', -loss_accum)
    return -loss_accum


def dloss_dv_no_for(v, spr_mats, tag_idx_tup):
    """derivative function of MEMM loss (no for loop)"""
    tag_idx_fixed_list = [i * spr_mats.shape[0] / len(tag_idx_tup) + idx for i, idx in enumerate(tag_idx_tup)]
    deriv_accum = spr_mats[tag_idx_fixed_list].sum(axis=0).astype(np.float64)
    softmax = np.reshape(np.exp(spr_mats.dot(v)), (len(tag_idx_tup), -1))
    softmax /= np.sum(softmax, axis=1)[:, None]
    deriv_accum -= (spr_mats.transpose().dot(np.reshape(softmax, -1))).transpose()
    deriv_accum -= LAMBDA * v
    return -deriv_accum


def dloss_dv(v, spr_mats):
    """
    calculate the derivative of the loss function
    """
    feature_dim = spr_mats[0][0].shape[1]
    vector_linear_sum = sparse.csr_matrix((1, feature_dim))
    vector_exp_sum = sparse.csr_matrix((1, feature_dim))

    for spr_mat, tag_idx in spr_mats:
        # add word, tag pair feature vector to the accumulator
        vector_linear_sum += spr_mat.getrow(tag_idx)

        # calculate the softmax for each tag for current word
        softmax_for_each_tag = np.exp(spr_mat.dot(v))
        softmax_for_each_tag /= np.sum(softmax_for_each_tag)
        vector_exp_sum += (spr_mat.transpose().dot(softmax_for_each_tag)).transpose()
    return -(vector_linear_sum - vector_exp_sum - LAMBDA * v)
