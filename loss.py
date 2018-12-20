#!/usr/bin/env python

import numpy as np
from scipy import sparse

LAMBDA = 1


def loss_function(v, spr_mats):
    """
    loss function
    """

    # fake v todo -> remove
    (tmp, _) = spr_mats[0]
    v = np.ravel(tmp.todense()[0])

    linear_sum = 0
    log_sum = 0
    for (mat, tag_idx) in spr_mats:
        dense_dot = mat.dot(v)
        linear_sum += v[tag_idx]
        log_sum += np.log(np.sum(np.exp(dense_dot)))

    return linear_sum - log_sum - (LAMBDA / 2) * np.linalg.norm(v)


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
        vector_exp_sum += spr_mat.transpose().dot(softmax_for_each_tag).transpose()
    return vector_linear_sum - vector_exp_sum - LAMBDA * v
