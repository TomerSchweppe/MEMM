#!/usr/bin/env python

def loss_function(v,spr_mats):
    """
    loss function
    """

    # fake v todo -> remove
    (tmp,_) = spr_mats[0]
    v = np.ravel(tmp.todense()[0])


    linear_sum = 0
    log_sum = 0
    for (mat,tag_idx) in spr_mats:
        dense_dot = mat.dot(v)
        linear_sum += v[tag_idx]
        log_sum += np.log(np.sum(np.exp(dense_dot)))

    return linear_sum-log_sum-(LAMBDA/2)*np.linalg.norm(v)