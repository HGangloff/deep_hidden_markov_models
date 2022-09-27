'''
Expectation Maximization and Forward Backward algorithm for classical Hidden
Markov Chains
'''

import numpy as np
from functools import partial
import jax.numpy as jnp
import jax
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp

from utils import (jax_loggauss, vmap_jax_loggauss)

@partial(jax.jit, static_argnums=(0, 3))
def jax_compute_llkh(T, lX_pdf, lA, nb_classes=2):
    """
    Compute the loglikelihood of an observed sequence given model parameters
    Note that it needs an forward algorithm without rescaling
    """
    alpha_init = jnp.log(1 / nb_classes) + lX_pdf[:, 0]

    def scan_fn_a(alpha_t_1, t):
        alpha_t = jax_forward_one_step_no_rescaled(alpha_t_1, t, lX_pdf, lA)
        return alpha_t, alpha_t

    alpha_T, _ = jax.lax.scan(scan_fn_a, alpha_init, jnp.arange(1, T, 1))

    llkh = logsumexp(alpha_T)

    return llkh

def jax_forward_one_step(alpha_t_1, t, lX_pdf, lA, rescaled=True):
    
    alpha_t = (logsumexp(
                alpha_t_1[..., None]
                + lA, axis=0)
                + lX_pdf[:, t])
    if rescaled:
        alpha_t -= logsumexp(alpha_t)

    return alpha_t

def jax_forward_one_step_no_rescaled(alpha_t_1, t, lX_pdf, lA):
    return jax_forward_one_step(alpha_t_1, t, lX_pdf, lA, rescaled=False)

def jax_forward_one_step_rescaled(alpha_t_1, t, lX_pdf, lA):
    return jax_forward_one_step(alpha_t_1, t, lX_pdf, lA, rescaled=True)

@jax.jit
def jax_beta_one_step(beta_tp1, t, lX_pdf, lA):
    """
    It is rescaled
    """
    beta_t = logsumexp(lA + lX_pdf[:, t + 1] + beta_tp1, axis=1)
    beta_t -= logsumexp(beta_t)

    return beta_t

@partial(jax.jit, static_argnums=(0, 3))
def jax_log_forward_backward(T, lX_pdf, lA, nb_classes=2):
    alpha_init = jnp.log(1 / nb_classes) + lX_pdf[:, 0]

    beta_init = jnp.log(jnp.ones(nb_classes))

    def scan_fn_a(alpha_t_1, t):
        alpha_t = jax_forward_one_step_rescaled(alpha_t_1, t, lX_pdf, lA)
        return alpha_t, alpha_t

    def scan_fn_b(beta_tp1, t):
        beta_t = jax_beta_one_step(beta_tp1, t, lX_pdf, lA)
        return beta_t, beta_t

    _, alpha = jax.lax.scan(scan_fn_a, alpha_init, jnp.arange(1, T, 1))
    alpha = jnp.concatenate([alpha_init[None, ...], alpha], axis=0)
    _, beta = jax.lax.scan(scan_fn_b, beta_init, jnp.arange(0, T - 1, 1),
                           reverse=True) 
    beta = jnp.concatenate([beta, beta_init[None, ...]], axis=0)

    return alpha, beta

@jax.jit
def jax_get_post_marginals_probas(lalpha, lbeta):
    post_marginals_probas = lalpha + lbeta
    post_marginals_probas -= logsumexp(post_marginals_probas,
                                axis=1, keepdims=True)
    pmp = jnp.exp(post_marginals_probas)
    pmp = jnp.where(pmp < 1e-5, 1e-5, pmp)
    pmp = jnp.where(pmp > 0.99999, 0.99999, pmp)
    return pmp

@partial(jax.jit, static_argnums=(0, 5))
def jax_get_post_pair_marginals_probas(T, lalpha, lbeta, lA, lX_pdf,
    nb_classes=2):
    post_pair_marginals_probas = jnp.empty((T - 1, nb_classes, nb_classes))

    for h_t_1 in range(nb_classes):
        for h_t in range(nb_classes):
            post_pair_marginals_probas = post_pair_marginals_probas.at[:, h_t_1, h_t].set(
                lalpha[:T - 1, h_t_1] +
                lA[h_t_1, h_t] +
                lX_pdf[h_t, 1:] + lbeta[1:, h_t])

    post_pair_marginals_probas -= logsumexp(post_pair_marginals_probas,
        axis=(1, 2), keepdims=True)
    ppmp = jnp.exp(post_pair_marginals_probas)
    ppmp = jnp.where(ppmp < 1e-5, 1e-5, ppmp)
    ppmp = jnp.where(ppmp > 0.99999, 0.99999, ppmp)
    return ppmp

def EM(T, X, nb_iter=1000, A_init=None, means_init=None,
    stds_init=None, nb_classes=2, nb_channels=1):


    # initialisation for the parameters
    A = A_init
    means = means_init
    stds = stds_init

    X_train = X
    T = len(X_train)

    llkh_train_list = []
    llkh_test_list = []

    for k in range(nb_iter):
        print("EM iteration", k)
        # TRAIN
        T = len(X_train)
        lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
            X_train, means[i, :], stds[i, :]), axis=0)
                            for i in range(nb_classes)], axis=0)
        lA = jnp.log(A)
        lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA,
            nb_classes=nb_classes)

        post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)
        post_pair_marginals_probas = jax_get_post_pair_marginals_probas(T,
                                        lalpha, lbeta, lA, lX_pdf,
                                        nb_classes=nb_classes)
        new_A = np.zeros(A.shape)
        new_means = np.zeros(means.shape)
        new_stds = np.zeros(stds.shape)

        # the None that is needed next and below is strange
        tmp_for_means= jnp.stack([post_marginals_probas[:, h][:, None] * (X_train)
                                 for h in range(nb_classes)], axis=1)

        for h_t in range(nb_classes):
            for h_t_1 in range(nb_classes):
                new_A[h_t_1, h_t] += (jnp.sum(post_pair_marginals_probas[:,
                    h_t_1, h_t]) / jnp.sum(post_pair_marginals_probas[:, h_t_1]
                    ))
            new_means[h_t] = (jnp.sum(tmp_for_means[:, h_t], axis=0) /
                jnp.sum(post_marginals_probas[:, h_t]))
        tmp_for_stds = jnp.stack([
            post_marginals_probas[:, h][:, None] * (X_train - new_means[h]) ** 2
            for h in range(nb_classes)], axis=1)
        for h_t in range(nb_classes):
            new_stds[h_t] = jnp.sqrt((jnp.sum(tmp_for_stds[:, h_t], axis=0) /
                jnp.sum(post_marginals_probas[:, h_t])))
        new_stds = jnp.where(new_stds <= 1e-5, 1e-5, new_stds)

        A = new_A
        means = new_means
        stds = new_stds

        lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
            X_train, jnp.array([means[i, c] for c in range(nb_channels)]),
                jnp.array([stds[i, c] for c in range(nb_channels)])), axis=0)
                            for i in range(nb_classes)], axis=0)
        lA = jnp.log(A)
        llkh_train = jax_compute_llkh(T, lX_pdf, lA)
        llkh_train_list.append(llkh_train / T)

    return A, means, stds
                    
def MPM_segmentation(T, X, A, means, stds, H=None,
                     nb_channels=1, nb_classes=2):
    lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
        X, means[i, :], stds[i, :]), axis=0)
                        for i in range(nb_classes)], axis=0)
    lA = jnp.log(A)
    lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA,
        nb_classes=nb_classes)

    post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)

    mpm_seg = np.argmax(post_marginals_probas, axis=1)

    if H is not None:
        e = np.count_nonzero(mpm_seg != H) / H.shape[0]
        print("Error in MPM segmentation", e)
        mpm_seg_ = mpm_seg
    else:
        e = None

    return np.asarray(mpm_seg), e
