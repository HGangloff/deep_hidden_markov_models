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
from jax.experimental.host_callback import id_print, id_tap

from utils import (jax_loggauss, vmap_jax_loggauss)

@partial(jax.jit, static_argnums=(0, 3))
def jax_compute_llkh(T, lX_pdf, lA):
    """
    Compute the loglikelihood of an observed sequence given model parameters
    Note that it needs an forward algorithm without rescaling
    """
    alpha_init = jnp.log(jnp.array([0.5, 0.5])) + lX_pdf[:, 0]

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
    Note: it is rescaled
    """
    beta_t = logsumexp(lA + lX_pdf[:, t + 1] + beta_tp1, axis=1)
    beta_t -= logsumexp(beta_t)

    return beta_t

@partial(jax.jit, static_argnums=(0, 3))
def jax_log_forward_backward(T, lX_pdf, lA):
    alpha_init = jnp.log(jnp.array([0.5, 0.5])) + lX_pdf[:, 0]

    beta_init = jnp.log(jnp.array([1., 1.]))

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
def jax_get_post_pair_marginals_probas(T, lalpha, lbeta, lA, lX_pdf):

    def scan_h_t_1(carry1, h_t_1):
        lalpha, lA, lX_pdf, lbeta = carry1
        def scan_h_t(carry2, h_t):
            lbeta, lA, lX_pdf = carry2
            res = lA[:, h_t][..., None] + lX_pdf[h_t, 1:] + lbeta[1:, h_t]
            return (lbeta, lA, lX_pdf), res
        carry2 = lbeta, lA, lX_pdf
        _, res_h_t = jax.lax.scan(scan_h_t, carry2, jnp.arange(2))
        
        res = lalpha[:-1, h_t_1] + res_h_t[:, h_t_1]
        return (lalpha, lA, lX_pdf, lbeta), res
    carry1 = lalpha, lA, lX_pdf, lbeta
    _, post_pair_marginals_probas = jax.lax.scan(scan_h_t_1, carry1,
        jnp.arange(2))
    post_pair_marginals_probas = jnp.moveaxis(post_pair_marginals_probas, -1, 0)

    post_pair_marginals_probas -= logsumexp(post_pair_marginals_probas,
        axis=(1, 2), keepdims=True)
    ppmp = jnp.exp(post_pair_marginals_probas)
    ppmp = jnp.where(ppmp < 1e-5, 1e-5, ppmp)
    ppmp = jnp.where(ppmp > 0.99999, 0.99999, ppmp)
    return ppmp

def EM(T, X, nb_iter, A_init, means_init, stds_init):

    # initialisation for the parameters
    A = A_init
    means = jnp.asarray(means_init)
    stds = jnp.asarray(stds_init)

    def scan_on_iter_loop(carry, k):
        (means, stds, A) = carry
        k = id_print(k, what="####### ITERATION #######")
        T = len(X)
        lX_pdf = jnp.stack([jax_loggauss(
            X, means[i], stds[i]) for i in range(2)],
            axis=0)
        lA = jnp.log(A)
        lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA)

        post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)
        post_pair_marginals_probas = jax_get_post_pair_marginals_probas(T,
                                        lalpha, lbeta, lA, lX_pdf)


        A = jnp.stack([
            jnp.stack([
                jnp.sum(post_pair_marginals_probas[:, h_t_1, h_t]) /
                jnp.sum(post_pair_marginals_probas[:, h_t_1])
                for h_t in range(2)],
                axis=0
                )
            for h_t_1 in range(2)],
            axis=0
        )

        tmp_for_means= jnp.stack([post_marginals_probas[:, h] * (X)
                                 for h in range(2)], axis=1)
        means = jnp.stack([
                jnp.sum(tmp_for_means[:, h_t]) /
                jnp.sum(post_marginals_probas[:, h_t]) for h_t in range(2)],
                axis=0)

        tmp_for_stds = jnp.stack([
            post_marginals_probas[:, h] * (X - means[h]) ** 2
            for h in range(2)], axis=1)
        stds = jnp.stack([
                jnp.sqrt((jnp.sum(tmp_for_stds[:, h_t]) /
                jnp.sum(post_marginals_probas[:, h_t]))) for h_t in
                range(2)],
                axis=0)
        stds = jnp.where(stds <= 1e-5, 1e-5, stds)

        return ((means, stds, A), k)
    carry = (means, stds, A)
    (means, stds, A), _ = jax.lax.scan(scan_on_iter_loop, carry,
        jnp.arange(0, nb_iter))


    return A, means, stds
                    
def MPM_segmentation(T, X, A, means, stds, H=None):
    lX_pdf = jnp.stack([vmap_jax_loggauss(
        X, means[i], stds[i]) for i in range(2)], axis=0)
    lA = jnp.log(A)
    lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA)

    post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)

    mpm_seg = np.argmax(post_marginals_probas, axis=1)

    if H is not None:
        e = np.count_nonzero(mpm_seg != H) / H.shape[0]
        print("Error in MPM segmentation", e)
        mpm_seg_ = mpm_seg
    else:
        e = None

    return np.asarray(mpm_seg), e
