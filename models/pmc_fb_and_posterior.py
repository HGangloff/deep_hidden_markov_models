'''
Functions common to PMC and D-PMC models
'''

from functools import partial
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp

@partial(jax.jit, static_argnums=(0, 3))
def jax_compute_llkh(T, lX_pdf, lA, nb_classes):

    """
    Compute the loglikelihood of an observed sequence given model parameters
    Note that it needs an forward algorithm without rescaling
    """
    alpha_init = jnp.array([jnp.log(1 / nb_classes)
                            for h in range(nb_classes)])

    def scan_fn_a(alpha_t_1, t):
        alpha_t = jax_forward_one_step_no_rescaled(alpha_t_1, t, lX_pdf, lA)
        return alpha_t, alpha_t

    alpha_T, _ = jax.lax.scan(scan_fn_a, alpha_init, jnp.arange(1, T, 1))

    llkh = logsumexp(alpha_T)

    return llkh

def jax_forward_one_step(alpha_t_1, t, lX_pdf, lA, rescaled=True):
    alpha_t = (logsumexp(
            alpha_t_1[..., None] + lA[..., t - 1] + lX_pdf[..., t], axis=0)
            )
    if rescaled:
        alpha_t -= logsumexp(alpha_t)

    return alpha_t

def jax_forward_one_step_no_rescaled(alpha_t_1, t, lX_pdf, lA):
    return jax_forward_one_step(alpha_t_1, t, lX_pdf, lA, rescaled=False)

def jax_forward_one_step_rescaled(alpha_t_1, t, lX_pdf, lA):
    return jax_forward_one_step(alpha_t_1, t, lX_pdf, lA, rescaled=True)

@jax.jit
def jax_beta_one_step(beta_tp1, t, lX_pdf, lA):
    beta_t = logsumexp(lA[..., t] + lX_pdf[..., t + 1] + beta_tp1, axis=1)
    beta_t -= logsumexp(beta_t)

    return beta_t

@partial(jax.jit, static_argnums=(0, 3))
def jax_log_forward_backward(T, lX_pdf, lA, nb_classes):
    # NOTE: if we normalize alpha beta we do not have access to the llkh
    alpha_init = jnp.array([jnp.log(1 / nb_classes)
                            for h in range(nb_classes)])

    beta_init = jnp.log(jnp.array([1. for h_t in range(nb_classes)]))

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
                           # reverse : reverse the first axis of the arange and
                           # of the samples stacked in beta
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
def jax_get_post_pair_marginals_probas(T, lalpha, lbeta, lA, lX_pdf, nb_classes):
    def scan_h_t_1(carry1, h_t_1):
        lalpha, lA, lX_pdf, lbeta = carry1
        def scan_h_t(carry2, h_t):
            lbeta, lA, lX_pdf = carry2
            res = lA[:, h_t, :T - 1] + lX_pdf[:, h_t, 1:] + lbeta[1:, h_t]
            return (lbeta, lA, lX_pdf), res
        carry2 = lbeta, lA, lX_pdf
        _, res_h_t = jax.lax.scan(scan_h_t, carry2, jnp.arange(nb_classes))
        # res_h_t is (h_t: 2, h_t_1: 2, T - 1), NOTE res_h_t is stacked on h_t !
        
        res = lalpha[:-1, h_t_1] + res_h_t[:, h_t_1]
        return (lalpha, lA, lX_pdf, lbeta), res
    carry1 = lalpha, lA, lX_pdf, lbeta
    _, post_pair_marginals_probas = jax.lax.scan(scan_h_t_1, carry1,
        jnp.arange(nb_classes))
    # post_pair_marginals_probas is (h_t_1: 2, h_t: 2, T - 1)
    post_pair_marginals_probas = jnp.moveaxis(post_pair_marginals_probas, -1, 0)
    # post_pair_marginals_probas is (T - 1, h_t_1: 2, h_t: 2)

    post_pair_marginals_probas -= logsumexp(post_pair_marginals_probas,
        axis=(1, 2), keepdims=True)
    ppmp = jnp.exp(post_pair_marginals_probas)
    ppmp = jnp.where(ppmp < 1e-5, 1e-5, ppmp)
    ppmp = jnp.where(ppmp > 0.99999, 0.99999, ppmp)
    return ppmp
