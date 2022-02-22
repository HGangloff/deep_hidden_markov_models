'''
Maximum likelihood learning and Forward Backward estimation for the
Pairwise Markov Chains
'''

import numpy as np

from functools import partial
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from jax.experimental.host_callback import id_print

import optax

from utils import (jax_loggauss, vmap_jax_loggauss)
from pmc_fb_and_posterior import (jax_log_forward_backward,
    jax_get_post_marginals_probas, jax_get_post_pair_marginals_probas,
    jax_compute_llkh)

def reconstruct_A(T, A_sig_params, X):
    def scan_h_t_1(carry, h_t_1):
        A_sig_params = carry
        tmp = jnp.exp(-(A_sig_params[0, h_t_1] * X[:-1] +
            A_sig_params[1, h_t_1]))
        tmp = tmp / (1 + tmp)
        return (A_sig_params), jnp.array([1 - tmp, tmp])
    carry = A_sig_params

    _, A = jax.lax.scan(scan_h_t_1, carry, jnp.arange(2))

    A = jnp.where(A < 1e-5, 1e-5, A)
    A = jnp.where(A > 0.99999, 0.99999, A)

    return A

def reconstruct_means_stds_from_params(T, X, a, b, c):
        means_0 = jnp.stack([a[0, h_t] * X[:-1] + b[0, h_t] for h_t in range(2)], axis=0)
        means_0 = jnp.concatenate([jnp.array([[a[0, 0] * X[-1] + b[0, 0]],
                                        [a[0, 1] * X[-1] + b[0, 1]]]),
            means_0], axis=1)
        means_1 = jnp.stack([a[1, h_t] * X[:-1] + b[1, h_t] for h_t in range(2)], axis=0)
        means_1 = jnp.concatenate([jnp.array([[a[1, 0] * X[-1] + b[1, 0]],
                                        [a[1, 1] * X[-1] + b[1, 1]]]),
            means_1], axis=1)
        means = jnp.stack([means_0, means_1], axis=0)
        stds = jnp.sqrt(jnp.tile(c[..., None], (1, 1, T)))
        stds = jnp.where(stds < 1e-5, 1e-5, stds)
        return means, stds

def gradient_llkh(T, X, nb_iter, A_init, means_init, stds_init, H_gt=None,
    alpha=0.01):

    @partial(jax.jit, static_argnums=(3))
    def compute_llkh_wrapper(A_sig_params, means, stds, T, X):
        lX_pdf = jnp.stack([
                    jnp.stack([
                        jax_loggauss(X, means[j, i], stds[j, i])
                    for i in range(2)], axis=0)
            for j in range(2)], axis=0)

        A = reconstruct_A(T, A_sig_params, X)
        lA = jnp.log(A)
        return -jax_compute_llkh(T, lX_pdf, lA)

    Llkh_grad_A_sig_params = jax.grad(compute_llkh_wrapper, argnums=0)

    T = len(X)

    A_sig_params = np.zeros((2, 2)) 
    A_sig_params[1, 0] = -np.log(A_init[0, 1] / (1 - A_init[0, 1]))
    A_sig_params[1, 1] = -np.log(A_init[1, 1] / (1 - A_init[1, 1]))

    means = np.stack([
        np.stack([means_init for k in range(2)], axis=0)
        for t in range(T)], axis=2)
    stds = np.stack([
        np.stack([stds_init for k in range(2)], axis=0)
        for t in range(T)], axis=2)

    a = np.zeros((2, 2))
    b = np.zeros((2, 2))
    c = np.zeros((2, 2))

    opt = optax.adam(alpha)

    opt_state_A_sig_params = opt.init(A_sig_params)

    def scan_on_iter_loop(carry, k):
        (A_sig_params, a, b, c, means, stds, opt_state_A_sig_params) = carry
        T = len(X)
        lX_pdf = jnp.stack([
                    jnp.stack([
                        jax_loggauss(X, means[j, i], stds[j, i])
                    for i in range(2)], axis=0)
            for j in range(2)], axis=0)
        A = reconstruct_A(T, A_sig_params, X)
        lA = jnp.log(A)
        lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA)

        post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)
        post_pair_marginals_probas = jax_get_post_pair_marginals_probas(T,
                                        lalpha, lbeta, lA, lX_pdf)

        llkh_grad_A_sig_params = Llkh_grad_A_sig_params(A_sig_params, means,
            stds, T, X)
        llkh_grad_A_sig_params, opt_state_A_sig_params = \
            opt.update(llkh_grad_A_sig_params, opt_state_A_sig_params)

        A_sig_params = optax.apply_updates(A_sig_params, llkh_grad_A_sig_params)

        tmp_X = jnp.stack([
            jnp.stack([post_pair_marginals_probas[:, h_t_1, h_t] * X[1:]
                       for h_t in range(2)], axis=1)
            for h_t_1 in range(2)],
            axis=1)
        tmp_X_1 = jnp.stack([
            jnp.stack([post_pair_marginals_probas[:, h_t_1, h_t] * X[:-1]
                       for h_t in range(2)], axis=1)
            for h_t_1 in range(2)],
            axis=1)
        tmp_X_1_X = jnp.stack([
            jnp.stack([post_pair_marginals_probas[:, h_t_1, h_t] *
                        X[:-1] * X[1:]
                    for h_t in range(2)], axis=1)
            for h_t_1 in range(2)],
            axis=1)
        tmp_X_1_X_1 = jnp.stack([
            jnp.stack([post_pair_marginals_probas[:, h_t_1, h_t] * X[:-1] ** 2
                        for h_t in range(2)], axis=1)
            for h_t_1 in range(2)],
            axis=1)
        tmp_X_ = jnp.stack([
            post_marginals_probas[1:, h_t] * X[1:]
            for h_t in range(2)], axis=1)
        tmp_X_1_ = jnp.stack([
            post_marginals_probas[1:, h_t] * X[:-1]
            for h_t in range(2)], axis=1)
        tmp_X_1_X_ = jnp.stack([
            post_marginals_probas[1:, h_t] * X[:-1] * X[1:]
            for h_t in range(2)], axis=1)
        tmp_X_1_X_1_ = jnp.stack([
            post_marginals_probas[1:, h_t] * X[:-1] ** 2
            for h_t in range(2)], axis=1)

        a = jnp.stack([jnp.stack([
            (1 / jnp.sum(tmp_X_1_X_1[:, h_t_1, h_t]) *
            1 / (1 - jnp.sum(tmp_X_1[:, h_t_1, h_t]) ** 2 /
            (jnp.sum(post_pair_marginals_probas[:, h_t_1, h_t]) *
            jnp.sum(tmp_X_1_X_1[:, h_t_1, h_t]))) *
            (jnp.sum(tmp_X_1_X[:, h_t_1, h_t]) -
            jnp.sum(tmp_X_1[:, h_t_1, h_t]) *
            jnp.sum(tmp_X[:, h_t_1, h_t]) /
            jnp.sum(post_pair_marginals_probas[:, h_t_1, h_t])))
            for h_t in range(2)], axis=0)
            for h_t_1 in range(2)], axis=0)
        a_ = jnp.stack([(1 / jnp.sum(tmp_X_1_X_1_[:, h_t]) *
            1 / (1 - jnp.sum(tmp_X_1_[:, h_t]) ** 2 /
            (jnp.sum(post_marginals_probas[1:, h_t]) *
            jnp.sum(tmp_X_1_X_1_[:, h_t]))) *
            (jnp.sum(tmp_X_1_X_[:, h_t]) - jnp.sum(tmp_X_1_[:, h_t]) *
            jnp.sum(tmp_X_[:, h_t]) / jnp.sum(post_marginals_probas[1:, h_t])))
            for h_t in range(2)], axis=0)
        b_ = jnp.stack([(jnp.sum(tmp_X_[:, h_t]) - a_[h_t] *
            jnp.sum(tmp_X_1_[:, h_t])) / 
        jnp.sum(post_marginals_probas[1:, h_t]) for h_t in range(2)],
            axis=0)

        c_ = jnp.stack([(jnp.sum(post_marginals_probas[1:, h_t] *
            jnp.square(X[1:] - a_[h_t] * X[:-1] - b_[h_t])) /
            jnp.sum(post_marginals_probas[1:, h_t])) for h_t in range(2)],
            axis=0)

        b = jnp.stack([b_ for h in range(2)], axis=0)
        c = jnp.stack([c_ for h in range(2)], axis=0)

        means, stds = reconstruct_means_stds_from_params(T, X, a, b, c)

        llkh = jax_compute_llkh(T, lX_pdf, lA)
        llkh = - llkh
        llkh = id_print(llkh / T, what="llkh")

        k = id_print(k, what="####### ITERATION #######")

        return (A_sig_params, a, b, c, means, stds, opt_state_A_sig_params), k
    carry = (A_sig_params, a, b, c, means, stds, opt_state_A_sig_params)

    (A_sig_params, a, b, c, means, stds, opt_state_A_sig_params), _ = \
        jax.lax.scan(scan_on_iter_loop, carry, jnp.arange(0, nb_iter))

    # final parameters
    T = len(X)
    means, stds = reconstruct_means_stds_from_params(T, X, a, b, c)
    A = reconstruct_A(T, A_sig_params, X)

    return (A, means, stds, A_sig_params, [a, b, c])
                    
def MPM_segmentation(T, X, A_sig_params, abc, H=None):
    a, b, c = abc
    means, stds = reconstruct_means_stds_from_params(T, X, a, b, c)
    A = reconstruct_A(T, A_sig_params, X)

    lX_pdf = jnp.stack([
                jnp.stack([
                    jax_loggauss(X, means[j, i], stds[j, i])
                for i in range(2)], axis=0)
        for j in range(2)], axis=0)
    lA = jnp.log(A)
    lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA)

    post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)

    mpm_seg = np.argmax(post_marginals_probas, axis=1)

    if H is not None:
        e = np.count_nonzero(mpm_seg != H) / H.shape[0]
        print("Error in MPM segmentation", e)
    else:
        e = None

    return mpm_seg, e
