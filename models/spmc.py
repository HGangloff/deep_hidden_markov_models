'''
Maximum likelihood learning and Forward Backward estimation for the Semi
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
from spmc_fb_and_posterior import (jax_log_forward_backward,
    jax_get_post_marginals_probas, jax_get_post_pair_marginals_probas,
    jax_compute_llkh)

def reconstruct_A(T, A_sig_params_0, A_sig_params_1, X, nb_classes=2):
    A = jnp.ones((nb_classes, nb_classes, T - 1))

    for h_t_1 in range(nb_classes):
        tmp = jnp.exp(jnp.dot(X[:T - 1], jnp.swapaxes(A_sig_params_0[h_t_1], 0, 1))
                        + A_sig_params_1[h_t_1])
        tmp = tmp.T
        tmp = tmp / jnp.sum(tmp, axis=0, keepdims=True)
        A = A.at[h_t_1].set(tmp)
        A = jnp.where(A < 1e-5, 1e-5, A)
        A = jnp.where(A > 0.99999, 0.99999, A)

    return A


def gradient_llkh(T, X, nb_iter, A_init, means_init,
    stds_init, H_gt=None, alpha=0.01, nb_classes=2, nb_channels=1):

    def compute_llkh_wrapper(A_sig_params_0, A_sig_params_1, means, stds,
            T, X, nb_channels=1, nb_classes=2):
        lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
            X, jnp.array([means[i, c] for c in range(nb_channels)]),
                jnp.array([stds[i, c] for c in range(nb_channels)])), axis=0)
                            for i in range(nb_classes)], axis=0)
        A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X,
            nb_classes=nb_classes)
        lA = jnp.log(A)
        return -jax_compute_llkh(T, lX_pdf, lA, nb_classes=nb_classes,
            nb_channels=nb_channels)

    Llkh_grad_A_sig_params_0 = jax.grad(compute_llkh_wrapper, argnums=0)
    Llkh_grad_A_sig_params_1 = jax.grad(compute_llkh_wrapper, argnums=1)

    T = len(X)

    # NOTE we currently have an initialization to an equiprobable matrix
    # term multiplied with X, there are the channels
    A_sig_params_0 = jnp.zeros((nb_classes, nb_classes, nb_channels))
    # dim1 : FROM (h_{k-1}), dim2 : TO (h_k)
    # term which do not mutiply with X -> no channels
    A_sig_params_1 = 3 * jnp.eye(nb_classes)
    # dim1 : FROM, dim2 : TO
    A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X,
        nb_classes=nb_classes)

    means = np.stack([means_init for t in range(T)], axis=2)
    stds = np.stack([stds_init for t in range(T)], axis=2)
    # means and stds are [nb_classes, nb_channels, T]

    opt = optax.adam(alpha)

    opt_state_A_sig_params_0 = opt.init(A_sig_params_0)
    opt_state_A_sig_params_1 = opt.init(A_sig_params_1)

    for k in range(nb_iter):
        print("\nGradient EM iteration", k)
        T = len(X)
        lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
            X, jnp.array([means[i, c] for c in range(nb_channels)]),
                jnp.array([stds[i, c] for c in range(nb_channels)])), axis=0)
                            for i in range(nb_classes)], axis=0)
        lA = jnp.log(A)
        lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA,
            nb_classes=nb_classes)

        post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)

        llkh_grad_A_sig_params_0 = Llkh_grad_A_sig_params_0(A_sig_params_0,
            A_sig_params_1, means, stds, T, X, nb_channels=nb_channels,
            nb_classes=nb_classes)
        llkh_grad_A_sig_params_0, opt_state_A_sig_params_0 = \
            opt.update(llkh_grad_A_sig_params_0, opt_state_A_sig_params_0)

        llkh_grad_A_sig_params_1 = Llkh_grad_A_sig_params_1(A_sig_params_0,
            A_sig_params_1, means, stds, T, X, nb_channels=nb_channels,
            nb_classes=nb_classes)
        llkh_grad_A_sig_params_1, opt_state_A_sig_params_1 = \
            opt.update(llkh_grad_A_sig_params_1, opt_state_A_sig_params_1)

        # apply all the updates
        A_sig_params_0 = optax.apply_updates(A_sig_params_0, llkh_grad_A_sig_params_0)
        A_sig_params_1 = optax.apply_updates(A_sig_params_1, llkh_grad_A_sig_params_1)

        a = np.zeros((nb_classes, nb_channels))
        b = np.zeros((nb_classes, nb_channels))
        c = np.zeros((nb_classes, nb_channels))
        ### a coefficients, first need precompute some quantities
        tmp_X = np.stack([
            post_marginals_probas[1:, h][:, None] * X[1:] for h in
            range(nb_classes)], axis=1)
        tmp_X_1 = np.stack([
            post_marginals_probas[1:, h][:, None] * X[:-1] for h in
            range(nb_classes)], axis=1)
        tmp_X_1_X = np.stack([
            post_marginals_probas[1:, h][:, None] * X[:-1] * X[1:]
            for h in range(nb_classes)], axis=1)
        tmp_X_1_X_1 = np.stack([
            post_marginals_probas[1:, h][:, None] * X[:-1] ** 2
            for h in range(nb_classes)], axis=1)

        for h_t in range(nb_classes):
            a[h_t] = (1 / np.sum(tmp_X_1_X_1[:, h_t]) *
                1 / (1 - np.sum(tmp_X_1[:, h_t]) ** 2 /
                (np.sum(post_marginals_probas[1:, h_t]) *
                np.sum(tmp_X_1_X_1[:, h_t]))) *
                (np.sum(tmp_X_1_X[:, h_t]) - np.sum(tmp_X_1[:, h_t]) *
                np.sum(tmp_X[:, h_t]) / np.sum(post_marginals_probas[1:, h_t])))

            b[h_t] = ((np.sum(tmp_X[:, h_t]) - a[h_t] * np.sum(tmp_X_1[:, h_t])) /
                np.sum(post_marginals_probas[1:, h_t]))

            c[h_t] = (np.sum(post_marginals_probas[1:, h_t][:, None] *
                np.square(X[1:] - a[h_t] * X[:-1] - b[h_t])) /
                np.sum(post_marginals_probas[1:, h_t]))

        # Reconstruct parameters
        means = np.stack([a[h_t] * X[:-1] + b[h_t] for h_t in
            range(nb_classes)], axis=0)
        means = np.swapaxes(means, 1, 2)
        means = np.concatenate([means[..., 0][..., None], means], axis=2)
        stds = np.sqrt(np.stack([c for t in range(T)], axis=1))
        stds[stds < 1e-5] = 1e-5
        stds = np.swapaxes(stds, 1, 2)

        lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
            X, jnp.array([means[i, c] for c in range(nb_channels)]),
                jnp.array([stds[i, c] for c in range(nb_channels)])), axis=0)
                            for i in range(nb_classes)], axis=0)
        A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X,
            nb_classes=nb_classes)
        lA = jnp.log(A)
        llkh_train = jax_compute_llkh(T, lX_pdf, lA, nb_classes=nb_classes,
            nb_channels=nb_channels)
        print("likelihood", llkh_train)

    # final parameters
    T = len(X)
    means = np.stack([a[h_t] * X[:-1] + b[h_t]
                        for h_t in range(nb_classes)], axis=0)
    means = np.swapaxes(means, 1, 2)
    means = np.concatenate([means[..., 0][..., None], means], axis=2)
    stds = np.sqrt(np.stack([c for t in range(T)], axis=1))
    stds[stds < 1e-5] = 1e-5
    stds = np.swapaxes(stds, 1, 2)
    A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X,
        nb_classes=nb_classes)

    return (A, means, stds, (A_sig_params_0, A_sig_params_1), [a, b, c])
                    
def MPM_segmentation(T, X, A_sig_params, abc,
    H=None, nb_channels=1, nb_classes=2):
    if A_sig_params is not None and abc is not None:
        A_sig_params_0, A_sig_params_1 = A_sig_params
        a, b, c = abc
        means = np.stack([a[h_t] * X[:-1] + b[h_t]
                            for h_t in range(nb_classes)], axis=0)
        means = np.swapaxes(means, 1, 2)
        means = np.concatenate([means[..., 0][..., None], means], axis=2)
        stds = np.sqrt(np.stack([c for t in range(T)], axis=1))
        stds[stds < 1e-5] = 1e-5
        stds = np.swapaxes(stds, 1, 2)
        A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X,
            nb_classes=nb_classes)
        
    lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
        X, jnp.array([means[i, c] for c in range(nb_channels)]),
            jnp.array([stds[i, c] for c in range(nb_channels)])), axis=0)
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
