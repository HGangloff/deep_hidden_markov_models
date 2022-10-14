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
    jax_get_post_marginals_probas,
    jax_compute_llkh)

@partial(jax.jit, static_argnums=(0, 4))
def reconstruct_A(T, A_sig_params_0, A_sig_params_1, X, nb_classes):
    def scan_h_t_1(carry, h_t_1):
        A_sig_params_0, A_sig_params_1 = carry
        tmp = jnp.exp(jnp.dot(X[:-1], jnp.swapaxes(A_sig_params_0[h_t_1], 0, 1))
                        + A_sig_params_1[h_t_1])
        tmp = tmp.T
        tmp = tmp / jnp.sum(tmp, axis=0, keepdims=True)
        return (A_sig_params_0, A_sig_params_1), tmp
    carry = (A_sig_params_0, A_sig_params_1)

    # we get back the stack of samples (ie, the stack of second position return
    # of the scan function)
    _, A = jax.lax.scan(scan_h_t_1, carry, jnp.arange(nb_classes))

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

    # needed for the initialization of the carry of the main loop
    a = np.zeros((nb_classes, nb_channels))
    b = np.zeros((nb_classes, nb_channels))
    c = np.zeros((nb_classes, nb_channels))

    opt = optax.adam(alpha)

    opt_state_A_sig_params_0 = opt.init(A_sig_params_0)
    opt_state_A_sig_params_1 = opt.init(A_sig_params_1)

    def scan_on_iter_loop(carry, k):
        (A_sig_params_0, A_sig_params_1, a, b, c, means, stds, opt_state_A_sig_params_0,
            opt_state_A_sig_params_1) = carry
        T = len(X)
        lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
            X, jnp.array([means[i, c] for c in range(nb_channels)]),
                jnp.array([stds[i, c] for c in range(nb_channels)])), axis=0)
                            for i in range(nb_classes)], axis=0)
        A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X,
            nb_classes=nb_classes)
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

        ### a coefficients, first need precompute some quantities
        tmp_X = jnp.stack([
            post_marginals_probas[1:, h][:, None] * X[1:] for h in
            range(nb_classes)], axis=1)
        tmp_X_1 = jnp.stack([
            post_marginals_probas[1:, h][:, None] * X[:-1] for h in
            range(nb_classes)], axis=1)
        tmp_X_1_X = jnp.stack([
            post_marginals_probas[1:, h][:, None] * X[:-1] * X[1:]
            for h in range(nb_classes)], axis=1)
        tmp_X_1_X_1 = jnp.stack([
            post_marginals_probas[1:, h][:, None] * X[:-1] ** 2
            for h in range(nb_classes)], axis=1)

        a = jnp.stack([1 / jnp.sum(tmp_X_1_X_1[:, h_t], axis=0) *
            1 / (1 - jnp.sum(tmp_X_1[:, h_t], axis=0) ** 2 /
            (jnp.sum(post_marginals_probas[1:, h_t]) *
            jnp.sum(tmp_X_1_X_1[:, h_t], axis=0))) *
            (jnp.sum(tmp_X_1_X[:, h_t], axis=0) - jnp.sum(tmp_X_1[:, h_t], axis=0) *
            jnp.sum(tmp_X[:, h_t], axis=0) / jnp.sum(post_marginals_probas[1:, h_t]))
            for h_t in range(nb_classes)],
            axis=0
        )

        b = jnp.stack([(jnp.sum(tmp_X[:, h_t], axis=0) - a[h_t]
                * jnp.sum(tmp_X_1[:, h_t], axis=0)) / 
            jnp.sum(post_marginals_probas[1:, h_t])
            for h_t in range(nb_classes)],
            axis=0
        )

        c = jnp.stack([jnp.sum(post_marginals_probas[1:, h_t][:, None] *
            jnp.square(X[1:] - a[h_t] * X[:-1] - b[h_t]), axis=0) /
                jnp.sum(post_marginals_probas[1:, h_t])
                for h_t in range(nb_classes)],
            axis=0
        )

        # needed for the carry to keep the same signature when nb_channels==1
        a = a[..., None] if a.ndim == 1 else a
        b = b[..., None] if b.ndim == 1 else b
        c = c[..., None] if c.ndim == 1 else c

        # Reconstruct parameters
        means = jnp.stack([a[h_t] * X[:-1] + b[h_t] for h_t in
            range(nb_classes)], axis=0)
        means = jnp.swapaxes(means, 1, 2)
        means = jnp.concatenate([means[..., 0][..., None], means], axis=2)
        stds = jnp.sqrt(jnp.tile(c[..., None], (1, 1, T))) # None is needed for
        # tiling to work good, using a for loop on T here destroys the scan lax
        # and never compiles
        stds = jnp.where(stds < 1e-5, 1e-5, stds)

        llkh = -compute_llkh_wrapper(A_sig_params_0, A_sig_params_1, means, stds,
            T, X, nb_channels, nb_classes)
        llkh = id_print(llkh, what="loglikelihood")


        k = id_print(k, what="####### ITERATION #######")

        return (A_sig_params_0, A_sig_params_1, a, b, c, means, stds, opt_state_A_sig_params_0,
        opt_state_A_sig_params_1), k

    carry = (A_sig_params_0, A_sig_params_1, a, b, c, means, stds,
        opt_state_A_sig_params_0, opt_state_A_sig_params_1) 
    (A_sig_params_0, A_sig_params_1, a, b, c, means, stds, opt_state_A_sig_params_0,
        opt_state_A_sig_params_1), _ = jax.lax.scan(scan_on_iter_loop, carry,
        jnp.arange(0, nb_iter)
    )

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
