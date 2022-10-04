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

@partial(jax.jit, static_argnums=(0, 5))
def reconstruct_means_stds_from_params(T, X, a, b, c, nb_classes):
    # needed for the carry to keep the same signature when nb_channels==1
    a = a[..., None] if a.ndim == 2 else a
    b = b[..., None] if b.ndim == 2 else b
    c = c[..., None] if c.ndim == 2 else c

    means = jnp.stack([
        jnp.concatenate(
        [jnp.stack([a[h_t_1, h_t] * X[-1] + b[h_t_1, h_t] for h_t in
        range(nb_classes)], axis=0)[..., None, :],
        jnp.stack(
        [a[h_t_1, h_t] * X[:-1] + b[h_t_1, h_t] for h_t in range(nb_classes)],
        axis=0)],
        axis=1) for h_t_1 in range(nb_classes)],
        axis=0)
    means = jnp.swapaxes(means, 2, 3)
    stds = jnp.sqrt(jnp.tile(c[..., None], (1, 1, 1, T)))
    stds = jnp.where(stds < 1e-5, 1e-5, stds)
    return means, stds

def gradient_llkh(T, X, nb_iter, A_init, means_init, stds_init, H_gt=None,
    alpha=0.01, nb_classes=2, nb_channels=1):

    @partial(jax.jit, static_argnums=(4, 6, 7))
    def compute_llkh_wrapper(A_sig_params_0, A_sig_params_1, means, stds,
        T, X, nb_channels, nb_classes):
        lX_pdf = jnp.stack([
                            jnp.stack([
                jnp.sum(vmap_jax_loggauss(
                X, jnp.array([means[j, i, c] for c in range(nb_channels)]),
                jnp.array([stds[j, i, c] for c in range(nb_channels)])), axis=0)
                            for i in range(nb_classes)], axis=0)
            for j in range(nb_classes)], axis=0)

        A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X,
            nb_classes=nb_classes)
        lA = jnp.log(A)
        return -jax_compute_llkh(T, lX_pdf, lA, nb_classes=nb_classes)

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

    a = np.zeros((nb_classes, nb_classes, nb_channels))
    b = np.zeros((nb_classes, nb_classes, nb_channels))
    c = np.zeros((nb_classes, nb_classes, nb_channels))
    means = np.stack([
        np.stack([means_init for k in range(nb_classes)], axis=0)
        for t in range(T)], axis=3)
    stds = np.stack([
        np.stack([stds_init for k in range(nb_classes)], axis=0)
        for t in range(T)], axis=3)

    a = a[..., None] if a.ndim == 2 else a
    b = b[..., None] if b.ndim == 2 else b
    c = c[..., None] if c.ndim == 2 else c

    opt = optax.adam(alpha)

    opt_state_A_sig_params_0 = opt.init(A_sig_params_0)
    opt_state_A_sig_params_1 = opt.init(A_sig_params_1)

    def scan_on_iter_loop(carry, k):
        (A_sig_params_0, A_sig_params_1, a, b, c, means, stds,
        opt_state_A_sig_params_0, opt_state_A_sig_params_1) = carry
        # NOTE all these for loop slow down the process !
        T = len(X)
        lX_pdf = jnp.stack([
                            jnp.stack([
                jnp.sum(vmap_jax_loggauss(
                X, jnp.array([means[j, i, c] for c in range(nb_channels)]),
                jnp.array([stds[j, i, c] for c in range(nb_channels)])), axis=0)
                            for i in range(nb_classes)], axis=0)
            for j in range(nb_classes)], axis=0)
        A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X, nb_classes)
        lA = jnp.log(A)
        lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA,
            nb_classes=nb_classes)

        post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)
        post_pair_marginals_probas = jax_get_post_pair_marginals_probas(T,
                                        lalpha, lbeta, lA, lX_pdf, nb_classes)

        llkh_grad_A_sig_params_0 = Llkh_grad_A_sig_params_0(A_sig_params_0,
            A_sig_params_1, means, stds, T, X, nb_channels, nb_classes)
        llkh_grad_A_sig_params_0, opt_state_A_sig_params_0 = \
            opt.update(llkh_grad_A_sig_params_0, opt_state_A_sig_params_0)

        llkh_grad_A_sig_params_1 = Llkh_grad_A_sig_params_1(A_sig_params_0,
            A_sig_params_1, means, stds, T, X, nb_channels, nb_classes)
        llkh_grad_A_sig_params_1, opt_state_A_sig_params_1 = \
            opt.update(llkh_grad_A_sig_params_1, opt_state_A_sig_params_1)

        # apply all the updates
        A_sig_params_1 = optax.apply_updates(A_sig_params_1, llkh_grad_A_sig_params_1)
        A_sig_params_0 = optax.apply_updates(A_sig_params_0, llkh_grad_A_sig_params_0)

        tmp_X = jnp.stack([
            jnp.stack([post_pair_marginals_probas[:, h_t_1, h_t][:, None] * X[1:]
                       for h_t in range(nb_classes)], axis=1)
            for h_t_1 in range(nb_classes)],
            axis=1)
        tmp_X_1 = jnp.stack([
            jnp.stack([post_pair_marginals_probas[:, h_t_1, h_t][:, None] * X[:-1]
                       for h_t in range(nb_classes)], axis=1)
            for h_t_1 in range(nb_classes)],
            axis=1)
        tmp_X_1_X = jnp.stack([
            jnp.stack([post_pair_marginals_probas[:, h_t_1, h_t][:, None] *
                        X[:-1] * X[1:]
                    for h_t in range(nb_classes)], axis=1)
            for h_t_1 in range(nb_classes)],
            axis=1)
        tmp_X_1_X_1 = jnp.stack([
            jnp.stack([post_pair_marginals_probas[:, h_t_1, h_t][:, None] * X[:-1] ** 2
                        for h_t in range(nb_classes)], axis=1)
            for h_t_1 in range(nb_classes)],
            axis=1)
        tmp_X_ = jnp.stack([
            post_marginals_probas[1:, h_t][:, None] * X[1:]
            for h_t in range(nb_classes)], axis=1)
        tmp_X_1_ = jnp.stack([
            post_marginals_probas[1:, h_t][:, None] * X[:-1]
            for h_t in range(nb_classes)], axis=1)
        tmp_X_1_X_ = jnp.stack([
            post_marginals_probas[1:, h_t][:, None] * X[:-1] * X[1:]
            for h_t in range(nb_classes)], axis=1)
        tmp_X_1_X_1_ = jnp.stack([
            post_marginals_probas[1:, h_t][:, None] * X[:-1] ** 2
            for h_t in range(nb_classes)], axis=1)

        # NOTE in the PMC model, b and  c are updated as in the S-PMC model
        # otherwise we cannot get any result. Probably due to the symmetric
        # structure
        a = jnp.stack([jnp.stack([
            (1 / jnp.sum(tmp_X_1_X_1[:, h_t_1, h_t], axis=0) *
            1 / (1 - jnp.sum(tmp_X_1[:, h_t_1, h_t], axis=0) ** 2 /
            (jnp.sum(post_pair_marginals_probas[:, h_t_1, h_t]) *
            jnp.sum(tmp_X_1_X_1[:, h_t_1, h_t], axis=0))) *
            (jnp.sum(tmp_X_1_X[:, h_t_1, h_t], axis=0) -
            jnp.sum(tmp_X_1[:, h_t_1, h_t], axis=0) *
            jnp.sum(tmp_X[:, h_t_1, h_t], axis=0) /
            jnp.sum(post_pair_marginals_probas[:, h_t_1, h_t])))
            for h_t in range(nb_classes)], axis=0)
            for h_t_1 in range(nb_classes)], axis=0)
                #b[h_t_1, h_t] = ((jnp.sum(tmp_X[:, h_t_1, h_t]) - a[h_t_1, h_t] *
                #    jnp.sum(tmp_X_1[:, h_t_1, h_t])) / 
                #    jnp.sum(post_pair_marginals_probas[:, h_t_1, h_t]))

                #c[h_t_1, h_t] = (jnp.sum(post_pair_marginals_probas[:, h_t_1, h_t] *
                #    jnp.square(X[1:] - a[h_t_1, h_t] * X[:-1]
                #    - b[h_t_1, h_t])) / jnp.sum(post_pair_marginals_probas[:, h_t_1, h_t]))
        a_ = jnp.stack([(1 / jnp.sum(tmp_X_1_X_1_[:, h_t], axis=0) *
            1 / (1 - jnp.sum(tmp_X_1_[:, h_t], axis=0) ** 2 /
            (jnp.sum(post_marginals_probas[1:, h_t]) *
            jnp.sum(tmp_X_1_X_1_[:, h_t], axis=0))) *
            (jnp.sum(tmp_X_1_X_[:, h_t], axis=0) - jnp.sum(tmp_X_1_[:, h_t], axis=0) *
            jnp.sum(tmp_X_[:, h_t], axis=0) / jnp.sum(post_marginals_probas[1:, h_t])))
            for h_t in range(nb_classes)], axis=0)
        a_ = a_[..., None] if a_.ndim == 1 else a_
        b_ = jnp.stack([(jnp.sum(tmp_X_[:, h_t], axis=0) - a_[h_t] *
            jnp.sum(tmp_X_1_[:, h_t], axis=0)) / 
        jnp.sum(post_marginals_probas[1:, h_t]) for h_t in range(nb_classes)],
            axis=0)

        c_ = jnp.stack([(jnp.sum(post_marginals_probas[1:, h_t][:, None] *
            jnp.square(X[1:] - a_[h_t] * X[:-1] - b_[h_t]), axis=0) /
            jnp.sum(post_marginals_probas[1:, h_t])) for h_t in range(nb_classes)],
            axis=0)

        b = jnp.stack([b_ for h in range(nb_classes)], axis=0)
        c = jnp.stack([c_ for h in range(nb_classes)], axis=0)

        ## needed for the carry to keep the same signature when nb_channels==1
        a = a[..., None] if a.ndim == 2 else a
        b = b[..., None] if b.ndim == 2 else b
        c = c[..., None] if c.ndim == 2 else c

        means, stds = reconstruct_means_stds_from_params(T, X, a, b, c, nb_classes)

        llkh = jax_compute_llkh(T, lX_pdf, lA, nb_classes=nb_classes)
        llkh = id_print(llkh, what="llkh")

        k = id_print(k, what="####### ITERATION #######")

        return (A_sig_params_0, A_sig_params_1, a, b, c, means, stds,
        opt_state_A_sig_params_0, opt_state_A_sig_params_1), k
    carry = (A_sig_params_0, A_sig_params_1, a, b, c, means, stds,
        opt_state_A_sig_params_0, opt_state_A_sig_params_1)

    (A_sig_params_0, A_sig_params_1, a, b, c, means, stds,
    opt_state_A_sig_params_0, opt_state_A_sig_params_1), _ = \
        jax.lax.scan(scan_on_iter_loop, carry, jnp.arange(0, nb_iter))

    # final parameters
    T = len(X)
    means, stds = reconstruct_means_stds_from_params(T, X, a, b, c, nb_classes)
    A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X, nb_classes)

    return (A, means, stds, (A_sig_params_0, A_sig_params_1), [a, b, c])
                    
def MPM_segmentation(T, X, A_sig_params, abc, H=None, nb_classes=2,
    nb_channels=1):
    a, b, c = abc
    A_sig_params_0, A_sig_params_1 = A_sig_params
    means, stds = reconstruct_means_stds_from_params(T, X, a, b, c, nb_classes)
    A = reconstruct_A(T, A_sig_params_0, A_sig_params_1, X, nb_classes)

    lX_pdf = jnp.stack([
                        jnp.stack([
            jnp.sum(vmap_jax_loggauss(
            X, jnp.array([means[j, i, c] for c in range(nb_channels)]),
            jnp.array([stds[j, i, c] for c in range(nb_channels)])), axis=0)
                        for i in range(nb_classes)], axis=0)
        for j in range(nb_classes)], axis=0)
    lA = jnp.log(A)
    lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA, nb_classes)

    post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)

    mpm_seg = np.argmax(post_marginals_probas, axis=1)

    if H is not None:
        e = np.count_nonzero(mpm_seg != H) / H.shape[0]
        print("Error in MPM segmentation", e)
    else:
        e = None

    return np.asarray(mpm_seg), e
