'''
Maximum likelihood learning and Forward Backward estimation for the
Deep Semi Pairwise Markov Chains
'''

import numpy as np
from functools import partial
import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from jax.experimental.host_callback import id_print, id_tap

import optax
import haiku as hk

from utils import (jax_loggauss, vmap_jax_loggauss, vmap_jax_dot)
from spmc_fb_and_posterior import (jax_log_forward_backward,
    jax_get_post_marginals_probas, jax_get_post_pair_marginals_probas,
    jax_compute_llkh)

def pretrain_networks(X, type_, the_net, the_net_params, 
    pre_seg, AA_init, mmeans_init, sstds_init,
    nb_classes=2, nb_channels=1):
    '''
    the_net is either A_net or meanvars_net
    '''
    @jax.jit
    def loss_A(the_net_params_unfrozen, the_net_params_frozen, batch):
        the_net_params = hk.data_structures.merge(the_net_params_unfrozen,
            the_net_params_frozen)
        pre_seg, X, outputs = batch

        X = X[:-1, :]
        T = len(X)
        r = jnp.squeeze(X)

        pre_seg_oh = jax.nn.one_hot(pre_seg[:-1], nb_classes - 1)
        h_times_r = jnp.reshape(
            jnp.rollaxis(vmap_jax_dot(pre_seg_oh[..., None], r[:, None]), 0, 2),
                        (T, - 1))
        in_ = jnp.concatenate([jnp.reshape(pre_seg_oh, (T, -1)),
                         h_times_r,
                         jnp.reshape(r, (T, -1))
                        ], axis=1)
        tmp = jnp.exp(the_net.apply(the_net_params, None, in_))
        tmp = tmp / jnp.sum(tmp, axis=1, keepdims=True)
        # avoid invalid values in log during the optimization
        tmp = jnp.where(tmp < 1e-5, 1e-5, tmp)
        tmp = jnp.where(tmp > 0.99999, 0.99999, tmp)
        return -np.mean(jnp.log(tmp) * outputs)

    @jax.jit
    def loss_meanvars(the_net_params_unfrozen, the_net_params_frozen, batch):
        the_net_params = hk.data_structures.merge(the_net_params_unfrozen,
            the_net_params_frozen)
        pre_seg, X, outputs = batch

        X = X[:-1, :]
        T = len(X)
        r = jnp.squeeze(X)

        pre_seg_oh = jax.nn.one_hot(pre_seg[:-1], nb_classes - 1)
        h_times_r = jnp.reshape(
            jnp.rollaxis(vmap_jax_dot(pre_seg_oh[..., None], r[:, None]), 0, 2),
                        (T, - 1))
        in_ = jnp.concatenate([jnp.reshape(pre_seg_oh, (T, -1)),
                         h_times_r,
                         jnp.reshape(r, (T, -1))
                        ], axis=1)
        meanvars = the_net.apply(the_net_params, None, in_)
        means = meanvars[..., :nb_channels]
        vars = jnp.square(meanvars[..., nb_channels:])
        return (jnp.mean(jnp.square(means - outputs[0]))
                +jnp.mean(jnp.square(vars - outputs[1])))
    
    T = len(X)
    pre_seg = jnp.asarray(pre_seg)

    if type_ == "meanvars":
        loss = loss_meanvars

        # NOTE choose the biggest values that will not make the optim problem
        # explode
        alpha = 0.01
        num_epochs = 500

        train_outputs = jnp.stack([mmeans_init[pre_seg[1:], :, np.arange(1,T)],
            jnp.square(sstds_init[pre_seg[1:], :, np.arange(1,T)])],axis=0)

    elif type_ == "A":
        loss = loss_A

        alpha = 0.001
        num_epochs = 500
        train_outputs = AA_init[pre_seg[:-1], :,
                                np.arange(T-1)]

    # find the frozen and unfrozen parameters of the network we want to
    # pretrain
    the_net_params_unfrozen, the_net_params_frozen = \
        hk.data_structures.partition(
        lambda m, n, p: m != "frozen_layer", the_net_params)

    opt = optax.adam(alpha)
    opt_state_the_net_params_unfrozen = opt.init(the_net_params_unfrozen)

    loss_grad_fun = jax.grad(loss, argnums=0)

    batch = [pre_seg, X, train_outputs] #[train_inputs, train_outputs]

    print("Start pretraining: ", num_epochs, "epoches")

    print_rate = 100
    def print_e(arg, transform):
        e, loss_val = arg
        print("Epoch", e, loss_val)
    def scan_on_epoch_loop(carry, e):
        the_net_params_unfrozen, opt_state_the_net_params_unfrozen = carry
        loss_grad_value = loss_grad_fun(the_net_params_unfrozen,
            the_net_params_frozen, batch)
        loss_grad_value, opt_state_the_net_params_unfrozen = opt.update(
            loss_grad_value, opt_state_the_net_params_unfrozen)
        the_net_params_unfrozen = optax.apply_updates(the_net_params_unfrozen,
            loss_grad_value)

        train_loss = loss(the_net_params_unfrozen, the_net_params_frozen, batch)

        e = jax.lax.cond(
            e % print_rate == 0,
            lambda _: id_tap(print_e, (e, train_loss), result=e),
            lambda _: e,
            operand=None
        )

        return (the_net_params_unfrozen, opt_state_the_net_params_unfrozen), e

    carry = the_net_params_unfrozen, opt_state_the_net_params_unfrozen
    (the_net_params_unfrozen, opt_state_the_net_params_unfrozen), _ = \
        jax.lax.scan(
            scan_on_epoch_loop,
            carry,
            jnp.arange(0, num_epochs)
    )

    the_net_params = hk.data_structures.merge(the_net_params_unfrozen,
        the_net_params_frozen)

    return the_net_params

@partial(jax.jit, static_argnums=(0, 1, 4))
def reconstruct_A(T, A_net, A_net_params, X, nb_classes):
    r = jnp.squeeze(X[:T-1])

    def scan_h_t_1(carry, h_t_1):
        A_net_params = carry
        # one hot encode, note nb_classes -1 to have a [0, 0] vector
        h_t_1_oh = jax.nn.one_hot(h_t_1, nb_classes - 1, axis=-1)
        h_t_1_vec = jnp.full((T - 1), h_t_1)
        h_t_1_vec_oh = jax.nn.one_hot(h_t_1_vec, nb_classes - 1, axis=-1)
        # NOTE always keep T - 1 on first dimension
        # Next is : the dot product we need then roll T (brought by r) on first
        # axis, then reshape and then stack to form the correct input vector
        h_times_r = jnp.reshape(
            jnp.rollaxis(vmap_jax_dot(h_t_1_vec_oh[..., None], r[:, None]), 0, 2),
                        (T - 1, - 1))
        in_ = jnp.concatenate([jnp.reshape(h_t_1_vec_oh, (T - 1, -1)),
                         h_times_r,
                         jnp.reshape(r, (T - 1, -1))
                        ], axis=1)
        tmp = jnp.squeeze(jnp.exp(A_net.apply(A_net_params, None, in_)))
        tmp = tmp.T
        Atmp = tmp / jnp.sum(tmp, axis=0, keepdims=True)
        return A_net_params, Atmp
    carry = A_net_params

    # we get back the stack of samples (ie, the stack of second position return
    # of the scan function)
    _, A = jax.lax.scan(scan_h_t_1, carry, jnp.arange(nb_classes))

    A = jnp.where(A < 1e-5, 1e-5, A)
    A = jnp.where(A > 0.99999, 0.99999, A)

    return A

@partial(jax.jit, static_argnums=(0, 1, 4, 5))
def reconstruct_means_stds(T, meanvars_net, meanvars_net_params,
    X, nb_classes, nb_channels):

    X = jnp.concatenate([jnp.asarray(X[1:2]), jnp.asarray(X[:-1])], axis=0)
    # NOTE here we have to bother to form a means[0] and stds[0]
    # but what we do is just take  a random value and put it first
    r = jnp.squeeze(X)
    def scan_h_t(carry, h_t):
        meanvars_net_params = carry
        h_t_oh = jax.nn.one_hot(h_t, nb_classes - 1, axis=-1)
        h_t_vec = jnp.full((T), h_t)
        h_t_vec_oh = jax.nn.one_hot(h_t_vec, nb_classes - 1, axis=-1)
        # NOTE always keep T on first dimension
        # Next is : the dot product we need then roll T (brought by r) on first
        # axis, then reshape and then stack to form the correct input vector
        h_times_r = jnp.reshape(
            jnp.rollaxis(vmap_jax_dot(h_t_vec_oh[..., None], r[:, None]), 0, 2),
                        (T, - 1))
        in_ = jnp.concatenate([jnp.reshape(h_t_vec_oh, (T, -1)),
                         h_times_r,
                         jnp.reshape(r, (T, -1))
                        ], axis=1)
        meanvars = meanvars_net.apply(meanvars_net_params, None, in_)

        return (meanvars_net_params), meanvars
    carry = meanvars_net_params

    _, meanvars = jax.lax.scan(
        scan_h_t,
        carry,
        jnp.arange(nb_classes)
    )

    means = jnp.moveaxis(meanvars[..., :nb_channels], -1, 1)
    stds = jnp.moveaxis(jnp.sqrt(jnp.square(meanvars[..., nb_channels:])), -1,
    1)
    stds = jnp.where(stds < 1e-5, 1e-5, stds)

    return means, stds 

def gradient_llkh(T, X, key, nb_iter, A_init, means_init,
    stds_init, pre_seg, A_sig_params_init, norm_params_init,
    H_gt=None, with_output_constraint=True, with_pretrain=True,
    alpha=0.01, with_gpu=False, nb_classes=2, nb_channels=1):
    cpus = jax.devices("cpu")
    try:
        gpus = jax.devices("gpu")
        gpu_available = True
    except RuntimeError:
        gpu_available = False
    if not with_gpu:
        gpu_available = False
        
    nb_neurons_on_last_hidden = 3
    nb_neurons_on_input = 3
    nb_neurons_on_last_hidden = ((nb_classes - 1) + (nb_classes - 1) * nb_channels +
                            nb_channels)
    nb_neurons_on_input = ((nb_classes - 1) + (nb_classes - 1) * nb_channels +
                            nb_channels)

    def A_net_def(x):
        mlp = hk.Sequential([
            hk.Flatten(), 
            hk.Linear(100), jax.nn.relu,
            hk.Linear(nb_neurons_on_last_hidden),
            hk.Linear(nb_classes, name="frozen_layer")
        ])
        return mlp(x)

    dummy_in_shape = jnp.ones((T, nb_neurons_on_input))
    A_net = hk.transform(A_net_def)
    key, subkey = jax.random.split(key)
    A_net_params = A_net.init(subkey, dummy_in_shape)

    def meanvars_net_def(x):
        mlp = hk.Sequential([
            hk.Flatten(),
            hk.Linear(100), jax.nn.relu,
            hk.Linear(nb_neurons_on_last_hidden),
            hk.Linear(2 * nb_channels, name="frozen_layer")
            ])
        return mlp(x)

    dummy_in_shape = jnp.ones((T, nb_neurons_on_input))
    meanvars_net = hk.transform(meanvars_net_def)
    key, subkey = jax.random.split(key)
    meanvars_net_params = meanvars_net.init(subkey, dummy_in_shape)

    # otherwise our constants are
    # treated as Traced Shaped Array NOTE
    @partial(jax.jit, static_argnums=(5, 6)) 
    def compute_llkh_wrapper_one_batch(A_net_params_unfrozen, A_net_params_frozen,
        meanvars_net_params_unfrozen, meanvars_net_params_frozen, X,
        nb_classes, nb_channels):
        T = len(X)
        A_net_params = hk.data_structures.merge(A_net_params_unfrozen,
            A_net_params_frozen)
        meanvars_net_params = hk.data_structures.merge(
            meanvars_net_params_unfrozen, meanvars_net_params_frozen)

        means, stds = reconstruct_means_stds(T, meanvars_net,
            meanvars_net_params, X, nb_classes, nb_channels)
        lX_pdf = jnp.stack([jnp.sum(vmap_jax_loggauss(
            X, jnp.array([means[i, c] for c in range(nb_channels)]),
                jnp.array([stds[i, c] for c in range(nb_channels)])), axis=0)
                            for i in range(nb_classes)], axis=0)

        A = reconstruct_A(T, A_net, A_net_params, X,
            nb_classes)
        lA = jnp.log(A)

        # NOTE return minus the llkh because we want to maximize
        return -jax_compute_llkh(T, lX_pdf, lA, nb_classes, nb_channels)

    vmap_compute_llkh_wrapper_one_batch = jax.vmap(
        compute_llkh_wrapper_one_batch,
        in_axes=(None, None, None, None, 0, None, None)
    )

    # otherwise our constants are
    # treated as Traced Shaped Array NOTE
    @partial(jax.jit, static_argnums=(5, 6)) 
    def compute_llkh_wrapper(A_net_params_unfrozen, A_net_params_frozen,
        meanvars_net_params_unfrozen, meanvars_net_params_frozen, X,
        nb_classes, nb_channels):
        return jnp.mean(
            vmap_compute_llkh_wrapper_one_batch(A_net_params_unfrozen,
            A_net_params_frozen, meanvars_net_params_unfrozen,
            meanvars_net_params_frozen, X, nb_classes, nb_channels)
        )

    Llkh_grad_A_net_params_unfrozen = jax.grad(compute_llkh_wrapper, argnums=0)
    Llkh_grad_meanvars_net_params_unfrozen = jax.grad(compute_llkh_wrapper,
        argnums=2)

    if with_output_constraint:
    # next we set the last layer parameters frozen the the equivalent network
    # defined by the non deep parameters
    # NOTE the following fixes the order of input in networks
    # do not invert the order of inputs in the networks since they are
    # linked with the equations for equivalency of the last layer with the
    # related non deep model
    # NOTE The following also fixes that fact that the output of the meanvars
    # network is the variance and not the stds
    # NOTE The square root is taken on c thus we put a square each time the
    # network is applied. This way we have a positivity constraint on the var
        print("Setting the last layer with non deep values")
        wei = np.zeros((nb_neurons_on_last_hidden, 2, nb_channels))
        # means
        offset = 0
        for i in range(nb_classes - 1):
            wei[i, 0] = (norm_params_init[1][i + 1, :] - norm_params_init[1][0, :])
        offset = nb_classes - 1
        for i in range(nb_classes - 1):
            for j in range(nb_channels):
                wei[offset + i * nb_channels + j, 0] = (norm_params_init[0][i + 1, j] -
                            norm_params_init[0][0, j])
        offset += (nb_classes - 1) * nb_channels
        for j in range(nb_channels):
            wei[offset + j, 0] = norm_params_init[0][0, j]
        # stds
        for i in range(nb_classes - 1):
            wei[i, 1] = (jnp.sqrt(norm_params_init[2][i + 1, :]) -
                jnp.sqrt(norm_params_init[2][0, :]))
        # the rest of wei[:, 1] remains at 0
        wei = wei.reshape((nb_neurons_on_last_hidden, -1)) # row major !
        b = np.zeros((2, nb_channels))
        # means
        b[0] = norm_params_init[1][0, :]
        # stds
        b[1] = np.sqrt(norm_params_init[2][0, :])
        b = b.reshape((2 * nb_channels)) # row major (all the means then all
        # the stds)

        meanvars_net_params = \
            hk.data_structures.to_mutable_dict(meanvars_net_params)
        meanvars_net_params['frozen_layer'] = \
            hk.data_structures.to_immutable_dict({'w':wei, 'b':b})

        A_sig_params_0_init, A_sig_params_1_init = A_sig_params_init
        wei = np.zeros((nb_neurons_on_last_hidden, nb_classes))
        offset = 0
        for i in range(nb_classes - 1):
            wei[i] = (A_sig_params_1_init[i + 1, :] -
                        A_sig_params_1_init[0, :])
        offset = nb_classes - 1
        for i in range(nb_classes - 1):
            for j in range(nb_channels):
                wei[offset + i * nb_channels + j] = \
                        (A_sig_params_0_init[i + 1, :, j] -
                            A_sig_params_0_init[0, :, j])
        offset += (nb_classes - 1) * nb_channels
        for j in range(nb_channels):
            wei[offset + j] = A_sig_params_0_init[0, :, j]
        b = np.zeros((nb_classes))
        b = A_sig_params_1_init[0, :]

        A_net_params = \
            hk.data_structures.to_mutable_dict(A_net_params)
        A_net_params['frozen_layer'] = \
            hk.data_structures.to_immutable_dict({'w':wei, 'b':b})

    # Set the new hidden layer to implement identity (by forcing to 0 +
    # residual)
    wei = np.zeros((nb_neurons_on_input, 100))
    wei = jnp.asarray(wei)
    key, subkey = jax.random.split(key)
    wei += 0.01 * jax.random.normal(subkey, shape=wei.shape)
    b = jnp.zeros((100))
    A_net_params = \
        hk.data_structures.to_mutable_dict(A_net_params)
    A_net_params['linear'] = \
        hk.data_structures.to_immutable_dict({'w':wei, 'b':b})
    wei = np.zeros((100, nb_neurons_on_last_hidden))
    wei = jnp.asarray(wei)
    key, subkey = jax.random.split(key)
    wei += 0.01 * jax.random.normal(subkey, shape=wei.shape)
    b = jnp.zeros((nb_neurons_on_last_hidden))
    A_net_params = \
        hk.data_structures.to_mutable_dict(A_net_params)
    A_net_params['linear_1'] = \
        hk.data_structures.to_immutable_dict({'w':wei, 'b':b})
    wei = np.zeros((nb_neurons_on_input, 100))
    wei = jnp.asarray(wei)
    key, subkey = jax.random.split(key)
    wei += 0.01 * jax.random.normal(subkey, shape=wei.shape)
    b = jnp.zeros((100))
    meanvars_net_params = \
        hk.data_structures.to_mutable_dict(meanvars_net_params)
    meanvars_net_params['linear'] = \
        hk.data_structures.to_immutable_dict({'w':wei, 'b':b})
    wei = np.zeros((100, nb_neurons_on_last_hidden))
    wei = jnp.asarray(wei)
    key, subkey = jax.random.split(key)
    wei += 0.01 * jax.random.normal(subkey, shape=wei.shape)
    b = jnp.zeros((nb_neurons_on_last_hidden))
    meanvars_net_params = \
        hk.data_structures.to_mutable_dict(meanvars_net_params)
    meanvars_net_params['linear_1'] = \
        hk.data_structures.to_immutable_dict({'w':wei, 'b':b})

    if with_pretrain and pre_seg is not None:
        print("Pretraining by backpropagation")
        if gpu_available:
            X_gpu = jax.device_put(X, gpus[0])
            # put the next init arguments on GPU because non deep models have
            # been trained on CPUs
            A_init_gpu = jax.device_put(A_init, gpus[0])
            means_init_gpu = jax.device_put(means_init, gpus[0])
            stds_init_gpu = jax.device_put(stds_init, gpus[0])
            pre_seg_gpu = jax.device_put(pre_seg, gpus[0])

            meanvars_net_params = pretrain_networks(X_gpu, "meanvars",
                meanvars_net, meanvars_net_params,
                pre_seg_gpu[:len(X)], A_init_gpu,
                means_init_gpu, stds_init_gpu,
                nb_channels=nb_channels, nb_classes=nb_classes)

            A_net_params = pretrain_networks(X_gpu, 
            "A", A_net, A_net_params, pre_seg_gpu,
                A_init_gpu, means_init_gpu, stds_init_gpu,
                nb_channels=nb_channels, nb_classes=nb_classes)
        else:
            X_cpu = jax.device_put(X, cpus[0])
            meanvars_net_params = pretrain_networks(X_cpu, "meanvars",
                meanvars_net, meanvars_net_params,
                pre_seg[:len(X)], A_init,
                means_init, stds_init,
                nb_channels=nb_channels, nb_classes=nb_classes)

            A_net_params = pretrain_networks(X_cpu, 
            "A", A_net, A_net_params, pre_seg,
                A_init, means_init, stds_init,
                nb_channels=nb_channels, nb_classes=nb_classes)

        # NOTE Back to CPU
        meanvars_net_params = jax.device_put(meanvars_net_params, cpus[0])
        A_net_params = jax.device_put(A_net_params, cpus[0])

    # next we define the frozen and unfrozen weights with haiku partition
    A_net_params_unfrozen, A_net_params_frozen = \
        hk.data_structures.partition(
        lambda m, n, p: m != "frozen_layer", A_net_params)
    meanvars_net_params_unfrozen, meanvars_net_params_frozen = \
        hk.data_structures.partition(
        lambda m, n, p: m != "frozen_layer", meanvars_net_params)

    ################
    ## MAIN  LOOP ##
    ################

    nb_batches = 1
    T = len(X)
    X_batched = jnp.reshape(X[..., None], (nb_batches, -1, nb_channels)) 

    opt = optax.adam(alpha)

    opt_state_A_net_params_unfrozen = opt.init(A_net_params_unfrozen)
    opt_state_meanvars_net_params_unfrozen = \
        opt.init(meanvars_net_params_unfrozen)

    def scan_on_iter_loop(carry, k):
        (A_net_params_unfrozen, meanvars_net_params_unfrozen,
            opt_state_A_net_params_unfrozen,
            opt_state_meanvars_net_params_unfrozen) = carry


        llkh_grad_A_net_params_unfrozen = Llkh_grad_A_net_params_unfrozen(
            A_net_params_unfrozen, A_net_params_frozen,
            meanvars_net_params_unfrozen, meanvars_net_params_frozen,
            X_batched, nb_classes, nb_channels)
        llkh_grad_A_net_params_unfrozen, opt_state_A_net_params_unfrozen = \
            opt.update(llkh_grad_A_net_params_unfrozen,
                opt_state_A_net_params_unfrozen)

        llkh_grad_meanvars_net_params_unfrozen = \
            Llkh_grad_meanvars_net_params_unfrozen(
            A_net_params_unfrozen, A_net_params_frozen,
            meanvars_net_params_unfrozen, meanvars_net_params_frozen,
            X_batched, nb_classes, nb_channels)
        (llkh_grad_meanvars_net_params_unfrozen,
        opt_state_meanvars_net_params_unfrozen) = \
            opt.update(llkh_grad_meanvars_net_params_unfrozen,
                opt_state_meanvars_net_params_unfrozen)

        # make all the updates
        A_net_params_unfrozen = optax.apply_updates(
            A_net_params_unfrozen, llkh_grad_A_net_params_unfrozen)
        meanvars_net_params_unfrozen = optax.apply_updates(
        meanvars_net_params_unfrozen, llkh_grad_meanvars_net_params_unfrozen)

        # compute likelihood to see the progress
        llkh = compute_llkh_wrapper_one_batch(A_net_params_unfrozen, A_net_params_frozen,
            meanvars_net_params_unfrozen, meanvars_net_params_frozen, X,
            nb_classes, nb_channels)

        k = id_print(k, what="####### ITERATION #######")
        llkh = id_print(llkh / T, what="loglikelihood")

        return (A_net_params_unfrozen, meanvars_net_params_unfrozen,
            opt_state_A_net_params_unfrozen, opt_state_meanvars_net_params_unfrozen), k

    carry = (A_net_params_unfrozen, meanvars_net_params_unfrozen,
        opt_state_A_net_params_unfrozen, opt_state_meanvars_net_params_unfrozen)
    (A_net_params_unfrozen, meanvars_net_params_unfrozen,
        opt_state_A_net_params_unfrozen,
        opt_state_meanvars_net_params_unfrozen), _ = jax.lax.scan(
        scan_on_iter_loop,
        carry,
        jnp.arange(0, nb_iter)
    )

    # final parameters
    A_net_params = hk.data_structures.merge(A_net_params_unfrozen,
        A_net_params_frozen)
    meanvars_net_params = hk.data_structures.merge(
        meanvars_net_params_unfrozen, meanvars_net_params_frozen)
    means, stds = reconstruct_means_stds(T, meanvars_net,
        meanvars_net_params, X,
        nb_channels=nb_channels, nb_classes=nb_classes)
    A = reconstruct_A(T, A_net, A_net_params, X,
        nb_classes)

    return (A, means, stds, (A_net, A_net_params),
            (meanvars_net, meanvars_net_params))

def MPM_segmentation(T, X, A_net, A_net_params,
    meanvars_net, meanvars_net_params,
    H=None, nb_channels=1, nb_classes=2):
    if meanvars_net_params is not None and meanvars_net is not None:
        means, stds = reconstruct_means_stds(T, meanvars_net,
            meanvars_net_params, X,
            nb_classes=nb_classes, nb_channels=nb_channels)
    if A_net_params is not None and A_net is not None:
        A = reconstruct_A(T, A_net, A_net_params, X,
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
