'''
Maximum likelihood learning and Forward Backward estimation for the
Deep Semi Pairwise Markov Chains
'''

import numpy as np
from functools import partial
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from jax.experimental.host_callback import id_print, id_tap

import optax
import haiku as hk

from utils import (jax_loggauss, vmap_jax_loggauss,
    vmap_jax_dot)
from spmc_fb_and_posterior import (jax_log_forward_backward,
    jax_get_post_marginals_probas, jax_get_post_pair_marginals_probas,
    jax_compute_llkh)

def pretrain_networks(X, type_, the_net, the_net_params, 
    pre_seg, AA_init, mmeans_init, sstds_init):
    '''
    the_net is either A_net or meanvars_net
    '''
    @jax.jit
    def loss_A(the_net_params_unfrozen, the_net_params_frozen,
            train_inputs, train_outputs, pre_seg):
        the_net_params = hk.data_structures.merge(the_net_params_unfrozen,
            the_net_params_frozen)

        T = len(train_inputs)
        r = jnp.squeeze(train_inputs)

        inputs = jnp.stack([pre_seg, pre_seg * r, r], axis=1)

        pred_w2 = jnp.squeeze(jnp.exp(
                            -the_net.apply(the_net_params, None, inputs)
                            ))
        pred_w2 = pred_w2 / (1 + pred_w2)
        pred_w2 = jnp.where(pred_w2 < 1e-5, 1e-5, pred_w2)
        pred_w2 = jnp.where(pred_w2 > 0.99999, 0.99999, pred_w2)
        return -jnp.mean(jnp.log(pred_w2) * train_outputs +
                jnp.log(1 - pred_w2) * (1 - train_outputs))

    vmap_loss_A_one_batch = jax.vmap(loss_A,
        in_axes=(None, None, 0, 0, 0))

    @jax.jit
    def loss_meanvars(the_net_params_unfrozen, the_net_params_frozen,
        train_inputs, train_outputs, pre_seg):
        the_net_params = hk.data_structures.merge(the_net_params_unfrozen,
            the_net_params_frozen)

        T = len(train_inputs)
        r = jnp.squeeze(train_inputs)

        inputs = jnp.stack([pre_seg, pre_seg * r, r], axis=1)
        meanvars = the_net.apply(the_net_params, None, inputs)
        means = meanvars[..., 0]
        vars = jnp.square(meanvars[..., 1])
        return (jnp.mean(jnp.sum(jnp.square(means - train_outputs[0])))
                +jnp.mean(jnp.sum(jnp.square(vars - train_outputs[1]))))

    vmap_loss_meanvars_one_batch = jax.vmap(loss_meanvars,
        in_axes=(None, None, 0, 1, 0)) # NOTE parallelize on the 2nd axis
        # of train_outputs argument

    nb_batches = 64
    T = len(X)

    if type_ == "meanvars":
        vmap_loss_one_batch = vmap_loss_meanvars_one_batch 

        alpha = 0.01
        num_epochs = 501

        # Cutting of the seq must be done at each batch
        train_inputs = jnp.reshape(X, (nb_batches, -1)) 
        train_inputs = train_inputs[:, :-1]
        train_outputs = jnp.stack([mmeans_init[pre_seg[:], jnp.arange(T)],
            jnp.square(sstds_init[pre_seg[:], jnp.arange(T)])], axis=0)
        train_outputs = jnp.reshape(train_outputs, (2, nb_batches, -1))
        train_outputs = train_outputs[:, :, 1:]
        pre_seg = jnp.reshape(pre_seg[..., None], (nb_batches, -1))
        pre_seg = pre_seg[:, 1:]


    elif type_ == "A":
        vmap_loss_one_batch = vmap_loss_A_one_batch

        alpha = 0.001
        num_epochs = 201
        train_inputs = jnp.reshape(X, (nb_batches, -1)) 
        train_inputs = train_inputs[:, :-1]
        train_outputs = AA_init[pre_seg[:], jnp.ones((T,)).astype(int), jnp.arange(T)]
        train_outputs = jnp.reshape(train_outputs, (nb_batches, -1))
        train_outputs = train_outputs[:, :-1]
        pre_seg = jnp.reshape(pre_seg[..., None], (nb_batches, -1))
        pre_seg = pre_seg[:, :-1]


    @jax.jit
    def loss(the_net_params_unfrozen, the_net_params_frozen,
        train_inputs, train_outputs, pre_seg):
        return jnp.mean(vmap_loss_one_batch(the_net_params_unfrozen,
            the_net_params_frozen,
            train_inputs, train_outputs, pre_seg))

    # find the frozen and unfrozen parameters of the network we want to
    # pretrain
    the_net_params_unfrozen, the_net_params_frozen = \
        hk.data_structures.partition(
        lambda m, n, p: m != "frozen_layer", the_net_params)

    opt = optax.adam(alpha)
    opt2 = optax.adam(alpha / 1000)

    opt_state_the_net_params_unfrozen = opt.init(the_net_params_unfrozen)
    loss_grad_fun = jax.grad(loss, argnums=0)


    print("Start pretraining: ", num_epochs, "epoches")

    print_rate = 100
    def print_e(arg, transform):
        e, loss_val = arg
        print("Epoch", e, loss_val)
    def scan_on_epoch_loop(carry, e):
        (the_net_params_unfrozen, opt_state_the_net_params_unfrozen) = carry

        loss_grad_value = loss_grad_fun(the_net_params_unfrozen,
            the_net_params_frozen, train_inputs,
            train_outputs, pre_seg)
        loss_grad_value, opt_state_the_net_params_unfrozen = opt.update(
            loss_grad_value, opt_state_the_net_params_unfrozen)
        the_net_params_unfrozen = optax.apply_updates(the_net_params_unfrozen,
            loss_grad_value)

        train_loss = loss(the_net_params_unfrozen, the_net_params_frozen,
            train_inputs, train_outputs, pre_seg)

        e = jax.lax.cond(
            e % print_rate == 0,
            lambda _: id_tap(print_e, (e, train_loss), result=e),
            lambda _: e,
            operand=None
        )

        return (the_net_params_unfrozen, opt_state_the_net_params_unfrozen), e

    carry = (the_net_params_unfrozen, opt_state_the_net_params_unfrozen)
    (the_net_params_unfrozen, opt_state_the_net_params_unfrozen), _ = jax.lax.scan(
            scan_on_epoch_loop,
            carry,
            jnp.arange(0, num_epochs)
    )

    the_net_params = hk.data_structures.merge(the_net_params_unfrozen,
        the_net_params_frozen)

    print("End pretraining")

    return the_net_params

@partial(jax.jit, static_argnums=(0, 1))
def reconstruct_A(T, A_net, A_net_params, X):
    r = jnp.squeeze(X[:T-1])
    def scan_h_t_1(carry, h_t_1):
        A_net_params = carry
        h_vec = jnp.full((T - 1, ), h_t_1)
        Atmp = jnp.squeeze(jnp.exp(-A_net.apply(A_net_params, None,
                    jnp.array([h_vec, r * h_vec, r]).T)))
        Atmp = Atmp / (1 + Atmp)
        return A_net_params, jnp.array([1 - Atmp, Atmp])
    carry = A_net_params

    _, A = jax.lax.scan(scan_h_t_1, carry, jnp.arange(2))

    A = jnp.where(A < 1e-5, 1e-5, A)
    A = jnp.where(A > 0.99999, 0.99999, A)

    return A

@partial(jax.jit, static_argnums=(0, 1))
def reconstruct_means_stds(T, meanvars_net, meanvars_net_params, X):
    stds = jnp.empty((2, T))
    means = jnp.empty((2, T))

    X = jnp.concatenate([jnp.array([X[1]]), X[:-1]])
    r = jnp.squeeze(X)

    def scan_h_t(carry, h_t):
        meanvars_net_params = carry
        h_vec = jnp.full((T, ), h_t)
        meanvars = meanvars_net.apply(meanvars_net_params, None,
                jnp.array([h_vec, r * h_vec, r]).T)
        return meanvars_net_params, meanvars

    carry = meanvars_net_params
    _, meanvars = jax.lax.scan(
        scan_h_t,
        carry,
        jnp.arange(2)
    )

    means = meanvars[..., 0]
    stds = jnp.sqrt(jnp.square(meanvars[..., 1]))
    stds = jnp.where(stds < 1e-5, 1e-5, stds)

    return means, stds

def gradient_llkh(T, X, key, nb_iter, A_init, means_init,
    stds_init, pre_seg, A_sig_params_init, norm_params_init,
    H_gt=None, with_output_constraint=True, with_pretrain=True,
    alpha=0.01, with_gpu=False):

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

    def A_net_def(x):
        mlp1 = hk.Sequential([
            hk.Flatten(), 
            hk.Linear(100), jax.nn.relu,
            hk.Linear(nb_neurons_on_last_hidden),
        ])
        mlp2 = hk.Sequential([
            hk.Linear(1, name="frozen_layer")
            ])
        return mlp2(mlp1(x) + x)

    dummy_in_shape = jnp.ones((T, nb_neurons_on_input))
    A_net = hk.transform(A_net_def)
    key, subkey = jax.random.split(key)
    A_net_params = A_net.init(subkey, dummy_in_shape)

    def meanvars_net_def(x):
        mlp1 = hk.Sequential([
            hk.Flatten(), 
            hk.Linear(100), jax.nn.relu,
            hk.Linear(nb_neurons_on_last_hidden),
        ])
        mlp2 = hk.Sequential([
            hk.Linear(2, name="frozen_layer")
            ])
        return mlp2(mlp1(x) + x)

    dummy_in_shape = jnp.ones((T, nb_neurons_on_input))
    meanvars_net = hk.transform(meanvars_net_def)
    key, subkey = jax.random.split(key)
    meanvars_net_params = meanvars_net.init(subkey, dummy_in_shape)

    @jax.jit
    def compute_llkh_wrapper_one_batch(A_net_params_unfrozen, A_net_params_frozen,
        meanvars_net_params_unfrozen, meanvars_net_params_frozen, X):
        T = len(X)
        A_net_params = hk.data_structures.merge(A_net_params_unfrozen,
            A_net_params_frozen)
        meanvars_net_params = hk.data_structures.merge(
            meanvars_net_params_unfrozen, meanvars_net_params_frozen)

        means, stds = reconstruct_means_stds(T, meanvars_net,
            meanvars_net_params, X)
        
        lX_pdf = jnp.stack([jax_loggauss(
            X, means[i], stds[i]) for i in range(2)],
            axis=0)

        A = reconstruct_A(T, A_net, A_net_params, X)
        lA = jnp.log(A)

        # NOTE return minus the llkh because we want to maximize
        return -jax_compute_llkh(T, lX_pdf, lA)

    vmap_compute_llkh_wrapper_one_batch = jax.vmap(
        compute_llkh_wrapper_one_batch,
        in_axes=(None, None, None, None, 0)
    )

    @jax.jit
    def compute_llkh_wrapper(A_net_params_unfrozen, A_net_params_frozen,
        meanvars_net_params_unfrozen, meanvars_net_params_frozen, X):
        return jnp.mean(
            vmap_compute_llkh_wrapper_one_batch(A_net_params_unfrozen,
            A_net_params_frozen, meanvars_net_params_unfrozen,
            meanvars_net_params_frozen, X)
        )

    Llkh_grad_A_net_params_unfrozen = jax.grad(compute_llkh_wrapper, argnums=0)
    Llkh_grad_meanvars_net_params_unfrozen = jax.grad(compute_llkh_wrapper,
        argnums=2)

    if with_output_constraint:
    # next we set the last layer parameters frozen the the equivalent network
    # defined by the non deep parameters
        print("Setting the last layer with non deep values")
        wei = np.random.randn(3, 2) * 0.01
        b = np.zeros((2))
        wei[2, 1] = 0
        wei[1, 1] = 0
        wei[0, 1] = (jnp.sqrt(norm_params_init[2][1]) -
            jnp.sqrt(norm_params_init[2][0]))
        wei[0, 0] = norm_params_init[1][1] - norm_params_init[1][0]
        wei[1, 0] = norm_params_init[0][1] - norm_params_init[0][0]
        wei[2, 0] = norm_params_init[0][0]
        b[1] = np.sqrt(norm_params_init[2][0])
        b[0] = norm_params_init[1][0]

        meanvars_net_params = \
            hk.data_structures.to_mutable_dict(meanvars_net_params)
        meanvars_net_params['frozen_layer'] = \
            hk.data_structures.to_immutable_dict({'w':wei, 'b':b})

        meanvars_net_params = \
            hk.data_structures.to_mutable_dict(meanvars_net_params)
        meanvars_net_params['frozen_layer'] = \
            hk.data_structures.to_immutable_dict({'w':wei, 'b':b})

        wei = np.random.randn(3, 1) * 0.01
        b = np.zeros((1))
        wei[0] = A_sig_params_init[1][1] - A_sig_params_init[1][0]
        wei[1] = A_sig_params_init[0][1] - A_sig_params_init[0][0]
        wei[2] = A_sig_params_init[0][0]
        b[0] = A_sig_params_init[1][0]
        b = jnp.asarray(b)
        wei = jnp.asarray(wei)

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
        else:
            X_gpu = X

        try:
            meanvars_net_params = pretrain_networks(X_gpu, "meanvars",
                meanvars_net, meanvars_net_params,
                pre_seg_gpu[:len(X)], A_init_gpu,
                means_init_gpu, stds_init_gpu)
        except RuntimeError:
            # probably RESOURCE_EXHAUSTED on the GPU 
            print("Problem with pretraining on GPU, not enough memory ?")
            X_cpu = jax.device_put(X, cpus[0])
            meanvars_net_params = pretrain_networks(X_cpu, "meanvars",
                meanvars_net, meanvars_net_params,
                pre_seg[:len(X)], A_init,
                means_init, stds_init)
        try:
            A_net_params = pretrain_networks(X_gpu, 
            "A", A_net, A_net_params, pre_seg_gpu,
                A_init_gpu, means_init_gpu, stds_init_gpu)
        except RuntimeError:
            # probably RESOURCE_EXHAUSTED on the GPU 
            print("Problem with pretraining on GPU, not enough memory ?")
            X_cpu = jax.device_put(X, cpus[0])
            A_net_params = pretrain_networks(X_cpu, 
            "A", A_net, A_net_params, pre_seg,
                A_init, means_init, stds_init)

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
    X_ori = X.copy()
    T = len(X_ori)
    X = jnp.reshape(X[..., None], (nb_batches, -1)) 

    opt = optax.adam(alpha)
    opt2 = optax.adam(alpha / 1000)

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
            X)
        llkh_grad_A_net_params_unfrozen, opt_state_A_net_params_unfrozen = \
            opt.update(llkh_grad_A_net_params_unfrozen,
                opt_state_A_net_params_unfrozen)

        llkh_grad_meanvars_net_params_unfrozen = \
            Llkh_grad_meanvars_net_params_unfrozen(
            A_net_params_unfrozen, A_net_params_frozen,
            meanvars_net_params_unfrozen, meanvars_net_params_frozen,
            X)
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
        llkh = compute_llkh_wrapper(A_net_params_unfrozen, A_net_params_frozen,
            meanvars_net_params_unfrozen, meanvars_net_params_frozen, X)
        llkh = - llkh

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

    # NOTE MERGE BACK
    A_net_params = hk.data_structures.merge(A_net_params_unfrozen,
        A_net_params_frozen)
    meanvars_net_params = hk.data_structures.merge(
        meanvars_net_params_unfrozen, meanvars_net_params_frozen)


    # final parameters
    T = len(X_ori)
    means, stds = reconstruct_means_stds(T, meanvars_net,
        meanvars_net_params, X_ori)
    A = reconstruct_A(T, A_net, A_net_params, X_ori)

    return (A, means, stds, (A_net, A_net_params),
            (meanvars_net, meanvars_net_params))
                    
def MPM_segmentation(T, X, A_net, A_net_params,
    meanvars_net, meanvars_net_params,
    H=None):
    means, stds = reconstruct_means_stds(T, meanvars_net,
        meanvars_net_params, X)
    A = reconstruct_A(T, A_net, A_net_params,
        X)
    lX_pdf = jnp.stack([jax_loggauss(
        X, means[i], stds[i]) for i in range(2)],
        axis=0)
    lA = jnp.log(A)
    lalpha, lbeta = jax_log_forward_backward(T, lX_pdf, lA)

    post_marginals_probas = jax_get_post_marginals_probas(lalpha, lbeta)

    mpm_seg = jnp.argmax(post_marginals_probas, axis=1)

    if H is not None:
        e = jnp.count_nonzero(mpm_seg != H) / H.shape[0]
        print("Error in MPM segmentation", e)
    else:
        e = None

    return jnp.asarray(mpm_seg), e
