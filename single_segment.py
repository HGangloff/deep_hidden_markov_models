'''
Code to use, on a single image, the models from
Generalized Pairwise and Triplet Markov Chains: example of a
deep parametrization for unsupervised signal processing
Hugo Gangloff, Katherine Morales and Yohan Petetin
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '' #NOTE NOTE NOTE
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import jax

sys.path.insert(1, './models/')
sys.path.insert(1, './utils/')

from hmcin import MPM_segmentation as hmcin_MPM_segmentation
from hmcin import EM as hmcin_EM
from spmc import MPM_segmentation as spmc_MPM_segmentation
from spmc import gradient_llkh as spmc_gradient_llkh
from pmc import MPM_segmentation as pmc_MPM_segmentation
from pmc import gradient_llkh as pmc_gradient_llkh
from dspmc import MPM_segmentation as dspmc_MPM_segmentation
from dspmc import gradient_llkh as dspmc_gradient_llkh
from dpmc import MPM_segmentation as dpmc_MPM_segmentation
from dpmc import gradient_llkh as dpmc_gradient_llkh

from chain_to_image_functions import chain_to_image, image_to_chain


def seg_image(img_path, img_name, key):
    np.seterr(all="warn")
    img = np.array(Image.open(img_path))
    img_ = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    Image.fromarray(
        (img_ * 255).astype(np.uint8)
    ).save("./figures/" + img_name + "_img.png")

    r, rr = 0.75, 1.5
    #r, rr, r3, r4 = 0.75, 1.5, 2.25, 3
    obs = [0.5]
    H = image_to_chain(img)[:]
    H[H == 255] = 2
    H[H == 128] = 1
    H[H == 100] = 3
    H[H == 156] = 4
    print(img.shape, H.shape)
    b = np.array([0., float(r), float(rr)])#, float(r3), float(r4)])
    for i in range(1, 256*256):
        obs.append(np.sin(b[H[i]] + obs[-1]) + np.random.randn()*0.5)
    X = np.asarray(obs).astype(np.float64)
    X = (X - np.amin(X)) / (np.amax(X) - np.amin(X))
    Image.fromarray(
        (chain_to_image(X) * 255).astype(np.uint8)
    ).save("./figures/" + img_name + "_X.png")
    T = len(H)

    #X = image_to_chain(img).astype(np.float64)
    #X = (X - np.amin(X)) / (np.amax(X) - np.amin(X))
    #T = len(X)

    if X.ndim == 1:
        X = X[:, None]

    nb_classes = 5
    nb_channels = 1

    # KMeans segmentation
    print("KMeans segmentation")
    kmeans_seg = KMeans(n_clusters=nb_classes).fit(X.reshape((-1, nb_channels))).labels_
    means = np.array([np.mean(X[kmeans_seg == h], axis=0) for h in range(nb_classes)])
    stds = np.array([np.std(X[kmeans_seg == h], axis=0) for h in range(nb_classes)])
    if means.ndim == 1:
        means = means[:, None]
        stds = stds[:, None]
    A = np.zeros((nb_classes, nb_classes))
    p0 = np.zeros((nb_classes))
    for t in range(1, T):
        A[kmeans_seg[t - 1], kmeans_seg[t]] += 1
        p0[kmeans_seg[t]] += 1
    p0 /= np.sum(p0)
    A[0] /= np.sum(A[0])
    A[1] /= np.sum(A[1])

    # HMCIN segmentation
    print("HMCIN segmentation")
    # Force CPU computation, nothing is batched here to go on GPU
    cpus = jax.devices("cpu")
    X_cpu = jax.device_put(X, cpus[0])
    (hmcin_A, hmcin_means, hmcin_stds) = hmcin_EM(T, X_cpu, nb_iter=5,
        A_init=A, means_init=means, stds_init=stds, nb_classes=nb_classes,
        nb_channels=nb_channels)
    hmcin_mpm_seg, _ = hmcin_MPM_segmentation(T, X,
        hmcin_A, hmcin_means, hmcin_stds, nb_classes=nb_classes,
        nb_channels=nb_channels)
    #plt.imshow(chain_to_image(hmcin_mpm_seg))
    #plt.show()
    img_hmcin_seg = chain_to_image(hmcin_mpm_seg.astype(np.int32))
    img_hmcin_seg = ((img_hmcin_seg - np.amin(img_hmcin_seg)) /
        (np.amax(img_hmcin_seg) - np.amin(img_hmcin_seg)))
    Image.fromarray(
        (img_hmcin_seg * 255).astype(np.uint8)
    ).save("./figures/" + img_name + "_hmcin.png")

    # SPMC segmentation
    #print("SPMC segmentation")
    ## Force CPU computation, nothing is batched here to go on GPU
    #cpus = jax.devices("cpu")
    #X_cpu = jax.device_put(X, cpus[0])
    #(spmc_A, spmc_means, spmc_stds, spmc_A_sig_params,
    #    spmc_norm_params) = spmc_gradient_llkh(T, X_cpu, nb_iter=50,
    #    A_init=hmcin_A, means_init=hmcin_means, stds_init=hmcin_stds,
    #    alpha=0.01,#0.01,
    #    nb_classes=nb_classes, nb_channels=nb_channels)
    #spmc_mpm_seg, _ = spmc_MPM_segmentation(T, X,
    #    spmc_A_sig_params, spmc_norm_params, nb_classes=nb_classes,
    #    nb_channels=nb_channels)
    #img_spmc_seg = chain_to_image(spmc_mpm_seg.astype(np.int32))
    #img_spmc_seg = ((img_spmc_seg - np.amin(img_spmc_seg)) /
    #    (np.amax(img_spmc_seg) - np.amin(img_spmc_seg)))
    #Image.fromarray(
    #    (img_spmc_seg * 255).astype(np.uint8)
    #).save("./figures/" + img_name + "_spmc.png")

    ## PMC segmentation
    print("PMC segmentation")
    ## Force CPU computation, nothing is batched here to go on GPU
    cpus = jax.devices("cpu")
    X_cpu = jax.device_put(X, cpus[0])
    (pmc_A, pmc_means, pmc_stds, pmc_A_sig_params, pmc_norm_params) = \
        pmc_gradient_llkh(T, X_cpu, nb_iter=5, A_init=hmcin_A,
        means_init=hmcin_means, stds_init=hmcin_stds,
        alpha=0.01,
        nb_classes=nb_classes, nb_channels=nb_channels)
    pmc_mpm_seg, _ = pmc_MPM_segmentation(T, X, pmc_A_sig_params,
        pmc_norm_params, nb_classes=nb_classes, nb_channels=nb_channels)
    img_pmc_seg = chain_to_image(pmc_mpm_seg.astype(np.int32))
    img_pmc_seg = ((img_pmc_seg - np.amin(img_pmc_seg)) /
        (np.amax(img_pmc_seg) - np.amin(img_pmc_seg)))
    Image.fromarray(
        (img_pmc_seg * 255).astype(np.uint8)
    ).save("./figures/" + img_name + "_pmc.png")

    ## DSPMC segmentation
    #print("DSPMC segmentation")
    #key, subkey = jax.random.split(key)
    #(dspmc_A, dspmc_means, dspmc_stds,
    #    dspmc_A_ffnet_and_params, dspmc_meanvars_ffnet_and_params,
    #) = dspmc_gradient_llkh(T, X, subkey, nb_iter=50,
    #    A_init=spmc_A, means_init=spmc_means, stds_init=spmc_stds,
    #    pre_seg=spmc_mpm_seg,
    #    A_sig_params_init=spmc_A_sig_params,
    #    norm_params_init=spmc_norm_params,
    #    with_pretrain=True, with_output_constraint=True,
    #    with_gpu=True,
    #    alpha=0.00001,
    #    nb_classes=nb_classes, nb_channels=nb_channels
    #    )
    #dspmc_mpm_seg, _ = dspmc_MPM_segmentation(T, X,
    #    dspmc_A_ffnet_and_params[0],
    #    dspmc_A_ffnet_and_params[1],
    #    dspmc_meanvars_ffnet_and_params[0],
    #    dspmc_meanvars_ffnet_and_params[1],
    #    nb_classes=nb_classes, nb_channels=nb_channels
    #    )
    #img_dspmc_seg = chain_to_image(dspmc_mpm_seg.astype(np.int32))
    #img_dspmc_seg = ((img_dspmc_seg - np.amin(img_dspmc_seg)) /
    #    (np.amax(img_dspmc_seg) - np.amin(img_dspmc_seg)))
    #Image.fromarray(
    #    (img_dspmc_seg * 255).astype(np.uint8)
    #).save("./figures/" + img_name + "_dspmc.png")

    # DPMC segmentation
    print("DPMC segmentation")
    key, subkey = jax.random.split(key)
    (dpmc_A, dpmc_means, dpmc_stds,
        dpmc_A_ffnet_and_params, dpmc_meanvars_ffnet_and_params) = \
            dpmc_gradient_llkh(T, X,
        subkey, nb_iter=50,
        A_init=pmc_A, means_init=pmc_means, stds_init=pmc_stds,
        pre_seg=pmc_mpm_seg,
        norm_params_init=pmc_norm_params,
        A_sig_params_init=pmc_A_sig_params,
        with_pretrain=True, with_output_constraint=True,
        with_gpu=True, alpha=0.01, nb_classes=nb_classes,
        nb_channels=nb_channels)
    dpmc_mpm_seg, _ = dpmc_MPM_segmentation(T, X,
        dpmc_A_ffnet_and_params[0],
        dpmc_A_ffnet_and_params[1],
        dpmc_meanvars_ffnet_and_params[0],
        dpmc_meanvars_ffnet_and_params[1],
        nb_classes=nb_classes, nb_channels=nb_channels)

    fig, axes = plt.subplots(1, 7)
    axes[0].imshow(img)
    axes[1].set_title("H")
    axes[1].imshow(chain_to_image(X))
    axes[1].set_title("X")
    axes[2].imshow(chain_to_image(hmcin_mpm_seg.astype(np.int32)))
    axes[2].set_title("HMCIN")
    #axes[3].imshow(chain_to_image(spmc_mpm_seg.astype(np.int32)))
    #axes[3].set_title("SPMC")
    axes[4].imshow(chain_to_image(pmc_mpm_seg.astype(np.uint32)))
    axes[4].set_title("PMC")
    #axes[5].imshow(chain_to_image(dspmc_mpm_seg.astype(np.uint32)))
    #axes[5].set_title("DSPMC")
    axes[6].imshow(chain_to_image(dpmc_mpm_seg.astype(np.uint32)))
    axes[6].set_title("DPMC")
    plt.show()

if __name__ == "__main__":
    cpus = jax.devices("cpu")
    try:
        gpus = jax.devices("gpu")
    except RuntimeError as e:    
        print("RuntimeError: ", e)
        gpus = []
    print(cpus, gpus) # even though jax.devices() only shows the GPU there is
    # always CPU to be found
    #os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false" # do not preallocate GPU
    #os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = "platform" # dynamically

    np.random.seed(0)
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    jax.config.update("jax_enable_x64", True)
    # NOTE: jax will now break if nan appear
    jax.config.update("jax_debug_nans", True)

    #img_path = './test_images/dragonfly_bw.png'
    img_path = './test_images/cattle_3cl.png'
    img_name = 'cattle3'
    #img_path = './test_images/great_pyrenees_124.png'
    #img_name = 'great_pyrenees_124'

    seg_image(img_path, img_name, key)
