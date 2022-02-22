'''
Code to reproduce Scenario 21 of the paper
Generalized Pairwise and Triplet Markov Chains: example of a
deep parametrization for unsupervised signal processing
Hugo Gangloff, Katherine Morales and Yohan Petetin
'''

import os
from os import listdir
from os.path import isfile, join
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
import jax

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

def seg_moving_means(exp_name, key):
    np.seterr(all="warn")
    img_dir = "./test_images/cattles256/"
    images = [img for img in listdir(img_dir) if isfile(join(img_dir, img))]
    for f_name in ["_km.txt", "_hmcin.txt",
        "_spmc.txt",
        "_pmc.txt",
        "_dspmc.txt",
        "_dpmc.txt",
        ]:
        f_name = exp_name + f_name
        with open(f_name, "w") as f:
            print("image_name, ", end="", file=f)
            for r in np.arange(0.1, 1.2, 0.1):
                print(str(r) + ", ", end="", file=f)
            print("\n", end="", file=f)
    for img_name in images:
        for f_name in ["_km.txt", "_hmcin.txt",
            "_spmc.txt",
            "_pmc.txt",
            "_dspmc.txt",
            "_dpmc.txt",
            ]:
            f_name = exp_name + f_name
            with open(f_name, "a") as f:
                print(img_name + ", ", end="", file=f)
        for r in np.arange(0.1, 1.2, 0.1):
            img = np.array(Image.open(join(img_dir, img_name)))
            img[img == 255] = 1
            obs = [0.5]
            H = image_to_chain(img)[:]
            b = np.array([0., float(r)])
            for i in range(1, 256*256):
                obs.append(np.sin(b[H[i]] + obs[-1]) + np.random.randn()*0.5)
            X = np.asarray(obs).astype(np.float64)
            T = len(H)

            nb_classes = 2
            nb_channels = 1

            #X_img = chain_to_image(X)
            #X_img = ((X_img - np.amin(X_img)) /
            #    (np.amax(X_img) - np.amin(X_img)))
            #Image.fromarray((X_img * 255).astype(np.uint8)).save("./figures/obs.png")
            #Image.fromarray((img * 255).astype(np.uint8)).save("./figures/img.png")


            # KMeans segmentation
            kmeans_seg = KMeans(n_clusters=2).fit(X.reshape((-1, 1))).labels_
            kmeans_e = np.count_nonzero(kmeans_seg != H) / H.shape[0]
            if kmeans_e > 0.5:
                kmeans_e = 1 - kmeans_e
                print("Error in KMeans segmentation", kmeans_e)
                inverted_km_seg = kmeans_seg.copy()
                inverted_km_seg[kmeans_seg == 0] = 1
                inverted_km_seg[kmeans_seg == 1] = 0
                kmeans_seg = inverted_km_seg
            print("Error in KMeans segmentation", kmeans_e)
            with open(exp_name + "_km.txt", "a") as f:
                print(str(kmeans_e) + ", ", end="", file=f)    
            means = np.array([np.mean(X[kmeans_seg == 0]),
                                np.mean(X[kmeans_seg == 1])])
            stds = np.array([np.std(X[kmeans_seg == 0]),
                                np.std(X[kmeans_seg == 1])])
            A = np.zeros((2, 2))
            for t in range(1, T):
                A[kmeans_seg[t - 1], kmeans_seg[t]] += 1
            A[0] /= np.sum(A[0])
            A[1] /= np.sum(A[1])

            # HMCIN segmentation
            (hmcin_A, hmcin_means, hmcin_stds) = hmcin_EM(T, X, nb_iter=50,
                A_init=A, means_init=means, stds_init=stds)
            hmcin_mpm_seg, hmcin_e = hmcin_MPM_segmentation(T, X,
                hmcin_A, hmcin_means, hmcin_stds, H)
            with open(exp_name + "_hmcin.txt", "a") as f:
                print(str(hmcin_e) + ", ", end="", file=f)    
            #img_hmcin_seg = chain_to_image(hmcin_mpm_seg.astype(np.int32))
            #Image.fromarray(
            #    (img_hmcin_seg * 255).astype(np.uint8)
            #).save("./figures/hmcin.png")

            (spmc_A, spmc_means, spmc_stds, spmc_A_sig_params,
                spmc_norm_params) = spmc_gradient_llkh(T, X, nb_iter=50,
                A_init=hmcin_A, means_init=hmcin_means, stds_init=hmcin_stds,
                H_gt=H, alpha=0.01)
            spmc_mpm_seg, spmc_e = spmc_MPM_segmentation(T, X,
                spmc_A_sig_params, spmc_norm_params, H=H)
            with open(exp_name + "_spmc.txt", "a") as f:
                print(str(spmc_e) + ", ", end="", file=f)    
            #img_spmc_seg = chain_to_image(spmc_mpm_seg.astype(np.int32))
            #Image.fromarray(
            #    (img_spmc_seg * 255).astype(np.uint8)
            #).save("./figures/spmc.png")

            (pmc_A, pmc_means, pmc_stds, pmc_A_sig_params, pmc_norm_params) = \
                pmc_gradient_llkh(T, X, nb_iter=50, A_init=hmcin_A,
                means_init=hmcin_means, stds_init=hmcin_stds, H_gt=H,
                alpha=0.01)
            pmc_mpm_seg, pmc_e = pmc_MPM_segmentation(T, X, pmc_A_sig_params,
                pmc_norm_params, H)
            with open(exp_name + "_pmc.txt", "a") as f:
                print(str(pmc_e) + ", ", end="", file=f)    
            #img_pmc_seg = chain_to_image(pmc_mpm_seg.astype(np.int32))
            #Image.fromarray(
            #    (img_pmc_seg * 255).astype(np.uint8)
            #).save("./figures/pmc.png")

            key, subkey = jax.random.split(key)
            (dspmc_A, dspmc_means, dspmc_stds,
                dspmc_A_ffnet_and_params, dspmc_meanvars_ffnet_and_params,
            ) = dspmc_gradient_llkh(T, X, subkey, nb_iter=50,
                A_init=spmc_A, means_init=spmc_means, stds_init=spmc_stds,
                pre_seg=spmc_mpm_seg,
                A_sig_params_init=spmc_A_sig_params,
                norm_params_init=spmc_norm_params,
                H_gt=H, with_pretrain=True, with_output_constraint=True,
                with_gpu=True, alpha=0.01)
            dspmc_mpm_seg, dspmc_e = dspmc_MPM_segmentation(T, X,
                dspmc_A_ffnet_and_params[0],
                dspmc_A_ffnet_and_params[1],
                dspmc_meanvars_ffnet_and_params[0],
                dspmc_meanvars_ffnet_and_params[1],
                H=H)
            with open(exp_name + "_dspmc.txt", "a") as f:
                print(str(dspmc_e) + ", ", end="", file=f)    
            #img_dspmc_seg = chain_to_image(dspmc_mpm_seg.astype(np.int32))
            #Image.fromarray(
            #    (img_dspmc_seg * 255).astype(np.uint8)
            #).save("./figures/dspmc.png")

            key, subkey = jax.random.split(key)
            (dpmc_A, dpmc_means, dpmc_stds,
                dpmc_A_ffnet_and_params, dpmc_meanvars_ffnet_and_params) = \
                    dpmc_gradient_llkh(T, X,
                subkey, nb_iter=50,
                A_init=pmc_A, means_init=pmc_means, stds_init=pmc_stds,
                pre_seg=pmc_mpm_seg,
                norm_params_init=pmc_norm_params,
                A_sig_params_init=pmc_A_sig_params,
                H_gt=H, with_pretrain=True, with_output_constraint=True,
                with_gpu=True, alpha=0.01)
            dpmc_mpm_seg, dpmc_e = dpmc_MPM_segmentation(T, X,
                dpmc_A_ffnet_and_params[0],
                dpmc_A_ffnet_and_params[1],
                dpmc_meanvars_ffnet_and_params[0],
                dpmc_meanvars_ffnet_and_params[1],
                H)
            with open(exp_name + "_dpmc.txt", "a") as f:
                print(str(dpmc_e) + ", ", end="", file=f)    
            #img_dpmc_seg = chain_to_image(dpmc_mpm_seg.astype(np.int32))
            #Image.fromarray(
            #    (img_dpmc_seg * 255).astype(np.uint8)
            #).save("./figures/dpmc.png")

        for f_name in ["_km.txt",
            "_hmcin.txt",
            "_spmc.txt",
            "_pmc.txt",
            "_dspmc.txt",
            "_dpmc.txt",
            ]:
            f_name = exp_name + f_name
            with open(f_name, "a") as f:
                print("\n", end="", file=f)

if __name__ == "__main__":
    jax.config.update("jax_platform_name", 'cpu')
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false" # do not preallocate GPU
    os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = "platform" # dynamically

    np.random.seed(0)
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    jax.config.update("jax_enable_x64", True)
    # NOTE: jax will now break if nan appear
    jax.config.update("jax_debug_nans", True)

    exp_name = "scenario_21_experiment"
    seg_moving_means(exp_name, key)
