# Deep Hidden Markov Models

## Installation
A simple way is to create a new conda environment and install several packages
```
conda create --name deep_hmm python=3.9
conda activate deep_hmm
conda install -c anaconda cudnn
conda install jax cuda-nvcc -c conda-forge -c nvidia
conda install -c anaconda scikit-learn 
conda install -c conda-forge matplotlib 
conda install -c conda-forge dm-haiku 
conda install -c conda-forge optax
conda install pillow
```

Repository linked with the publications

> Deep parameterizations of pairwise and triplet Markov models for unsupervised
> classification of sequential data, H. Gangloff, K. Morales and Y. Petetin, 2022. 
> (https://hal.archives-ouvertes.fr/view/index/docid/3584314)

> A general parametrization framework for Pairwise Markov Models: an application to unsupervised image segmentation, H. Gangloff, K. Morales, Y. Petetin, International Workshop on Machine Learning for Signal Processing (MLSP), 2021.
> (https://ieeexplore.ieee.org/document/9596395)

The code is expected to grow overtime. Currently, the available models are:
- Hidden Markov Chains
- Semi Pairwise Markov Chains
- Pairwise Markov Chains
- Deep Semi Pairwise Markov Chains
- Deep Pairwise Markov Chains

Currently, you will find the following algorithms:
- Expectation Maximization (for HMCs)
- Maximum Likelihood with gradient ascent (for the other models)
- Forward Backward in logspace (for all the models)
- Pretraining via backpropagation (for D-SPMCs and D-PMCs)

The code is built with JAX (autotomatic differentiation) and Haiku (deep neural networks). For efficiency, the backpropagation pretrainings are performed on GPU while the likelihood maximizations are performed on CPU.

For more details, refer to the publication.
