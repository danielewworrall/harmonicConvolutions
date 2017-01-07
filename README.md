# Harmonic Networks: Deep Translation and Rotation Equivariance

> Please read the following information carefully and let us know if anything is missing/you have discovered a bug. We always welcome your feedback!

This folder contains the basic material to construct Harmonic Networks (HNets). Please see our <a href="http://visual.cs.ucl.ac.uk/pubs/harmonicNets/index.html"> project page </a> for more details.
* `train.py` is the main entry point for our code.
* `model_assembly_train.py` contains our multi-gpu trainig routines that serve as an example of how our functions can interface with regular tensorflow code.
* `io_helpers.py` contains our code for downloading, processing and batching datasets.
* `harmonic_network_ops.py` contains core HNet implementations.
* `harmonic_network_helpers.py` contains handy functions for using these (such as block definitions).
* `harmonic_network_models.py` contains the model definitions that are necessary to reproduce our results.

All other scripts are for our purposes only and can be safely ignored by the user.

____
> NOTE: The model definitions and core functions are independent of our training code and can be used without it.
____

To run the MNIST example from the paper, navigate to the parent directory of this repo and type:
```python
python train.py 0 mnist deep_mnist <yourDataDirectory>
```
Here, `<yourDataDirectory>` is the folder into which the datasets will be downloaded, the `0` means we will be using GPU 0, `mnist` signifies the dataset to train on, and `deep_stable` the network mdoel as defined in `harmonic_network_models.py`.
You can train on more than one GPU by making the first argument a comma-separated list. For example `0,1,2` would run the training code on the first three GPUs of a system.

Dependencies:
* we require tensorflow 0.12 as documented <a href="https://www.tensorflow.org/versions/r0.12/api_docs/index.html">here</a>. Newer versions of the API may be supported in future and older versions are not supported.

Please note that
* this is work in progress, so pull often!
* the API is not yet stable and subject to change (particularly, we are working on improving the ease of use of our convolution functions).

Todos which we have completed:
- [x] API for core HC functions
- [x] Easy rotated MNIST example
- [x] Short tutorial

Todos which we are currently working on:
- [ ] Providing easy training code for our BSD experiments
- [ ] Providing multi-threaded reads for data-feeding
- [ ] API simplication
- [ ] Longer tutorial

# HNet Tensorflow Tutorial
To come
