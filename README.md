# Phase invariant convolutions
This folder contains the basic material to construct Harmonic Networks (HNs)
* train.py is the main entry point for our code
* `harmonic_network_ops.py` contains core HC implementations
* `harmonic_network_helpers.py` contains handy functions for using these
* `harmonic_network_models.py` contains the model definitions that allow to reproduce our results

To run the MNIST example from the paper, navigate to the parent directory of this repo and type:
```python
python train.py 0 mnist deep_stable <yourDataDirectory>
```
Here, `<yourDataDirectory>` is the folder into which we the datasets will be downloaded, the `0` means we will be using GPU 0, `mnist` signifies the dataset to train on, and `deep_stable` the network mdoel as defined in `harmonic_network_models.py`

Please note that
* this is work in progress, so pull often!
* the API is not yet stable and subject to change (particularly, we are working on improving the ease of use of our convolution functions)

Todos which we have completed:
- [x] API for core HC functions
- [x] Easy rotated MNIST example

Todos which we are currently working on:
- [ ] Providing easy training code for our BSD experiments
- [ ] Providing multi-threaded reads for data-feeding