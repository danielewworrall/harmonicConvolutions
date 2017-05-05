# MNIST-rot experiments
This folder contains the script to run the MNIST-rot experiments.

The model is defined in `mnist_model.py`. To run the code, just run  
```bash
python run_mnist.py --combine_train_val True
```

This should download the MNIST-rot dataset and setup a log and checkpoint
folder, in which results are saved. The default settings for the model are
those we have arrived at for this task. Feel free to experiment with them. If
you find anything interesting, or any bugs for that matter, we'll be happy to
hear from you.

TODO
- [ ] Include pretrained model
- [ ] Retrieve the SOTA settings
