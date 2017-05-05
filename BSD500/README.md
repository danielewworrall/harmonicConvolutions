# BSD500 experiments
This folder contains the script to run the BSD500 experiments.

# 1 Download the dataset
We include a processed version of the relevant parts of BSD500 for the
experiments. For the full dataset and assocated resources, please visit
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html.

Run these commands to download the pickled dataset
```bash
wget https://www.dropbox.com/s/mh0ch9s1va5dq0v/bsd_pkl_float.zip
unzip bsd_pkl_float.zip
rm bsd_pkl_float.zip
```

#2 Run the model
The model is defined in `BSD_model.py`. To run the code, just run
```bash
python run_BSD.py --combine_train_val True
```

The default settings for the model are will produce similar results to in our
paper---feel free to experiment with them. If you find anything interesting, 
or any bugs for that matter, we'll be happy to hear from you.

#3 Bayesian hyperparameter optimization
It may be that you wish to use our code on your own datasets. If this is so,
then it may be the case that the default hyperparameters are not the best 
settings. The script ```bayesian_optimization.py``` is a useful tool to quickly
come to a set of hyperparameters which perform well on a given validation set.
Details can be found at: https://scikit-optimize.github.io/.

TODO
- [ ] Include pretrained model
- [ ] Retrieve the SOTA settings
