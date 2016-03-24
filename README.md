# cifar.torch

The code achieves 92.45% accuracy on CIFAR-10 just with horizontal reflections.

Corresponding blog post: http://torch.ch/blog/2015/07/30/cifar.html

Accuracies:

 | No flips | Flips
--- | --- | ---
VGG+BN+Dropout | 91.3% | 92.45%
NIN+BN+Dropout | 90.4% | 91.9%

Would be nice to add other architectures, PRs are welcome!

Data preprocessing:

```bash
OMP_NUM_THREADS=2 th -i provider.lua
```

```lua
provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)
```
Takes about 30 seconds and saves 1400 Mb file.

Training:

```bash
CUDA_VISIBLE_DEVICES=0 th train.lua --model vgg_bn_drop -s logs/vgg
```
