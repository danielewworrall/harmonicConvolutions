import numpy as np
import matplotlib.pyplot as plt

#experimentParameters
batch_sizes = np.load('dopeyExperiments1/batch_sizes.npy')
learning_rates = np.load('dopeyExperiments1/learning_rates.npy')
stddev_multipliers = np.load('dopeyExperiments1/stddev_multipliers.npy')

#results
testAccuracies = np.load('dopeyExperiments1/yResultsMNIST.npy')

#the size of the filters
filterSizes = np.zeros(5)
filterSizes[0] = 2
filterSizes[1] = 4
filterSizes[2] = 6
filterSizes[3] = 8
filterSizes[4] = 10

xLabels = []
for i in xrange(batch_sizes.shape[0]):
    xLabels.append("bs: " + ('%.2f' % batch_sizes[i]) + ";\n lr: " + ('%.4f' % learning_rates[i]) + ";\n std: " + ('%.2f' % stddev_multipliers[i]))

xValues = np.arange(batch_sizes.shape[0])
colours = ['b', 'g', 'r', 'g', 'm', 'y', 'k']
bar_width = 0.15
for i in xrange(filterSizes.shape[0]):
    plt.bar(xValues + bar_width * i, testAccuracies[i, :], bar_width, color=colours[i])
plt.xticks(xValues, xLabels)

axes = plt.gca()
axes.set_ylim([0.8,1.0])

plt.show()


