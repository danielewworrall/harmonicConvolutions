import os
import sys
import time

#import cv2
import numpy as np
import scipy.linalg as scilin
import scipy.ndimage.interpolation as sciint
import tensorflow as tf

import input_data

from steer_conv import *

from matplotlib import pyplot as plt

import scipy as sp
from scipy import ndimage
from scipy import misc

from equivariant import *

def trainSingleGPU(model, lr, batch_size, n_epochs, n_filters, use_batchNorm,
		trial_num, combine_train_val, std_mult,
        gpuIdx,
        isClassification, n_rows, n_cols, n_channels, n_classes, size_after_conv,
        trainx, trainy, validx, validy, testx, testy):
    n_input = n_rows * n_cols * n_channels
    #select the correct function to build the model
    if model == 'fullyConvolutional':
        modelFunc = fullyConvolutional
    elif model == 'fullyConvolutional_Dieleman':
        modelFunc = fullyConvolutional_Dieleman
    elif model == 'deep_complex_bias':
        modelFunc = deep_complex_bias
    else:
        print('Model unrecognized')
        sys.exit(1)
    print('Using model: %s' % model)
    #single gpu model
    #PARAMETERS-----------------------------------------------------------------------
    # Parameters
    lr = lr
    batch_size = batch_size
    n_epochs = n_epochs
    save_step = 100		# Not used yet
    model = model

    # Network Parameters
    dropout = 0.75 				# Dropout, probability to keep units
    n_filters = n_filters
    dataset_size = trainx.shape[0] + validx.shape[0]
    print("Total size of trainig set (train + validation): ", dataset_size)
    print("Total output size: ", n_classes)
    if isClassification:
        print("Using classification loss.")
    else:
        print("Using regression loss.")
	print
	print("Learning Rate: ", lr)
	print("Batch Size: ", batch_size)
	print("Num Filters: ", n_filters)
	if use_batchNorm:
		print("Using batchNorm (MomentumOptimiser, m=0.9, learning rate will be annealed)")
	else:
		print("Not using batchNorm (AdamOptimiser, learning rate will not be annealed)")
	#CREATE NETWORK-------------------------------------------------------------------
	# tf Graph input
	with tf.device('/gpu:%d' % gpuIdx):
		x = tf.placeholder(tf.float32, [batch_size, n_input])
		if isClassification:
			y = tf.placeholder(tf.int64, [batch_size])
		else:
			y = tf.placeholder(tf.float32, [batch_size, n_classes])
		learning_rate = tf.placeholder(tf.float32)
		keep_prob = tf.placeholder(tf.float32)
		phase_train = tf.placeholder(tf.bool)
		
		# Construct model
		pred = modelFunc(x, keep_prob, n_filters, n_rows, n_cols, n_channels, size_after_conv, n_classes, batch_size, phase_train, std_mult, use_batchNorm)
		
		# Define loss and optimizer
		if isClassification:
			cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(pred, y))
		else:
			cost = tf.reduce_sum(tf.pow(y - pred, 2)) / (2 * 37)
		
		if use_batchNorm:
			momentum=0.9
			nesterov=True
			psi_preconditioner = 5e0
			opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=nesterov)
		else:
			opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		# Evaluate model
		if isClassification:
			correct_pred = tf.equal(tf.argmax(pred, 1), y)
			accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
			print('  Evaluation metric constructed')
		else:
			accuracy = cost

		print('  Constructed loss')
		grads_and_vars = opt.compute_gradients(cost)
		modified_gvs = []
		for g, v in grads_and_vars:
			if 'psi' in v.name:
				g = psi_preconditioner*g
			modified_gvs.append((g, v))
		optimizer = opt.apply_gradients(modified_gvs)
		print('  Optimizer built')
		
	grad_summaries_op = []
	for g, v in grads_and_vars:
		if 'psi' in v.name:
			grad_summaries_op.append(tf.histogram_summary(v.name, g))
			
	# Initializing the variables
	init = tf.initialize_all_variables()
	print('  Variables initialized')
	
	if combine_train_val:
		mnist_trainx = np.vstack([mnist_trainx, mnist_validx])
		mnist_trainy = np.hstack([mnist_trainy, mnist_validy])
	
	# Summary writers
	acc_ph = tf.placeholder(tf.float32, [], name='acc_')
	acc_op = tf.scalar_summary("Validation Accuracy", acc_ph)
	cost_ph = tf.placeholder(tf.float32, [], name='cost_')
	cost_op = tf.scalar_summary("Training Cost", cost_ph)
	lr_ph = tf.placeholder(tf.float32, [], name='lr_')
	lr_op = tf.scalar_summary("Learning Rate", lr_ph)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.log_device_placement = False
	config.inter_op_parallelism_threads = 1 #prevent inter-session threads?
	sess = tf.Session(config=config)
	summary = tf.train.SummaryWriter('./logs/current', sess.graph)
	print('  Summaries constructed')
	
	# GRAPH EXECUTION---------------------------------------------------
	sess.run(init)
	saver = tf.train.Saver()
	epoch = 0
	start = time.time()
	step = 0.
	lr_current = lr
	counter = 0
	best = 0.
	validationAccuracy = -1
	print('  Begin training')
	# Keep training until reach max iterations
	while epoch < n_epochs:
		generator = minibatcher(trainx, trainy, batch_size, shuffle=True)
		cost_total = 0.
		acc_total = 0.
		vacc_total = 0.
		for i, batch in enumerate(generator):
			batch_x, batch_y = batch
			'''print('Test saving...')
			for bi in xrange(batch_size):
					fileName = str(batch_y[bi]) + '_' + str(bi) + '.png'
					sp.misc.imsave(fileName, np.reshape(batch_x[bi, :], (32, 32, 3)))'''
			#lr_current = lr/np.sqrt(1.+lr_decay*epoch)
			
			# Optimize
			feed_dict = {x: batch_x, y: batch_y, keep_prob: dropout,
						 learning_rate : lr_current, phase_train : True}
			__, cost_, acc_, gso = sess.run([optimizer, cost, accuracy,
										grad_summaries_op], feed_dict=feed_dict)
			cost_total += cost_
			acc_total += acc_
			for summ in gso:
				summary.add_summary(summ, step)
			step += 1
		cost_total /=(i+1.)
		acc_total /=(i+1.)
		
		if not combine_train_val:
			val_generator = minibatcher(validx, validy, batch_size, shuffle=False)
			for i, batch in enumerate(val_generator):
				batch_x, batch_y = batch
				'''print('Test saving...')
				for bi in xrange(batch_size):
					fileName = str(batch_y[bi]) + '_' + str(bi) + '.png'
					sp.misc.imsave(fileName, np.reshape(batch_x[bi, :], (32, 32, 3)))'''
				# Calculate batch loss and accuracy
				feed_dict = {x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
				vacc_ = sess.run(accuracy, feed_dict=feed_dict)
				vacc_total += vacc_
			vacc_total = vacc_total/(i+1.)
		
		feed_dict={cost_ph : cost_total, acc_ph : vacc_total, lr_ph : lr_current}
		summaries = sess.run([cost_op, acc_op, lr_op], feed_dict=feed_dict)
		for summ in summaries:
			summary.add_summary(summ, step)

		if use_batchNorm: #only change learning rate when NOT using Adam
			best, counter, lr_current = get_learning_rate(vacc_total, best, counter, lr_current, delay=10)
		
		print "[" + str(trial_num),str(epoch) + \
			"], Minibatch Loss: " + \
			"{:.6f}".format(cost_total) + ", Train Acc: " + \
			"{:.5f}".format(acc_total) + ", Time: " + \
			"{:.5f}".format(time.time()-start) + ", Counter: " + \
			"{:2d}".format(counter) + ", Val acc: " + \
			"{:.5f}".format(vacc_total)
		epoch += 1
		validationAccuracy = vacc_total
				
		if (epoch) % 50 == 0:
			save_model(saver, './', sess)
	
	print "Testing"
	if experimentIdx == 2 or experimentIdx == 3:
		print("Test labels not available for this dataset, relying on validation accuracy instead.")
		print('Test accuracy: %f' % (validationAccuracy,))
		save_model(saver, './', sess)
		sess.close()
		return validationAccuracy
	else:
		# Test accuracy
		tacc_total = 0.
		test_generator = minibatcher(testx, testy, batch_size, shuffle=False)
		for i, batch in enumerate(test_generator):
			batch_x, batch_y = batch
			feed_dict={x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
			tacc = sess.run(accuracy, feed_dict=feed_dict)
			tacc_total += tacc
		tacc_total = tacc_total/(i+1.)
		print('Test accuracy: %f' % (tacc_total,))
		save_model(saver, './', sess)
		sess.close()
		return tacc_total

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
    """
    if len(tower_grads) == 1:
        return tower_grads[0]
    else:
        print('Processing %d sets of gradients.' % len(tower_grads))
    average_grads = []
    for grad_and_vars in zip(*tower_grads): #for each grad, vars set
        print("GRAD VARS SET")
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, v in grad_and_vars:
            if g == None: #if no gradient, don't average'
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        #concat only if we have any entries
        if len(grads) == 0:
            continue
        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def trainMultiGPU(model, lr, batch_size, n_epochs, n_filters, use_batchNorm,
		trial_num, combine_train_val, std_mult,
        gpuIdxs,
        isClassification, n_rows, n_cols, n_channels, n_classes, size_after_conv,
        trainx, trainy, validx, validy, testx, testy):
    numGPUs = len(gpuIdxs)
    n_input = n_rows * n_cols * n_channels
    dropout = 0.75 
    print('Using Multi-GPU Model with %d devices.' % numGPUs)
    #select the correct function to build the model
    if model == 'fullyConvolutional':
        modelFunc = fullyConvolutional
    elif model == 'fullyConvolutional_Dieleman':
        modelFunc = fullyConvolutional_Dieleman
    elif model == 'deep_complex_bias':
        modelFunc = deep_complex_bias
    else:
        print('Model unrecognized')
        sys.exit(1)
    print('Using model: %s' % model)
    
    #create some variables to use in training
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    phase_train = tf.placeholder(tf.bool)

    #create optimiser
    if use_batchNorm:
		momentum=0.9
		nesterov=True
		opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=nesterov)
    else:
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #create some placeholders
    #one of x for each GPU
    xs = []
    ys = []
    for g in gpuIdxs:
        with tf.device('/gpu:%d' % g):
            xs.append(tf.placeholder(tf.float32, [int(batch_size / numGPUs), n_input]))
            if isClassification:
                ys.append(tf.placeholder(tf.int64, [int(batch_size / numGPUs)]))
            else:
                ys.append(tf.placeholder(tf.float32, [int(batch_size / numGPUs), n_classes]))

    #setup model for each GPU
    linearGPUIdx = 0
    gradientsPerGPU = []
    lossesPerGPU = []
    accuracyPerGPU = []
    for g in gpuIdxs:
        with tf.device('/gpu:%d' % g):
            print('Building Model on GPU: %d' % g)
            #if True:
            with tf.name_scope('%s_%d' % (model, 0)) as scope:
                #print(scope)
                #build model 
                prediction = modelFunc(xs[linearGPUIdx], keep_prob, n_filters, n_rows, n_cols, n_channels,\
                    size_after_conv, n_classes, int(batch_size / numGPUs), phase_train, std_mult, use_batchNorm)
                #define loss
                if isClassification:
                    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(prediction, ys[linearGPUIdx]))
                else:
                    loss = tf.reduce_mean(tf.pow(y - prediction, 2))
                #define accuracy
                correct_prediction = tf.equal(tf.argmax(prediction, 1), ys[linearGPUIdx])
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                #reuse variables for next # Reuse variables for the next tower.
                tf.get_variable_scope().reuse_variables()
                # Retain the summaries from the final tower.
                #summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                # Calculate the gradients for the batch of data on this CIFAR tower.
                grads = opt.compute_gradients(loss)
                # Keep track of the gradients across all towers.
                gradientsPerGPU.append(grads)
                # Keep track of losses per gpu
                lossesPerGPU.append(loss)
                # Keep track of accuracy per gpu
                accuracyPerGPU.append(accuracy)


        linearGPUIdx += 1
    
    #Now define the CPU-side synchronisation (occurs when gradients are averaged)
    grads = average_gradients(gradientsPerGPU)

    #Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads)

    #in case we wanna add something here later, lets use a group
    train_op = tf.group(apply_gradient_op)

    #avg losses
    print(lossesPerGPU)
    print(accuracyPerGPU)
    #avg_loss = tf.reduce_mean(tf.pack(lossesPerGPU))
    #avg_accuracy = tf.reduce_mean(tf.pack(accuracyPerGPU))
    avg_loss = loss
    avg_accuracy = accuracy
    #init all variables
    init = tf.initialize_all_variables()
    print('Variables initialized')

    #done for final refinement after ideal params selected in hyperopt
    if combine_train_val:
        trainx = np.vstack([trainx, validx])
        trainy = np.hstack([trainy, validy])

    # Summary writers
    acc_ph = tf.placeholder(tf.float32, [], name='acc_')
    acc_op = tf.scalar_summary("Validation Accuracy", acc_ph)
    cost_ph = tf.placeholder(tf.float32, [], name='cost_')
    cost_op = tf.scalar_summary("Training Cost", cost_ph)
    lr_ph = tf.placeholder(tf.float32, [], name='lr_')
    lr_op = tf.scalar_summary("Learning Rate", lr_ph)

    # Configure tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = False
    config.log_device_placement = False
    config.inter_op_parallelism_threads = 2 #we shouldn't need this, but stalls otherwise
    sess = tf.Session(config=config)
    summary = tf.train.SummaryWriter('./logs/current', sess.graph)
    print('Summaries constructed')

    # GRAPH EXECUTION---------------------------------------------------
    sess.run(init)
    saver = tf.train.Saver(tf.all_variables())
    epoch = 0
    start = time.time()
    step = 0.
    lr_current = lr
    counter = 0
    best = 0.
    validationAccuracy = -1
    sizePerGPU = int(batch_size / numGPUs)
    print('Starting trainig loop')
    # Keep training until reach max iterations
    while epoch < n_epochs: # epoch loop
        generator = minibatcher(trainx, trainy, batch_size, shuffle=True)
        cost_total = 0.
        acc_total = 0.
        vacc_total = 0.
        # accumulate batches until we have enough
        for i, batch in enumerate(generator): # batch loop
            batch_x, batch_y = batch
            #construct the feed_dictionary
            feed_dict = {keep_prob: dropout,
                    learning_rate : lr_current, phase_train : True}
            for g in xrange(numGPUs):
                feed_dict[xs[g]] = batch_x[g*sizePerGPU:(g+1)*sizePerGPU,:]
                feed_dict[ys[g]] = batch_y[g*sizePerGPU:(g+1)*sizePerGPU]
            # Optimize
            __, cost_, acc_ = sess.run([train_op, avg_loss, avg_accuracy], feed_dict=feed_dict)
            cost_total += cost_
            acc_total += acc_

            step += 1

        cost_total /=((i)+1.)
        acc_total /=((i)+1.)

        if not combine_train_val:
            val_generator = minibatcher(validx, validy, batch_size, shuffle=False)
            for i, batch in enumerate(val_generator):
                batch_x, batch_y = batch
                #construct the feed_dictionary
                feed_dict = {keep_prob: dropout,
                        learning_rate : lr_current, phase_train : True}
                for g in xrange(numGPUs):
                    feed_dict[xs[g]] = batch_x[g*sizePerGPU:(g+1)*sizePerGPU,:]
                    feed_dict[ys[g]] = batch_y[g*sizePerGPU:(g+1)*sizePerGPU]
                #run session
                vacc_ = sess.run(avg_accuracy, feed_dict=feed_dict)
                vacc_total += vacc_
                vacc_total = vacc_total/(i+1.)

        feed_dict={cost_ph : cost_total, acc_ph : vacc_total, lr_ph : lr_current}
        summaries = sess.run([cost_op, acc_op, lr_op], feed_dict=feed_dict)
        for summ in summaries:
            summary.add_summary(summ, step)

        if use_batchNorm: #only change learning rate when NOT using Adam
            best, counter, lr_current = get_learning_rate(vacc_total, best, counter, lr_current, delay=10)

        print "[" + str(trial_num),str(epoch) + \
        "], Epoch Loss: " + \
        "{:.6f}".format(cost_total) + ", Train Acc: " + \
        "{:.5f}".format(acc_total) + ", Time: " + \
        "{:.5f}".format(time.time()-start) + ", Counter: " + \
        "{:2d}".format(counter) + ", Val acc: " + \
        "{:.5f}".format(vacc_total)
        epoch += 1
        validationAccuracy = vacc_total

        if (epoch) % 50 == 0:
            save_model(saver, './', sess)

    print "Testing"
    if experimentIdx == 2 or experimentIdx == 3:
        print("Test labels not available for this dataset, relying on validation accuracy instead.")
        print('Test accuracy: %f' % (validationAccuracy,))
        save_model(saver, './', sess)
        sess.close()
        return validationAccuracy
    else:
        #run over test set
        '''
        # Test accuracy
        tacc_total = 0.
        test_generator = minibatcher(testx, testy, batch_size, shuffle=False)
        for i, batch in enumerate(test_generator):
        batch_x, batch_y = batch
        feed_dict={x: batch_x, y: batch_y, keep_prob: 1., phase_train : False}
        tacc = sess.run(accuracy, feed_dict=feed_dict)
        tacc_total += tacc
        tacc_total = tacc_total/(i+1.)
        print('Test accuracy: %f' % (tacc_total,))
        save_model(saver, './', sess)
        sess.close()
        return tacc_total
        '''

##### MAIN SCRIPT #####
def run(model='', lr=1e-2, batch_size=250, n_epochs=500, n_filters=30, use_batchNorm=True,
		trial_num='N', combine_train_val=False, std_mult=0.4, deviceIdxs=[0], experimentIdx = 0):
	tf.reset_default_graph()
	#1. LOAD DATA---------------------------------------------------------------------
	if experimentIdx == 0: #MNIST
		print("MNIST")
		# Load dataset
		train = np.load('/home/sgarbin/data/mnist_rotation_new/rotated_train.npz')
		valid = np.load('/home/sgarbin/data/mnist_rotation_new/rotated_valid.npz')
		test = np.load('/home/sgarbin/data/mnist_rotation_new/rotated_test.npz')
		trainx, trainy = train['x'], train['y']
		validx, validy = valid['x'], valid['y']
		testx, testy = test['x'], test['y']

		isClassification = True
		n_rows = 28
		n_cols = 28
		n_channels = 1
		n_input = n_rows * n_cols * n_channels
		n_classes = 10 				# MNIST total classes (0-9 digits)
		size_after_conv = 7 * 7
	elif experimentIdx == 1: #CIFAR10
		print("CIFAR10")
		# Load dataset
		trainx = np.load('/home/sgarbin/data/cifar_numpy/trainX.npy')
		trainy = np.load('/home/sgarbin/data/cifar_numpy/trainY.npy')
		
		validx = np.load('/home/sgarbin/data/cifar_numpy/validX.npy')
		validy = np.load('/home/sgarbin/data/cifar_numpy/validY.npy')

		testx = np.load('/home/sgarbin/data/cifar_numpy/testX.npy')
		testy = np.load('/home/sgarbin/data/cifar_numpy/testY.npy')

		isClassification = True
		n_rows = 32
		n_cols = 32
		n_channels = 3
		n_input = n_rows * n_cols * n_channels
		n_classes = 10 
		size_after_conv = 8 * 8

	elif experimentIdx == 2: #Plankton
		print("Plankton")
		# Load dataset
		trainx = np.load('/home/sgarbin/data/plankton_numpy/trainX.npy')
		trainy = np.load('/home/sgarbin/data/plankton_numpy/trainY.npy')
		
		validx = np.load('/home/sgarbin/data/plankton_numpy/validX.npy')
		validy = np.load('/home/sgarbin/data/plankton_numpy/validY.npy')

		testx = np.load('/home/sgarbin/data/plankton_numpy/testX.npy')
		testy = np.zeros(1) #symbolic only as not available

		isClassification = True
		n_rows = 95
		n_cols = 95
		n_channels = 1
		n_input = n_rows * n_cols * n_channels
		n_classes = 121
		size_after_conv = -1
	elif experimentIdx == 3: #Galaxies
		print("Galaxies")
		# Load dataset
		trainx = np.load('/home/sgarbin/data/galaxies_numpy/trainX.npy')
		trainy = np.load('/home/sgarbin/data/galaxies_numpy/trainY.npy')
		
		validx = np.load('/home/sgarbin/data/galaxies_numpy/validX.npy')
		validy = np.load('/home/sgarbin/data/galaxies_numpy/validY.npy')

		testx = np.load('/home/sgarbin/data/galaxies_numpy/testX.npy')
		testy = np.zeros(1) #symbolic only as not available

		isClassification = False
		n_rows = 64
		n_cols = 64
		n_channels = 3
		n_input = n_rows * n_cols * n_channels
		n_classes = 37
		size_after_conv = -1
	else:
		print('Dataset unrecognized, options are:')
		print('MNIST: 0')
		print('CIFAR: 1')
		print('PLANKTON: 2')
		print('GALAXIES: 3')
		sys.exit(1)
        if len(deviceIdxs) > 1:
            return trainMultiGPU(model, lr, batch_size, n_epochs, n_filters, use_batchNorm,
            trial_num, combine_train_val, std_mult, deviceIdxs, isClassification,
            n_rows, n_cols, n_channels, n_classes, size_after_conv,trainx,trainy,validx,validy,testx,testy)
        else:
            return trainSingleGPU(model, lr, batch_size, n_epochs, n_filters, use_batchNorm,
            trial_num, combine_train_val, std_mult, deviceIdxs[0], isClassification,
            n_rows, n_cols, n_channels, n_classes, size_after_conv,trainx,trainy,validx,validy,testx,testy)
#ENTRY POINT------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
	run(model='fullyConvolutional', lr=1e-3, batch_size=100, n_epochs=10, std_mult=0.4,
		n_filters=10, combine_train_val=False)
