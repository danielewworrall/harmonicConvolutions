import tensorflow as tf
import matplotlib.pyplot as pyPlot
import numpy as np

num_epochs = 4
batch_size = 1024
learning_rate = 1e-2 * 0.5
i_numRows = 28
i_numCols = 28
i_numChannels = 1
alpha = 1000
fully_connected_size = 32

TRAIN_FILE = 'data/MNIST/train.tfrecords'
VALIDATION_FILE = 'data/MNIST/validation.tfrecords'


#-----------------------IO----------------------------
def read_decode(filename_string_queue):
	with tf.name_scope('IO') as scope:
		reader = tf.TFRecordReader()

		#get next example
		key, example = reader.read(filename_string_queue)

		#parse
		features = tf.parse_single_example(serialized=example,
			features={
	          'image_raw': tf.FixedLenFeature([], tf.string),
	          'label': tf.FixedLenFeature([], tf.int64),
	      })

		#decode (will still be a string at this point)
		image = tf.decode_raw(features['image_raw'], tf.uint8)
		image = tf.reshape(image, [i_numRows, i_numCols, i_numChannels])

		#cast to float and normalise
		image = tf.cast(image, tf.float32) * (1.0 / 255.0) - 0.5

		label = tf.cast(features['label'], tf.int64)

		return image, label

def pipeline(fileName, batch_size, num_epochs):
	with tf.name_scope('IO') as scope:
		#only one filename for MNIST, so no need to shuffle
		filename_string_queue = tf.train.string_input_producer([fileName], num_epochs=num_epochs, shuffle=False)

		#add nodes to read and decode a single example
		image, label = read_decode(filename_string_queue)

		min_after_capacity = 10000
		capacity = min_after_capacity * batch_size * 4

		image_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_capacity)

		return image_batch, label_batch

#-----------------------MODEL-------------------------

#image input

#Add data loading graph
image_batch, label_batch = pipeline(TRAIN_FILE, batch_size, num_epochs)
input_summary = tf.image_summary("InputImages", image_batch, max_images=5)

Q_identity = tf.placeholder(tf.float32, shape=[9,9], name="Q_identity")

#matrix Q input
Q = tf.Variable(tf.random_normal(shape=[9,9],mean=0.0, stddev=0.05), name="Q")

#weight parameters of the convolution
w = tf.Variable(tf.random_normal(shape=[3,3],mean=0.0, stddev=0.05), name="w")

linearW0 = tf.Variable(tf.random_normal([i_numRows * i_numCols * i_numChannels, fully_connected_size]), name='linear_weight0')
linearb0 = tf.Variable(tf.random_normal([fully_connected_size]), name='linear_bias0')

linearW1 = tf.Variable(tf.random_normal([fully_connected_size, 10]), name='linear_weight1')
linearb1 = tf.Variable(tf.random_normal([10]), name='linear_bias1')



#calculate this once (9 * 1)
Q_t_w = tf.matmul(Q, tf.reshape(w, [9, 1]))

#define for loop functions
def compute_f_t_Q(Q_col):

	#reshape the column to get weights
	cWeights = tf.reshape(Q_col, [3, 3, 1, 1])

	#do convolution
	conv_res = tf.nn.conv2d(image_batch, cWeights, strides = [1, 1, 1, 1], padding = "SAME")

	return tf.squeeze(conv_res)

#compute convolution for every column of Q -> this will be (9, batch_size, numRows, numCols, numChannels)
f_t_Q = tf.map_fn(compute_f_t_Q, tf.transpose(Q), name="ConvolutionsftQ")

with tf.name_scope('NormalisedDotProduct') as scope:
	#tile Q_t_w
	with tf.name_scope('QtwTile') as scope:
		Q_t_w_tiled = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.tile(Q_t_w, [1, batch_size]), 2), [1, 1, i_numRows]), 3), [1, 1, 1,i_numCols], name="QtwTiling")


	numerator = tf.reduce_sum(tf.mul(f_t_Q, Q_t_w_tiled), [0])
	denominator = tf.add(tf.sqrt(tf.reduce_sum(tf.mul(Q_t_w_tiled, Q_t_w_tiled), [0])), tf.sqrt(tf.reduce_sum(tf.mul(f_t_Q, f_t_Q), [0])))

	phis = tf.truediv(numerator, denominator)

	phis_summary = tf.image_summary("Phis", tf.expand_dims(phis, 3), max_images=5)

	#replace with ACOS!!!!!	
	#phis = tf.cos(phis)

#use 2 linears layer to get the 10 outputs (NO softmax with this cross-entropy -> it already includes it!)
with tf.name_scope('FullyConnected') as scope:
	linear0 = tf.add(tf.matmul(tf.reshape(phis, [-1, i_numRows * i_numCols * i_numChannels]), linearW0), linearb0)
	logits = tf.add(tf.matmul(linear0, linearW1), linearb1)

#define loss
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, label_batch, name='xentropy')

with tf.name_scope('QConstraints') as scope:
	Q_loss = tf.scalar_mul(alpha, tf.reduce_sum(tf.pow(tf.add(tf.matmul(tf.transpose(Q), Q), tf.neg(Q_identity)), 2)))

loss = tf.add(Q_loss, tf.reduce_mean(cross_entropy))

#save some stuff to tensorboard
lossSummary = tf.scalar_summary("Loss", loss)
sparsity_Q_summary = tf.histogram_summary("Q Value Histogram", Q)

filter_Q_summary = tf.image_summary("ColsQAsFilters", tf.reshape(tf.transpose(Q), [9, 3, 3, 1]), max_images=9)
Q_summary = tf.image_summary("Q", tf.reshape(Q, [1, 9, 9, 1]), max_images=1)

#determine accuracy
with tf.name_scope('Accuracy') as scope:
	correct_prediction = tf.equal(label_batch, tf.argmax(logits,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	accuracySummary = tf.scalar_summary("Accuracy", accuracy)

#define optimiser
step = tf.Variable(0, name='global_step', trainable=False)
optim = tf.train.AdamOptimizer(learning_rate)
train_op = optim.minimize(loss, global_step=step)

#Graph construction, session etc. -------------------------------------------------
sess = tf.Session()

#initialise all variables
init = tf.initialize_all_variables()
sess.run(init)

#set up training loop
coordinator = tf.train.Coordinator()
threads_io = tf.train.start_queue_runners(sess=sess, coord=coordinator)

#create necessary stuff for summaries
summary_op = tf.merge_all_summaries()
summary = tf.train.SummaryWriter('logs', tf.get_default_graph())

#generate tensors to feed from python
inputQ_identity = np.identity(9)

loop_step = 0
try:
	while not coordinator.should_stop():
		# Run training steps or whatever
		result, currentLoss, summaryResult = sess.run([train_op, loss, summary_op], feed_dict={Q_identity: inputQ_identity})
		summary.add_summary(summaryResult, loop_step)
		print(currentLoss)
		loop_step += 1
		#or for a different session: graph_def=sess.graph_def
except tf.errors.OutOfRangeError:
	print('Done training -- epoch limit reached')
finally:
	# When done, ask the threads to stop.
	coordinator.request_stop()

# Wait for threads to finish.
coordinator.join(threads_io)

#get Q as numpy tensor
Q = Q.eval(session=sess)
sess.close()

#print some stuff
print("Values of Q:")
print(Q)

print("Eigenvalues of Q:")
Q_eig = np.linalg.eig(Q)
print(Q_eig)





