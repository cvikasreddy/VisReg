# from optparse import OptionParser
# parser = OptionParser()
# parser.add_option("-2", "--l2_coeff")
# parser.add_option("-v", "--vr_coeff")
# #parser.add_option("-d", "--drop")
# (options, args) = parser.parse_args()

# options = vars(options)
# beta = float(options['l2_coeff'])
# gamma = float(options['vr_coeff'])
# #dropout_prob = float(options['drop'])


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.layers import fc_layer, conv_layer, pool_layer, dropout_layer, change_to_fc, batch_norm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot = True)

batch_size = 64
conv = False

train_size = mnist.train.images.shape[0]
test_size = mnist.test.images.shape[0]
X_mean = mnist.train.images.mean(axis=0)
if conv == True:
	X_mean = X_mean.reshape([-1, 28, 28, 1])

sess = tf.InteractiveSession()

with tf.name_scope('input'):
	if conv == True:
		x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
	else:
		x = tf.placeholder(tf.float32, shape=[None, 28*28])
	y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('phase'):
    phase_train = tf.placeholder(tf.bool, name='phase_train')

def fc_net(x):
	fc1 = fc_layer(x, 28*28, 100, bn=False)
	fc2 = fc_layer(fc1, 100, 100, bn=False)
	out = fc_layer(fc2, 100, 10, act='softmax', bn=False)
	return out

def fc_bn_net(x):
	fc1 = fc_layer(x, 28*28, 100)
	fc2 = fc_layer(fc1, 100, 100)
	out = fc_layer(fc2, 100, 10, act='softmax')
	return out

def fc_bn_do_net(x):
	fc1 = fc_layer(x, 28*28, 100)
	#fc1 = dropout_layer(fc1, 0.1)

	fc2 = fc_layer(fc1, 100, 100)
	fc2 = dropout_layer(fc2, 0.8)
	
	out = fc_layer(fc2, 100, 10, act='softmax')
	return out

def conv_net(x):
	conv1 = conv_layer(x_conv, [3, 3, 1, 64], 64)
	conv2 = conv_layer(conv1, [3, 3, 64, 64], 64)
	pool1 = pool_layer(conv2)

	reshaped, reshaped_shape = change_to_fc(pool1)
	fc1 = fc_layer(reshaped, reshaped_shape, 1024)         #input_tensor, input_shape, output_shape, std, name, act
	out = fc_layer(fc1, 1024, 10, act='softmax')
	return out

def conv4_net(x):
	conv1 = conv_layer(x_conv, [3, 3, 1, 64], 64)
	conv2 = conv_layer(conv1, [3, 3, 64, 64], 64)
	pool1 = pool_layer(conv2)

	conv3 = conv_layer(pool1, [3, 3, 64, 128], 128)
	conv4 = conv_layer(conv3, [3, 3, 128, 128], 128)
	pool2 = pool_layer(conv4)

	reshaped, reshaped_shape = change_to_fc(pool2)
	fc1 = fc_layer(reshaped, reshaped_shape, 1024)         #input_tensor, input_shape, output_shape, std, name, act
	out = fc_layer(fc1, 1024, 10, act='softmax')
	return out

def conv4_do_net(x):
	conv1 = conv_layer(x_conv, [3, 3, 1, 64], 64)
	drop1 = dropout_layer(conv1, 0.3)
	conv2 = conv_layer(drop1, [3, 3, 64, 64], 64)
	pool1 = pool_layer(conv2)

	conv3 = conv_layer(pool1, [3, 3, 64, 128], 128)
	drop2 = dropout_layer(conv3, 0.4)
	conv4 = conv_layer(drop2, [3, 3, 128, 128], 128)
	pool2 = pool_layer(conv4)

	reshaped, reshaped_shape = change_to_fc(pool2)
	drop3 = dropout_layer(reshaped, 0.5)
	fc1 = fc_layer(drop3, reshaped_shape, 1024)         #input_tensor, input_shape, output_shape, std, name, act
	drop4 = dropout_layer(fc1, 0.5)
	out = fc_layer(drop4, 1024, 10, act='softmax')
	return out

if conv == True:
	pred = conv_net(x)
else:
	pred = fc_bn_net(x)

beta = 0.0
gamma = 0.0
num_epochs = 200

with tf.name_scope('loss'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	
	with tf.name_scope('l2_loss'):
		trainable_vars = [
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/weights'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/biases'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_1/weights'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_1/biases'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_2/weights'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_2/biases')
            ]
		l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars])
		cost = cost + beta*l2_loss

	with tf.name_scope("vr_cost"):
		cW = tf.Variable([-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0], trainable=False)
		cW = tf.reshape(cW, [3, 3, 1, 1])

		weights_fc1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/weights')[0]
		temp = tf.reshape(weights_fc1, [-1, 28, 28, 1])

		conved = tf.nn.conv2d(temp, cW, strides=[1, 1, 1, 1], padding='SAME')
		vr_loss = tf.reduce_mean(conved*conved)

		cost = cost + gamma*vr_loss

	cost_summ = tf.scalar_summary('loss', cost)
               
with tf.name_scope('accuracy'):
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	acc_summ = tf.scalar_summary('accuracy', accuracy)

saver = tf.train.Saver()

if conv == True:
	run = 'conv_vr_' + str(gamma) + '_l2_' + str(beta) + '_epochs_' + str(num_epochs)
else:
	run = 'fc_vr_' + str(gamma) + '_l2_' + str(beta) + '_epochs_' + str(num_epochs)
print run

tf.initialize_all_variables().run()

try:
	saver.restore(sess, 'models/'+run)
except ValueError:
	print "First time running"

if conv == True:
	test_x = mnist.test.images.reshape(-1, 28, 28, 1)[:5000]
else:
	test_x = mnist.test.images[:5000]
test_y = mnist.test.labels[:5000]
test_accuracy_1 = sess.run(accuracy, feed_dict={x: test_x - X_mean, y: test_y, phase_train : False})
#print "Testing Accuracy: " + str(test_accuracy)

if conv == True:
	test_x = mnist.test.images.reshape(-1, 28, 28, 1)[5000:]
else:
	test_x = mnist.test.images[5000:]
test_y = mnist.test.labels[5000:]
test_accuracy = 0.5*(test_accuracy_1 + sess.run(accuracy, feed_dict={x: test_x - X_mean, y: test_y, phase_train : False}))
print "Testing Accuracy: " + str(test_accuracy)

q = open('results.txt', 'a')
q.write(str(beta) + " " + str(gamma) + " " + str(test_accuracy) + '\n')
q.close()



def read_features_from_csv(filename, usecols=range(1,785)):
    features = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=usecols, dtype=np.float32)
    features = np.divide(features, 255.0) # scale 0..255 to 0..1
    return features

test_features = read_features_from_csv('data/kaggle/test.csv', usecols=None)
test_features = test_features.reshape([-1, 784])
print test_features.shape
readout = sess.run(pred, feed_dict={x: test_features - X_mean, phase_train : False})
#readout = model.get_readout(test_features)
readout = np.argmax(readout, axis=1)
readout = [np.arange(1, 1 + len(readout)), readout]
readout = np.transpose(readout)
np.savetxt('data/kaggle/kaggle_submission.csv', readout, fmt='%i,%i', header='ImageId,Label', comments='')