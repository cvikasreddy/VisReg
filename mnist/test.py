import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.layers import fc_layer, conv_layer, pool_layer, dropout_layer, change_to_fc, batch_norm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot = True)
batch_size = 64

train_size = mnist.train.images.shape[0]
X_mean = mnist.train.images.mean(axis=0)

sess = tf.InteractiveSession()

with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, shape=[None, 28*28])
	y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('phase'):
    phase_train = tf.placeholder(tf.bool, name='phase_train')

def fc_net(x):
	fc1 = fc_layer(x, 28*28, 100)
	fc2 = fc_layer(fc1, 100, 100)
	out = fc_layer(fc2, 100, 10, act='softmax')
	return out

def fc_bn_net(x):
	fc1 = fc_layer(x, 28*28, 100)
	fc1 = batch_norm(fc1)

	fc2 = fc_layer(fc1, 100, 100)
	fc2 = batch_norm(fc2)
	
	out = fc_layer(fc2, 100, 10, act='softmax')
	return out

def fc_bn_do_net(x):
	fc1 = fc_layer(x, 28*28, 100)
	fc1 = batch_norm(fc1)
	#fc1 = dropout_layer(fc1, 0.1)

	fc2 = fc_layer(fc1, 100, 100)
	fc2 = batch_norm(fc2)
	fc2 = dropout_layer(fc2, 0.8)
	
	out = fc_layer(fc2, 100, 10, act='softmax')
	return out

def conv_net(x):
	conv1 = conv_layer(x, [3, 3, 3, 64], 64)
	conv2 = conv_layer(conv1, [3, 3, 64, 64], 64)
	pool1 = pool_layer(conv2)

	reshaped, reshaped_shape = change_to_fc(pool1)
	fc1 = fc_layer(reshaped, reshaped_shape, 1024)         #input_tensor, input_shape, output_shape, std, name, act
	out = fc_layer(fc1, 1024, 10, act='softmax')
	return out

def conv4_net(x):
	conv1 = conv_layer(x, [3, 3, 3, 64], 64)
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
	conv1 = conv_layer(x, [3, 3, 3, 64], 64)
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

pred = fc_bn_net(x)

with tf.name_scope('loss'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	# with tf.name_scope('weight_decay'):
	# 	trainable_vars = [
 #                                #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv_layer'),
 #                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer')
 #            ]
 #        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars])
	# cost = cost + 0.0005*L2_loss
               
with tf.name_scope('accuracy'):
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

run = 'fc3_bn_do_l2'

tf.initialize_all_variables().run()

saver.restore(sess, 'models/'+run)

test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images - X_mean, y: mnist.test.labels, phase_train : False})
print "Accuracy on MNIST: " + str(test_accuracy)


"""
	Getting output on Kaggle dataset
"""
# read features
def read_features_from_csv(filename, usecols=range(1,785)):
    features = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=usecols, dtype=np.float32)
    features = np.divide(features, 255.0) # scale 0..255 to 0..1
    return features

# read labels and convert them to 1-hot vectors
def read_labels_from_csv(filename):
    labels_orig = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=0, dtype=np.int)
    labels = np.zeros([len(labels_orig), 10])
    labels[np.arange(len(labels_orig)), labels_orig] = 1
    labels = labels.astype(np.float32)
    return labels

# generate batches
def generate_batch(features, labels, batch_size):
    batch_indexes = np.random.random_integers(0, len(features) - 1, batch_size)
    batch_features = features[batch_indexes]
    batch_labels = labels[batch_indexes]
    return (batch_features, batch_labels)

x_test = read_features_from_csv('data/kaggle/test.csv', usecols=None)
y_test = sess.run(pred, feed_dict={x: x_test - X_mean, phase_train : False})

y_test = np.argmax(y_test, axis=1)
y_test = [np.arange(1, 1 + len(y_test)), y_test]
y_test = np.transpose(y_test)

np.savetxt('data/kaggle/kaggle_submission.csv', y_test, fmt='%i,%i', header='ImageId,Label', comments='')