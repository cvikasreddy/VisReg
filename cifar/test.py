import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf

from utils.data import CifarDataLoader
batch_size = 64
cifardataloader = CifarDataLoader(batch_size=64)
train_size = cifardataloader.train_size

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])

def fc_layer(x, in_dim, out_dim, act='relu'):
	weights = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.01))
	biases = tf.Variable(tf.constant(0.0, shape=[out_dim]))
	activations = tf.nn.relu(tf.matmul(x, weights) + biases) if(act == 'relu') else tf.nn.softmax(tf.matmul(x, weights) + biases)
	return activations

def conv_layer(x, kernel_shape, out_dim):
	weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01))
	biases = tf.Variable(tf.constant(0.0, shape=[out_dim]))
	activations = tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME'), biases))
	return activations

def pool_layer(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
    
def dropout_layer(x, prob):
	return tf.nn.dropout(x, prob)

def change_to_fc(x):     
    inp_shape = x.get_shape()
    dim = np.prod(np.array(inp_shape.as_list()[1:]))
    reshape = tf.reshape(x, [-1, dim])   
    return reshape, dim

def fc_net(x):
	fc1 = fc_layer(x, 32*32*3, 1024)
	fc2 = fc_layer(fc1, 1024, 128)
	out = fc_layer(fc2, 128, 10, act='softmax')
	return out

def conv_net(x):
	conv1 = conv_layer(x, [3, 3, 3, 64], 64)
	conv2 = conv_layer(conv1, [3, 3, 64, 64], 64)
	pool1 = pool_layer(conv2)

	reshaped, reshaped_shape = change_to_fc(pool1)
	fc1 = fc_layer(reshaped, reshaped_shape, 1024)         #input_tensor, input_shape, output_shape, std, name, act
	out = fc_layer(fc1, 1024, 10)
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
	out = fc_layer(fc1, 1024, 10)
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
	out = fc_layer(drop4, 1024, 10)
	return out

def vgg_net(x):
	conv1 = conv_layer(x, [3, 3, 3, 64], 64)
	drop1 = dropout_layer(conv1, 0.3)
	conv2 = conv_layer(drop1, [3, 3, 64, 64], 64)
	pool1 = pool_layer(conv2)

	conv3 = conv_layer(pool1, [3, 3, 64, 128], 128)
	drop2 = dropout_layer(conv3, 0.4)
	conv4 = conv_layer(drop2, [3, 3, 128, 128], 128)
	pool2 = pool_layer(conv4)

	conv5 = conv_layer(pool2, [3, 3, 128, 256], 256)
	drop3 = dropout_layer(conv5, 0.4)
	conv6 = conv_layer(drop3, [3, 3, 256, 256], 256)
	drop4 = dropout_layer(conv6, 0.4)
	conv7 = conv_layer(drop4, [3, 3, 256, 256], 256)
	pool3 = pool_layer(conv7)

	conv8 = conv_layer(pool3, [3, 3, 256, 512], 512)
	drop5 = dropout_layer(conv8, 0.4)
	conv9 = conv_layer(drop5, [3, 3, 512, 512], 512)
	drop6 = dropout_layer(conv9, 0.4)
	conv10 = conv_layer(drop6, [3, 3, 512, 512], 512)
	pool4 = pool_layer(conv10)

	conv11 = conv_layer(pool4, [3, 3, 512, 512], 512)
	drop7 = dropout_layer(conv11, 0.4)
	conv12 = conv_layer(drop7, [3, 3, 512, 512], 512)
	drop8 = dropout_layer(conv12, 0.4)
	conv13 = conv_layer(drop8, [3, 3, 512, 512], 512)
	pool5 = pool_layer(conv13)

	reshaped, reshaped_shape = change_to_fc(pool4)
	drop9 = dropout_layer(reshaped, 0.5)

	fc1 = fc_layer(drop9, reshaped_shape, 512)
	#batch_norm
	drop10 = dropout_layer(fc1, 0.5)
	out = fc_layer(drop10, 512, 10, act=tf.nn.softmax)

	return out

pred = vgg_net(x)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()
run = 'conv4_pool2_do3_fc_2'

tf.initialize_all_variables().run()
saver.restore(sess, 'models/'+run)

test_x = cifardataloader.test_dataset
test_y = cifardataloader.test_labels
loss, acc = sess.run([cost, accuracy], feed_dict = {x : test_x, y : test_y})

print loss, acc*100.0
