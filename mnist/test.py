import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.layers import fc_layer, conv_layer, pool_layer, dropout_layer, change_to_fc, batch_norm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot = True)

batch_size = 64
conv = True   #Use convolutions or not

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
	fc1 = fc_layer(x, 28*28, 1000)
	
	fc2 = fc_layer(fc1, 1000, 1000)
	fc2 = dropout_layer(fc2, 0.7)

	fc3 = fc_layer(fc2, 1000, 1000)
	fc3 = dropout_layer(fc3, 0.7)

	out = fc_layer(fc3, 1000, 10, act='softmax')
	return out

def conv_net(x):
	conv1 = conv_layer(x, [3, 3, 1, 64], 64)
	conv1 = dropout_layer(conv1, 1.0)

	conv2 = conv_layer(conv1, [3, 3, 64, 64], 64)
	conv2 = dropout_layer(conv2, 0.9)

	pool1 = pool_layer(conv2)

	reshaped, reshaped_shape = change_to_fc(pool1)
	fc1 = fc_layer(reshaped, reshaped_shape, 1024)         
	fc1 = dropout_layer(fc1, 0.9)

	out = fc_layer(fc1, 1024, 10, act='softmax')
	return out

if conv == True:
	pred = conv_net(x)
else:
	pred = fc_net(x)

beta = 0.0     #L2 loss coefficient
gamma = 0.0    #VR2 loss coefficient
theta = 0.0    #VR1 loss coefficient

with tf.name_scope('loss'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	
	with tf.name_scope('l2_loss'):
		trainable_vars = [
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/weights'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/biases'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_1/weights'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_1/biases'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_2/weights'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_2/biases'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_3/weights'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_3/biases')
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
		v1_loss = tf.reduce_mean(conved*tf.sign(conved))

		cost = cost + gamma*vr_loss
		cost = cost + theta*v1_loss

	cost_summ = tf.scalar_summary('loss', cost)
               
with tf.name_scope('accuracy'):
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	acc_summ = tf.scalar_summary('accuracy', accuracy)

saver = tf.train.Saver()

if conv == True:
	run = 'conv_vr2_' + str(gamma) + '_vr1_' + str(theta) + '_l2_' + str(beta) 
else:
	run = 'fc_vr2_' + str(gamma) + '_vr1_' + str(theta) + '_l2_' + str(beta) 
print run

tf.initialize_all_variables().run()

saver.restore(sess, 'models/'+run)

#Testing
val_acc = []
val_loss = []
val_loss_l2 = []
val_loss_vr = []
val_loss_v1 = []
num_batches = int(test_size*1.0/batch_size)

for batch_num in range(num_batches):
	batch_x, batch_y = mnist.test.next_batch(batch_size)
	if conv == True:
		batch_x = batch_x.reshape([-1, 28, 28, 1])

	loss, loss_l2, loss_vr, loss_v1, acc = sess.run([cost, l2_loss, vr_loss, v1_loss, accuracy], feed_dict = {x : batch_x - X_mean, y : batch_y, phase_train : False})
	val_acc.append(acc)
	val_loss.append(loss)
	val_loss_l2.append(loss_l2)
	val_loss_vr.append(loss_vr)
	val_loss_v1.append(loss_v1)

avg_val_acc = np.mean(np.array(val_acc)*100.0)
avg_val_loss = np.mean(val_loss)
avg_val_loss_l2 = np.mean(val_loss_l2)
avg_val_loss_vr = np.mean(val_loss_vr)
avg_val_loss_v1 = np.mean(val_loss_v1)

std_val_acc = np.std(np.array(val_acc)*100.0)
std_val_loss = np.std(val_loss)
std_val_loss_l2 = np.std(val_loss_l2)
std_val_loss_vr = np.std(val_loss_vr)
std_val_loss_v1 = np.std(val_loss_v1)

print str(round(avg_val_acc, 6)) + '+' + str(std_val_acc)
print str(round(avg_val_loss, 6)) + '+' + str(std_val_loss)
print str(round(avg_val_loss_l2, 6)) + '+' + str(std_val_loss_l2)
print str(round(avg_val_loss_vr, 6)) + '+' + str(std_val_loss_vr)
print str(round(avg_val_loss_v1, 6)) + '+' + str(std_val_loss_v1)
