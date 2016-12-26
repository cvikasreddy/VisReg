# from optparse import OptionParser
# parser = OptionParser()
# parser.add_option("-2", "--l2_coeff")
# parser.add_option("-v", "--vr_coeff")
# (options, args) = parser.parse_args()

# options = vars(options)
# beta = float(options['l2_coeff'])
# gamma = float(options['vr_coeff'])


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.layers import fc_layer, conv_layer, pool_layer, dropout_layer, change_to_fc, batch_norm

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot = True)

batch_size = 64
conv = True

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
	fc1 = fc_layer(x, 28*28, 1000)
	fc2 = fc_layer(fc1, 1000, 1000)
	out = fc_layer(fc2, 1000, 10, act='softmax')
	return out

def fc3_bn_net(x):
	fc1 = fc_layer(x, 28*28, 1000)
	#fc1 = dropout_layer(fc1, 0.7)

	fc2 = fc_layer(fc1, 1000, 1000)
	fc2 = dropout_layer(fc2, 0.7)

	fc3 = fc_layer(fc2, 1000, 1000)
	fc3 = dropout_layer(fc3, 0.7)

	out = fc_layer(fc3, 1000, 10, act='softmax')
	return out

def fc_bn_do_net(x):
	fc1 = fc_layer(x, 28*28, 100)
	#fc1 = dropout_layer(fc1, 0.1)

	fc2 = fc_layer(fc1, 100, 100)
	fc2 = dropout_layer(fc2, 0.8)
	
	out = fc_layer(fc2, 100, 10, act='softmax')
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

def le_net(x): #Implementation from tensorflow examples
	conv1 = conv_layer(x, [5, 5, 1, 32])
	conv1 = dropout_layer(conv1, keep_prob)
	pool1 = pool_layer(conv1)

	conv2 = conv_layer(conv1, [5, 5, 32, 64])
	conv2 = dropout_layer(conv2, keep_prob)
	pool2 = pool_layer(conv2)

	reshaped, reshaped_shape = change_to_fc(pool2)
	
	fc1 = fc_layer(reshaped, reshaped_shape, 512)    
	fc1 = dropout_layer(fc1, keep_prob)

	out = fc_layer(fc1, 512, 10, act='softmax')
	return out

def conv4_net(x):
	conv1 = conv_layer(x, [3, 3, 1, 64], 64)
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
	conv1 = conv_layer(x, [3, 3, 1, 64], 64)
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
	#pred = fc_bn_net(x)
	pred = fc3_bn_net(x)

beta = 0.0
gamma = 0.02
theta = 0.0
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
	run = 'conv_1.0_0.9_0.9_vr_' + str(gamma) + '_l2_' + str(beta) + '_epochs_' + str(num_epochs)
else:
	run = 'fc3_do_0.7_0.7_vr_' + str(gamma) + '_l2_' + str(beta) + '_epochs_' + str(num_epochs)
	#run = 'fc3_final_vr_' + str(gamma) + '_l2_' + str(beta) + '_epochs_' + str(num_epochs)
#run = 'v1_0.01'
print run

tf.initialize_all_variables().run()

saver.restore(sess, 'final_models/conv_no_reg/model')
#saver.restore(sess, 'models/'+run)
#saver.save(sess, 'final_models/fc_vr_0.02/model')

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



#Here test data is split into two equal parts and evaluated because of memory constraints of GPU
# if conv == True:
# 	test_x = mnist.test.images.reshape(-1, 28, 28, 1)[:5000]
# else:
# 	test_x = mnist.test.images[:5000]
# test_y = mnist.test.labels[:5000]
# test_accuracy_1, test_loss_1 = sess.run([accuracy, cost], feed_dict={x: test_x - X_mean, y: test_y, phase_train : False})

# if conv == True:
# 	test_x = mnist.test.images.reshape(-1, 28, 28, 1)[5000:]
# else:
# 	test_x = mnist.test.images[5000:]
# test_y = mnist.test.labels[5000:]
# test_accuracy_2, test_loss_2 = sess.run([accuracy, cost], feed_dict={x: test_x - X_mean, y: test_y, phase_train : False})

# test_accuracy = (test_accuracy_1 + test_accuracy_2)*100/2.0
# test_loss = (test_loss_1 + test_loss_2)/2.0
# print test_accuracy
# print test_loss


#Visualizing Filter
# import Image
# from random import randint, shuffle
# examples = 4
# #new_im = Image.new('L', (28*examples, 28*examples))
# new_im = Image.new('L', ((28+2)*examples, (28+2)*examples), color=256)
# vis = weights_fc1.eval()
# print vis.shape
#indices = [815, 487, 701, 641, 525, 371, 705, 21, 179, 711, 143, 251, 283, 923, 979, 347]
#indices = [144, 251, 496, 885, 95, 333, 41, 712, 9, 614, 420, 602, 350, 870, 927, 940]
#shuffle(indices)
# step = 0
# for i in xrange(examples):
# 	x_offset = 0
# 	#for node_index in xrange(examples):
# 	for j in xrange(examples):
# 		#node_index = indices[step]
# 		node_index = randint(0, 999)
# 		#print node_index
# 		step += 1
# 		weights_vis = np.array(list(vis[:, node_index])).reshape(28, 28)
# 		#weights_vis = (255.0 / weights_vis.max() * (weights_vis - weights_vis.min())).astype(np.uint8)
# 		weights_vis = (weights_vis - weights_vis.min())/(weights_vis.max() - weights_vis.min())*255.0
# 		temp = Image.fromarray(weights_vis)
		
# 		new_im.paste(temp, (x_offset, (28+2)*i))
# 		x_offset += (temp.size[0] + 2)

# new_im.save('final_imgs/grid4_no_reg.png')

#Histogram
# node_index = randint(0, 999)
# weights_vis = np.array(list(vis[:, node_index]))

# import matplotlib
# matplotlib.use('Agg')
# import seaborn as sns
# #sns_plot = sns.distplot(weights_vis, bins=100)
# #sns_plot.get_figure().savefig('hist_100.png')

# weights_vis = (weights_vis - weights_vis.min())/(weights_vis.max() - weights_vis.min())*255.0

# sns_plot = sns.distplot(weights_vis, bins=100)
# sns_plot.get_figure().savefig('hist_100_norm.png')
