import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.layers import fc_layer, conv_layer, pool_layer, dropout_layer, change_to_fc, batch_norm

from utils.data import CifarDataLoader
batch_size = 128
cifardataloader = CifarDataLoader(batch_size=batch_size)
train_size = cifardataloader.train_size

sess = tf.InteractiveSession()

with tf.name_scope('input'):
	x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
	y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('phase'):
    phase_train = tf.placeholder(tf.bool, name='phase_train')

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

	# conv11 = conv_layer(pool4, [3, 3, 512, 512], 512)
	# drop7 = dropout_layer(conv11, 0.4)
	# conv12 = conv_layer(drop7, [3, 3, 512, 512], 512)
	# drop8 = dropout_layer(conv12, 0.4)
	# conv13 = conv_layer(drop8, [3, 3, 512, 512], 512)
	# pool5 = pool_layer(conv13)

	reshaped, reshaped_shape = change_to_fc(pool4)
	drop9 = dropout_layer(reshaped, 0.5)

	fc1 = fc_layer(drop9, reshaped_shape, 512)
	fc1_bc = batch_norm(fc1, phase_train, 512)
	drop10 = dropout_layer(fc1_bc, 0.5)
	out = fc_layer(drop10, 512, 10, act='softmax')

	return out

pred = vgg_net(x)

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

run = 'vgg_small1'
#run = 'conv4_pool2_do3_fc_2'

tf.initialize_all_variables().run()
saver.restore(sess, 'models/'+run)

test_x = cifardataloader.test_dataset
test_y = cifardataloader.test_labels

acc = sess.run([accuracy], feed_dict = {x : test_x, y : test_y, phase_train : False})
print "Test Accuracy: " + str(acc)

#loss, acc = sess.run([cost, accuracy], feed_dict = {x : test_x, y : test_y})
#print loss, acc*100.0


"""
	1-fc: 
			Epoch: 100 Acc: 31.8125 Loss: 2.14188584824
			Epoch: 200 Acc: 34.2625 Loss: 2.11888634815
			Epoch: 300 Acc: 35.1975 Loss: 2.1101393671
			Epoch: 400 Acc: 35.685 Loss: 2.10513252087
			Epoch: 500 Acc: 36.0275 Loss: 2.10157400761	

	3-fc:
			Epoch: 100 Acc: 31.8575 Loss: 2.14869181023
			Epoch: 200 Acc: 34.8875 Loss: 2.11716543274
			Epoch: 300 Acc: 36.675 Loss: 2.10002483406
			Epoch: 400 Acc: 37.8575 Loss: 2.08819365063
			Epoch: 500 Acc: 38.7975 Loss: 2.07867454433

	2-conv, 1-pool, 2-fc, 0.00001:
			Epoch: 50 Acc: 36.29 Loss: 1.88115429344
			Epoch: 100 Acc: 41.8475 Loss: 1.73602261906
			Epoch: 150 Acc: 46.1275 Loss: 1.62426984196
			Epoch: 200 Acc: 49.69 Loss: 1.52813880062
			Epoch: 250 Acc: 52.8125 Loss: 1.44098321095
			Epoch: 300 Acc: 55.445 Loss: 1.3660812871
			Epoch: 350 Acc: 57.6825 Loss: 1.29974139624

	4-conv, 2-pool, 2-fc, 0.001:
			Epoch: 5 Acc: 15.4175 Loss: 2.25447894402
			Epoch: 10 Acc: 40.19 Loss: 1.70226551914
			Epoch: 15 Acc: 53.8275 Loss: 1.31818491688
			Epoch: 20 Acc: 62.22 Loss: 1.08816788721
			Epoch: 25 Acc: 71.885 Loss: 0.821931289482
			Epoch: 30 Acc: 83.0975 Loss: 0.508629581261
			Epoch: 35 Acc: 91.535 Loss: 0.25388867346

	4-conv, 2-pool, 2-fc, 4-do, 0.001:
			Epoch: 10 Acc: 29.16 Loss: 1.93795111866
			Epoch: 20 Acc: 39.57 Loss: 1.66450134621
			Epoch: 30 Acc: 45.935 Loss: 1.48454616661
			Epoch: 50 Acc: 55.015 Loss: 1.25829030704
			Epoch: 60 Acc: 58.325 Loss: 1.16739753323

	vgg-net, 0.3:


"""
