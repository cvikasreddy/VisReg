from utils.data import CifarDataLoader
batch_size = 128
cifardataloader = CifarDataLoader(batch_size=batch_size)
train_size = cifardataloader.train_size
test_size = cifardataloader.test_size

print "Data Loaded"

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.layers import fc_layer, conv_layer, pool_layer, dropout_layer, change_to_fc, batch_norm
sess = tf.InteractiveSession()

with tf.name_scope('input'):
	with tf.name_scope('x'):
		x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
	with tf.name_scope('y'):
		y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('phase'):
    phase_train = tf.placeholder(tf.bool, name='phase_train')

with tf.name_scope('dropout_prob'):
	keep_prob = tf.placeholder(tf.float32)

def conv_net(x):
	conv1 = conv_layer(x, [5, 5, 3, 64])
	conv1 = dropout_layer(conv1, keep_prob)

	pool1 = pool_layer(conv1, ksize=[1, 3, 3, 1])

	conv2 = conv_layer(pool1, [5, 5, 64, 64])
	conv2 = dropout_layer(conv2, keep_prob)

	pool2 = pool_layer(conv2, ksize=[1, 3, 3, 1])

	conv3 = conv_layer(pool2, [5, 5, 64, 64])
	conv3 = dropout_layer(conv3, keep_prob)

	pool3 = pool_layer(conv3, ksize=[1, 3, 3, 1])

	reshaped, reshaped_shape = change_to_fc(pool3)

	fc1 = fc_layer(reshaped, reshaped_shape, 384)
	fc1 = dropout_layer(fc1, keep_prob)

	fc2 = fc_layer(fc1, 384, 192)
	fc2 = dropout_layer(fc2, keep_prob)

	out = fc_layer(fc2, 192, 10, act='softmax', std=0.005)
	return out

pred = conv_net(x)

beta = 0.004   #L2 loss coefficient
gamma = 0.001  #VR2 loss coefficient
theta = 0.0    #VR1 loss coefficient
num_epochs = 1300

with tf.name_scope('loss'):
	with tf.name_scope('cross_entropy_loss'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	with tf.name_scope('weight_decay'):
		trainable_vars = [
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/weights'),
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_1/weights')
            ]
		l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars])
		cost = cost + beta*l2_loss

	with tf.name_scope("vr_cost"):
		cW = tf.constant([-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0])
		cW = tf.reshape(cW, [3, 3, 1, 1])

		weights_fc1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/weights')[0]
		temp = tf.reshape(weights_fc1, [-1, 32, 32, 1])

		conved = tf.nn.conv2d(temp, cW, strides=[1, 1, 1, 1], padding='SAME')
		vr_loss = tf.reduce_mean(conved*conved)
		v1_loss = tf.reduce_mean(conved*tf.sign(conved))

		cost = cost + gamma*vr_loss
		cost = cost + theta*v1_loss

	cost_summ = tf.scalar_summary('loss', cost)
               
with tf.name_scope('accuracy'):
	with tf.name_scope('correct_pred'):
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	with tf.name_scope('accuracy'):
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	acc_summ = tf.scalar_summary('accuracy', accuracy)

learning_rate = tf.placeholder(tf.float32)
lr_summ = tf.scalar_summary('lr', learning_rate)
saver = tf.train.Saver()

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

run = 'conv_net_vr2_' + str(gamma) + '_vr1_' + str(theta)  + '_l2_' + str(beta)
merged = tf.merge_all_summaries()

tf.initialize_all_variables().run()

train_writer = tf.train.SummaryWriter('logs/' + run + '/train/', sess.graph)
test_writer = tf.train.SummaryWriter('logs/' + run + '/test/')

step = 0
present_learning_rate = 0.01
best_acc = 0.00

for epoch in range(num_epochs):
	cifardataloader.reset_index()
	avg_loss = 0.0
	avg_acc = 0.0
	num_batches = int(train_size*1.0/batch_size)

	if((epoch >= 100) and (epoch%500 == 0)):
		present_learning_rate /= 1.3

	for batch_num in range(num_batches):
		step += 1.0

		batch = cifardataloader.next_batch()
		batch_x = batch['images']
		batch_y = batch['labels']

		loss, acc, _, summary = sess.run([cost, accuracy, train_step, merged], feed_dict = {x : batch_x, y : batch_y, learning_rate : present_learning_rate, keep_prob : 0.9, phase_train : True})
		
		avg_loss += loss
		avg_acc += acc
		train_writer.add_summary(summary, step)

	avg_acc = avg_acc*100.0/num_batches
	avg_loss = avg_loss*1.0/num_batches
	print "Train Epoch: " + str(epoch + 1) + ' Acc: ' + str(round(avg_acc, 6))  + ' Loss: ' + str(round(avg_loss, 6))

	#Test every 3rd epoch
	if((epoch%3 == 0) and (epoch != 0)): 

		avg_val_loss = 0.0
		avg_val_acc = 0.0
		num_batches = int(test_size*1.0/batch_size)

		for batch_num in range(num_batches):
			batch = cifardataloader.next_batch(data_type='test')
			batch_x = batch['images']#.reshape([-1, 32*32*3])
			batch_y = batch['labels']

			loss, acc, summary = sess.run([cost, accuracy, merged], feed_dict = {x : batch_x, y : batch_y, learning_rate : present_learning_rate, keep_prob : 1.0, phase_train : False})
			
			avg_val_loss += loss
			avg_val_acc += acc
			test_writer.add_summary(summary, step)

		avg_val_acc = avg_val_acc*100.0/num_batches
		avg_val_loss = avg_val_loss*1.0/num_batches
		print " "*60 + "Val   Epoch: " + str(epoch + 1) + ' Acc: ' + str(round(avg_val_acc, 6)) + ' Loss: ' + str(round(avg_val_loss, 6))

		if avg_val_acc > best_acc:
			save_path = saver.save(sess, 'models/'+run)
			best_acc = avg_val_acc

train_writer.close()
test_writer.close()

print "Training Complete"
