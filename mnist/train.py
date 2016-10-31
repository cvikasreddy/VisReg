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
	x_conv = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
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
	fc2 = dropout_layer(fc2, 0.8)

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

pred = conv_net(x)

beta = 0.01
gamma = 0.01

with tf.name_scope('loss'):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	
	# with tf.name_scope('l2_loss'):
	# 	trainable_vars = [
 #                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/weights'),
 #                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/biases'),
 #                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_1/weights'),
 #                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_1/biases'),
 #                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_2/weights'),
 #                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer_2/biases')
 #            ]
	# 	l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars])
	# 	cost = cost + beta*l2_loss

	# with tf.name_scope("vr_cost"):
	# 	cW = tf.Variable([-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0], trainable=False)
	# 	cW = tf.reshape(cW, [3, 3, 1, 1])

	# 	weights_fc1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer/weights')[0]
	# 	temp = tf.reshape(weights_fc1, [-1, 28, 28, 1])

	# 	conved = tf.nn.conv2d(temp, cW, strides=[1, 1, 1, 1], padding='SAME')
	# 	vr_loss = tf.reduce_mean(conved*conved)

	# 	cost = cost + gamma*vr_loss

	cost_summ = tf.scalar_summary('loss', cost)
               
with tf.name_scope('accuracy'):
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	acc_summ = tf.scalar_summary('accuracy', accuracy)

learning_rate = tf.placeholder(tf.float32)
lr_summary = tf.scalar_summary('lr', learning_rate)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()

run = 'conv'
#run = 'fc3_bn_do_0.3_l2_' + str(beta) + '_vr_' + str(gamma)
#run = 'fc3_bn_l2_' + str(beta) + '_vr_' + str(gamma)
print run

train_writer = tf.train.SummaryWriter('logs/' + run + '/train/', sess.graph)
test_writer = tf.train.SummaryWriter('logs/' + run + '/test/')
merged = tf.merge_all_summaries()

tf.initialize_all_variables().run()

step = 0
present_learning_rate = 0.001
try:
	saver.restore(sess, 'models/'+run)
except ValueError:
	print "First time running"
for epoch in range(250):
	avg_loss = 0.0
	avg_acc = 0.0
	num_batches = int(train_size*1.0/batch_size)

	avg_val_loss = 0.0
	avg_val_acc = 0.0
	num_val = 0.0

	if((epoch%75 == 0) and (epoch != 0) and (epoch < 100)):
		present_learning_rate /= 2.0
	if((epoch > 100) and (epoch%50 == 0)):
		present_learning_rate /= 2.0

	f = open('logs/log.txt', 'a')

	for batch_num in range(num_batches):
		step += 1.0

		batch_x, batch_y = mnist.train.next_batch(batch_size)
		batch_x = batch_x.reshape([-1, 28, 28, 1])

		loss, acc, _, summary = sess.run([cost, accuracy, train_step, merged], feed_dict = {x_conv : batch_x, y : batch_y, learning_rate : present_learning_rate, phase_train : True})
		

		avg_loss += loss
		avg_acc += acc
		train_writer.add_summary(summary, step)

		if batch_num % 100 == 0:
			batch_x, batch_y = mnist.test.next_batch(batch_size)
			batch_x = batch_x.reshape([-1, 28, 28, 1])

			summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict = {x_conv : batch_x, y : batch_y, phase_train : False, learning_rate : present_learning_rate}) #here learning_arte is not required but for merged summary it is required
			
			avg_val_loss += loss
			avg_val_acc += acc
			num_val += 1.0
			test_writer.add_summary(summary, step)

			save_path = saver.save(sess, 'models/'+run)

	avg_acc = avg_acc*100.0/num_batches
	avg_loss = avg_loss*1.0/num_batches
	f.write("Train Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_acc) + ' Loss: ' + str(avg_loss) + '\n')
	print "Train Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_acc) + '0'*(15-len(str(avg_acc))) + ' Loss: ' + str(avg_loss)

	avg_val_acc = avg_val_acc*100.0/num_val
	avg_val_loss = avg_val_loss*1.0/num_val
	f.write("Val   Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_val_acc) + ' Loss: ' + str(avg_val_loss) + '\n')
	print "Val   Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_val_acc) + '0'*(15-len(str(avg_val_acc))) + ' Loss: ' + str(avg_val_loss)

	f.close()

test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images - X_mean, y: mnist.test.labels, phase_train : False})
print "Testing Accuracy: " + str(test_accuracy)

q = open('results.txt', 'a')
q.write(str(beta) + " " + str(gamma) + " " + str(test_accuracy) + '\n')
q.close()

train_writer.close()
test_writer.close()


"""

	3-fc, 0.1:
			Epoch: 100 Acc: 99.3669965076 Loss: 1.46743904309
			Epoch: 200 Acc: 99.3924621653 Loss: 1.46718711362
			Epoch: 300 Acc: 99.4124708964 Loss: 1.4669823999
			Epoch: 400 Acc: 99.4142898719 Loss: 1.46696007626
			Epoch: 500 Acc: 99.4142898719 Loss: 1.46695895953

			Testing Accuracy: 97.88

	3-fc, bn, 0.1:
			Epoch: 50 Acc: 99.745343422600 Loss: 1.46396062274
			Epoch: 100 Acc: 99.779903958100 Loss: 1.46351644879
			Epoch: 150 Acc: 99.781722933600 Loss: 1.46345761297
			Epoch: 200 Acc: 99.776266007000 Loss: 1.46351254902
			Epoch: 250 Acc: 99.787179860300 Loss: 1.46341660678

	3-fc, bn, do:0.3, 0.1:
			Epoch: 50 Acc: 91.327124563400 Loss: 1.54828760885
			Epoch: 100 Acc: 92.642243888200 Loss: 1.53542097248
			Epoch: 150 Acc: 92.966021536700 Loss: 1.53214963934
			Epoch: 200 Acc: 93.198850407500 Loss: 1.52984198536

	3-fc, bn, do: => 0.01: (dropout after fc2 and before out)
			0.5  => 97.96
			0.9  => 98.06
			0.95 => 98.08
			0.99 => 98.22

	3-fc, bn, do: => 0.01: (dropout after fc2 and before out) (xavier init, random normal {0.3/(in_dim+out_dim)})
			0.99 => 98.2

	4-fc, bn, do: => 0.01: (dropout after fc2 and before out) (xavier init, random normal {0.3/(in_dim+out_dim)})
			0.99 => 97.61
			0.5, 0.99 => 97.04

	3-fc, bn, do:, l2: => 0.01: (dropout after fc2 and before out) (xavier init, random normal {0.3/(in_dim+out_dim)})
			0.99, 0.001 => 97.87
			0.8 , 0.001 => 97.98
			0   , 0.001 => 98.1
			0.8 , 0.01  => 98.1

	3-fc, bn, l2:, vr: => 0.01: (dropout after fc2 and before out) (xavier init, random normal {0.3/(in_dim+out_dim)})
			0.0 0.0 0.9803
			0.0 0.02 0.9816
			0.01 0.03 0.9809

"""