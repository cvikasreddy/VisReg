import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from tqdm import tqdm

import numpy as np
import tensorflow as tf

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

def fc_layer(x, in_dim, out_dim, act='relu'):
	with tf.name_scope('fc_layer'):
		weights = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.01))
		biases = tf.Variable(tf.constant(0.0, shape=[out_dim]))
		activations = tf.nn.relu(tf.matmul(x, weights) + biases) if(act == 'relu') else tf.nn.softmax(tf.matmul(x, weights) + biases)
	return activations

def conv_layer(x, kernel_shape, out_dim):
	with tf.name_scope('conv_layer'):
		weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01))
		biases = tf.Variable(tf.constant(0.0, shape=[out_dim]))
		activations = tf.nn.relu(tf.add(tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME'), biases))
	return activations

def pool_layer(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')
    
def dropout_layer(x, prob):
	with tf.name_scope('dropout_layer'):
		return tf.nn.dropout(x, prob)

def change_to_fc(x): 
	with tf.name_scope('reshape_layer'):    
		inp_shape = x.get_shape()
		dim = np.prod(np.array(inp_shape.as_list()[1:]))
		reshape = tf.reshape(x, [-1, dim])   
	return reshape, dim

def batch_norm(x, phase_train, n_out):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

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

learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

saver = tf.train.Saver()

run = 'vgg_small'
merged = tf.merge_all_summaries()

tf.initialize_all_variables().run()

train_writer = tf.train.SummaryWriter('logs/' + run + '/train/', sess.graph)
#train_writer = tf.scalar_summary('logs/' + run + '/train/', sess.graph)
test_writer = tf.train.SummaryWriter('logs/' + run + '/test/')

step = 0
present_learning_rate = 0.001
saver.restore(sess, 'models/'+run)
for epoch in range(500):
	cifardataloader.reset_index()
	avg_loss = 0.0
	avg_acc = 0.0
	num_batches = int(train_size*1.0/batch_size)

	avg_val_loss = 0.0
	avg_val_acc = 0.0
	num_val = 0.0

	if((epoch%100 == 0) and (epoch != 0)):
		present_learning_rate /= 2.0

	f = open('logs/log.txt', 'a')

	for batch_num in range(num_batches):
		step += 1.0

		batch = cifardataloader.next_batch()
		batch_x = batch['images']#.reshape([-1, 32*32*3])
		batch_y = batch['labels']

		#loss, acc, _, summary = sess.run([cost, accuracy, train_step, merged], feed_dict = {x : batch_x, y : batch_y, learning_rate : present_learning_rate, phase_train : True})
		loss, acc, _ = sess.run([cost, accuracy, train_step], feed_dict = {x : batch_x, y : batch_y, learning_rate : present_learning_rate, phase_train : True})

		avg_loss += loss
		avg_acc += acc
		#train_writer.add_summary(summary, step)

		if batch_num % 100 == 0:
			batch = cifardataloader.next_batch(data_type='val')
			batch_x = batch['images']#.reshape([-1, 32*32*3])
			batch_y = batch['labels']

			#summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict = {x : batch_x, y : batch_y, phase_train : False})
			loss, acc = sess.run([cost, accuracy], feed_dict = {x : batch_x, y : batch_y, phase_train : False})
			
			avg_val_loss += loss
			avg_val_acc += acc
			num_val += 1.0
			#test_writer.add_summary(summary, step)

			save_path = saver.save(sess, 'models/'+run + '1')

	avg_acc = avg_acc*100.0/num_batches
	avg_loss = avg_loss*1.0/num_batches
	f.write("Train Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_acc) + ' Loss: ' + str(avg_loss) + '\n')
	print "Train Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_acc) + '0'*(15-len(str(avg_acc))) + ' Loss: ' + str(avg_loss)

	avg_val_acc = avg_val_acc*100.0/num_val
	avg_val_loss = avg_val_loss*1.0/num_val
	f.write("Val   Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_val_acc) + ' Loss: ' + str(avg_val_loss) + '\n')
	print "Val   Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_val_acc) + '0'*(15-len(str(avg_val_acc))) + ' Loss: ' + str(avg_val_loss)

	f.close()

train_writer.close()
test_writer.close()

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