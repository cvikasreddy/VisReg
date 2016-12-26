from utils.data import CifarDataLoader
batch_size = 128
cifardataloader = CifarDataLoader(batch_size=batch_size)
train_size = cifardataloader.train_size
test_size = cifardataloader.test_size

print "Data Loaded"


# import subprocess, time

# total_memory = 12206
# max_occupied = 10000

# gpu_output = subprocess.check_output(["nvidia-smi"])
# memory_occupied = int(gpu_output.split('MiB / ' + str(total_memory) + 'MiB')[0].split('|')[-1])

# print memory_occupied
# while memory_occupied > max_occupied:
#     gpu_output = subprocess.check_output(["nvidia-smi"])
#     memory_occupied = int(gpu_output.split('MiB / ' + str(total_memory) + 'MiB')[0].split('|')[-1])
#     time.sleep(0.01)
# print "GPU available"


import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.layers import fc_layer, conv_layer, pool_layer, dropout_layer, change_to_fc, batch_norm

print "Import complete"

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.55)
sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
#sess = tf.InteractiveSession()

with tf.name_scope('input'):
	with tf.name_scope('x'):
		x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
	with tf.name_scope('y'):
		y = tf.placeholder(tf.float32, shape=[None, 10])

with tf.name_scope('phase'):
    phase_train = tf.placeholder(tf.bool, name='phase_train')

with tf.name_scope('dropout_prob'):
	keep_prob = tf.placeholder(tf.float32)

def fc_net(x):
	fc1 = fc_layer(x, 32*32*3, 1024)
	fc2 = fc_layer(fc1, 1024, 128)
	out = fc_layer(fc2, 128, 10, act='softmax')
	return out

def conv_net(x):
	conv1 = conv_layer(x, [3, 3, 3, 64])
	conv2 = conv_layer(conv1, [3, 3, 64, 64])
	pool1 = pool_layer(conv2)

	reshaped, reshaped_shape = change_to_fc(pool1)
	fc1 = fc_layer(reshaped, reshaped_shape, 1024)         #input_tensor, input_shape, output_shape, std, name, act
	out = fc_layer(fc1, 1024, 10, act='softmax')
	return out

def tensor_net(x):
	conv1 = conv_layer(x, [5, 5, 3, 64])
	conv1 = dropout_layer(conv1, keep_prob)

	pool1 = pool_layer(conv1, ksize=[1, 3, 3, 1])

	conv2 = conv_layer(pool1, [5, 5, 64, 64])
	conv2 = dropout_layer(conv2, keep_prob)

	pool2 = pool_layer(conv2, ksize=[1, 3, 3, 1])

	reshaped, reshaped_shape = change_to_fc(pool2)

	fc1 = fc_layer(reshaped, reshaped_shape, 384)
	fc1 = dropout_layer(fc1, keep_prob)

	fc2 = fc_layer(fc1, 384, 192)
	fc2 = dropout_layer(fc2, keep_prob)

	out = fc_layer(fc2, 192, 10, act='softmax', std=0.005)
	return out

def tensor_net_big(x):
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

def conv4_net(x):
	conv1 = conv_layer(x, [3, 3, 3, 64])
	conv2 = conv_layer(conv1, [3, 3, 64, 64])
	pool1 = pool_layer(conv2)

	conv3 = conv_layer(pool1, [3, 3, 64, 128])
	conv4 = conv_layer(conv3, [3, 3, 128, 128])
	pool2 = pool_layer(conv4)

	reshaped, reshaped_shape = change_to_fc(pool2)
	fc1 = fc_layer(reshaped, reshaped_shape, 1024)         #input_tensor, input_shape, output_shape, std, name, act
	out = fc_layer(fc1, 1024, 10, act='softmax')
	return out

def conv4_do_net(x):
	conv1 = conv_layer(x, [3, 3, 3, 64])
	drop1 = dropout_layer(conv1, 0.3)
	conv2 = conv_layer(drop1, [3, 3, 64, 64])
	pool1 = pool_layer(conv2)

	conv3 = conv_layer(pool1, [3, 3, 64, 128])
	drop2 = dropout_layer(conv3, 0.4)
	conv4 = conv_layer(drop2, [3, 3, 128, 128])
	pool2 = pool_layer(conv4)

	reshaped, reshaped_shape = change_to_fc(pool2)
	drop3 = dropout_layer(reshaped, 0.5)
	fc1 = fc_layer(drop3, reshaped_shape, 1024)         #input_tensor, input_shape, output_shape, std, name, act
	drop4 = dropout_layer(fc1, 0.5)
	out = fc_layer(drop4, 1024, 10, act='softmax')
	return out

def vgg_net(x):
	conv1 = conv_layer(x, [3, 3, 3, 64])
	drop1 = dropout_layer(conv1, 0.3)
	conv2 = conv_layer(drop1, [3, 3, 64, 64])
	pool1 = pool_layer(conv2)

	conv3 = conv_layer(pool1, [3, 3, 64, 128])
	drop2 = dropout_layer(conv3, 0.4)
	conv4 = conv_layer(drop2, [3, 3, 128, 128])
	pool2 = pool_layer(conv4)

	conv5 = conv_layer(pool2, [3, 3, 128, 256])
	drop3 = dropout_layer(conv5, 0.4)
	conv6 = conv_layer(drop3, [3, 3, 256, 256])
	drop4 = dropout_layer(conv6, 0.4)
	conv7 = conv_layer(drop4, [3, 3, 256, 256])
	pool3 = pool_layer(conv7)

	conv8 = conv_layer(pool3, [3, 3, 256, 512])
	drop5 = dropout_layer(conv8, 0.4)
	conv9 = conv_layer(drop5, [3, 3, 512, 512])
	drop6 = dropout_layer(conv9, 0.4)
	conv10 = conv_layer(drop6, [3, 3, 512, 512])
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

#pred = tensor_net(x)
pred = tensor_net_big(x)

print "Graph Loaded"

beta = 0.004 
gamma = 0.0
theta = 0.0

with tf.name_scope('loss'):
	with tf.name_scope('cross_entropy_loss'):
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
	with tf.name_scope('weight_decay'):
		trainable_vars = [
                                #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv_layer'),
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

saver = tf.train.Saver()

#run = 'tensor_net_big_temp'
run = 'tensor_net_big_vr_' + str(gamma) + '_v1_' + str(theta)  + '_l2_' + str(beta)

tf.initialize_all_variables().run()

saver.restore(sess, 'models/' + run)
print "Model Restored"

#Testing
#avg_val_acc = 0.0
num_batches = int(test_size*1.0/batch_size)

val_acc = []
val_loss = []
val_loss_l2 = []
val_loss_vr = []
val_loss_v1 = []

for batch_num in range(num_batches):
	batch = cifardataloader.next_batch(data_type='test')
	batch_x = batch['images']#.reshape([-1, 32*32*3])
	batch_y = batch['labels']

	#acc = sess.run(accuracy, feed_dict = {x : batch_x, y : batch_y, keep_prob : 1.0, phase_train : False})
	loss, loss_l2, loss_vr, loss_v1, acc = sess.run([cost, l2_loss, vr_loss, v1_loss, accuracy], feed_dict = {x : batch_x, y : batch_y, keep_prob : 1.0, phase_train : False})
	#avg_val_acc += acc
	val_acc.append(acc)
	val_loss.append(loss)
	val_loss_l2.append(loss_l2)
	val_loss_vr.append(loss_vr)
	val_loss_v1.append(loss_v1)
	
#avg_val_acc = avg_val_acc*100.0/num_batches
#print str(avg_val_acc) 

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
