from utils.data import CifarDataLoader
batch_size = 128
cifardataloader = CifarDataLoader(batch_size=batch_size)
train_size = cifardataloader.train_size
test_size = cifardataloader.test_size

print "Data Loaded"


import subprocess, time

total_memory = 4742
max_occupied = 3000

gpu_output = subprocess.check_output(["nvidia-smi"])

memory_occupied = int(gpu_output.split('MiB /  ' + str(total_memory) + 'MiB')[0].split('|')[-1])

print memory_occupied
while memory_occupied > max_occupied:
    gpu_output = subprocess.check_output(["nvidia-smi"])
    memory_occupied = int(gpu_output.split('MiB /  ' + str(total_memory) + 'MiB')[0].split('|')[-1])
    time.sleep(0.01)
print "GPU available"


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from tqdm import tqdm

import numpy as np
import tensorflow as tf

from utils.layers import fc_layer, conv_layer, pool_layer, dropout_layer, change_to_fc, batch_norm

print "Import complete"

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
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
gamma = 0.001
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

learning_rate = tf.placeholder(tf.float32)
lr_summ = tf.scalar_summary('lr', learning_rate)

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

restore_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='fc_layer') + \
					tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='conv_layer')
saver = tf.train.Saver(restore_var_list)

#run = 'tensor_net_big_temp'
run = 'tensor_net_big_vr_' + str(gamma) + '_v1_' + str(theta)  + '_l2_' + str(beta)
merged = tf.merge_all_summaries()

tf.initialize_all_variables().run()

train_writer = tf.train.SummaryWriter('logs/' + run + '/train/', sess.graph)
test_writer = tf.train.SummaryWriter('logs/' + run + '/test/')

step = 0
present_learning_rate = 0.01
#present_learning_rate = 0.0002
saver.restore(sess, 'models/' + 'tensor_net_big_vr_0.0_v1_0.0_l2_0.004')
#saver.restore(sess, 'models/' + run)
print "Model Restored"
best_acc = 75.00

for epoch in range(1000):
	cifardataloader.reset_index()
	avg_loss = 0.0
	avg_acc = 0.0
	num_batches = int(train_size*1.0/batch_size)

	if((epoch >= 100) and (epoch%500 == 0)):
		present_learning_rate /= 1.3

	f = open('logs/log.txt', 'a')

	for batch_num in range(num_batches):
		step += 1.0

		batch = cifardataloader.next_batch()
		batch_x = batch['images']
		batch_y = batch['labels']

		#0.6
		loss, acc, _, summary = sess.run([cost, accuracy, train_step, merged], feed_dict = {x : batch_x, y : batch_y, learning_rate : present_learning_rate, keep_prob : 0.9, phase_train : True})
		
		avg_loss += loss
		avg_acc += acc
		train_writer.add_summary(summary, step)

	avg_acc = avg_acc*100.0/num_batches
	avg_loss = avg_loss*1.0/num_batches
	f.write("Train Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_acc) + ' Loss: ' + str(avg_loss) + '\n')
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
		f.write("Val   Epoch: " + str(epoch + 1) + ' Acc: ' + str(avg_val_acc) + ' Loss: ' + str(avg_val_loss) + '\n')
		print " "*60 + "Val   Epoch: " + str(epoch + 1) + ' Acc: ' + str(round(avg_val_acc, 6)) + ' Loss: ' + str(round(avg_val_loss, 6))

		if avg_val_acc > best_acc:
			save_path = saver.save(sess, 'models/'+run)
			best_acc = avg_val_acc

	f.close()


train_writer.close()
test_writer.close()


"""
	TensorNet:
		LR: 0.001
			Train Epoch: 50 Acc: 67.223557692300 Loss: 5.29408353292
			Val   Epoch: 50 Acc: 56.361607142900 Loss: 5.39819186074

			Train Epoch: 70 Acc: 71.268028846200 Loss: 5.04468954771
	        Val   Epoch: 70 Acc: 60.602678571400 Loss: 5.13090276718

	        Train Epoch: 100 Acc: 75.410657051300 Loss: 4.72204439823
            Val   Epoch: 100 Acc: 61.607142857100 Loss: 4.83851664407

            Train Epoch: 130 Acc: 78.411458333300 Loss: 4.42451097782
            Val   Epoch: 130 Acc: 64.843750000000 Loss: 4.55479219982

            Train Epoch: 148 Acc: 80.114182692300 Loss: 4.25868216539
            Val   Epoch: 148 Acc: 63.504464285700 Loss: 4.40832144873

            Train Epoch: 200 Acc: 82.592147435900 Loss: 3.84687323998
        	Val   Epoch: 199 Acc: 64.508928571400 Loss: 4.02165944236

			Train Epoch: 230 Acc: 83.471554487200 Loss: 3.63957836383
			Val   Epoch: 229 Acc: 65.066964285700 Loss: 3.81760099956

			Train Epoch: 250 Acc: 83.826121794900 Loss: 3.51535388812
            Val   Epoch: 250 Acc: 65.959821428600 Loss: 3.68703535625

			Train Epoch: 271 Acc: 84.208733974400 Loss: 3.39186433951
            Val   Epoch: 271 Acc: 65.513392857100 Loss: 3.56734037399

            Train Epoch: 301 Acc: 84.423076923100 Loss: 3.23266950326
            Val   Epoch: 301 Acc: 66.183035714300 Loss: 3.40574717522

            Train Epoch: 307 Acc: 84.719551282100 Loss: 3.19965119056
            Val   Epoch: 307 Acc: 67.075892857100 Loss: 3.37270883151

            Train Epoch: 410 Acc: 86.035657051300 Loss: 2.75388657191
            Val   Epoch: 409 Acc: 67.968750000000 Loss: 2.93123565401

            Train Epoch: 451 Acc: 86.199919871800 Loss: 2.61647330125
            Val   Epoch: 451 Acc: 68.191964285700 Loss: 2.78996869496

            Train Epoch: 500 Acc: 87.105368589700 Loss: 2.46421079574
            Val   Epoch: 499 Acc: 68.973214285700 Loss: 2.64350642477

            Train Epoch: 500 Acc: 87.105368589700 Loss: 2.46421079574
            Val   Epoch: 550 Acc: 68.415178571400 Loss: 2.5235819476

            Train Epoch: 601 Acc: 87.938701923100 Loss: 2.22343301712
        	Val   Epoch: 601 Acc: 68.750000000000 Loss: 2.4124075004

        	Train Epoch: 650 Acc: 87.465945512800 Loss: 2.14306060045
        	Val   Epoch: 649 Acc: 65.959821428600 Loss: 2.35265997478
			
        	Train Epoch: 700 Acc: 89.158653846200 Loss: 2.04395255401
        	Val   Epoch: 700 Acc: 69.531250000000 Loss: 2.23739852224

        	Train Epoch: 749 Acc: 88.944310897400 Loss: 1.98213718763
        	Val   Epoch: 748 Acc: 69.866071428600 Loss: 2.17182247979

        	Train Epoch: 800 Acc: 90.058092948700 Loss: 1.91128149338
        	Val   Epoch: 799 Acc: 69.977678571400 Loss: 2.10631540843

        	Train Epoch: 850 Acc: 90.372596153800 Loss: 1.86018557946
            Val   Epoch: 850 Acc: 71.316964285700 Loss: 2.05173855168

            Train Epoch: 899 Acc: 90.919471153800 Loss: 1.81331128371
            Val   Epoch: 898 Acc: 72.767857142900 Loss: 2.00531516756

            Train Epoch: 950 Acc: 91.386217948700 Loss: 1.77313805146
            Val   Epoch: 949 Acc: 73.102678571400 Loss: 1.95493570396

            Train Epoch: 1000 Acc: 91.848958333300 Loss: 1.73676030514
            Val   Epoch: 1000 Acc: 73.549107142900 Loss: 1.92334336894

            Train Epoch: 50 Acc: 92.197516025600 Loss: 1.70781585253
            Val   Epoch: 49 Acc: 71.428571428600 Loss: 1.9129785129

            Train Epoch: 100 Acc: 92.922676282100 Loss: 1.65142323359
            Val   Epoch: 100 Acc: 73.437500000000 Loss: 1.84906366893

            Train Epoch: 151 Acc: 93.241185897400 Loss: 1.63233264685
            Val   Epoch: 151 Acc: 74.330357142900 Loss: 1.82643490178

            Train Epoch: 200 Acc: 91.318108974400 Loss: 1.6409849931
            Val   Epoch: 199 Acc: 73.214285714300 Loss: 1.817046472

            Train Epoch: 250 Acc: 93.713942307700 Loss: 1.60257020761
            Val   Epoch: 250 Acc: 74.107142857100 Loss: 1.79920085839

            Train Epoch: 301 Acc: 93.603766025600 Loss: 1.59524484048
            Val   Epoch: 301 Acc: 73.214285714300 Loss: 1.79661079815

            Train Epoch: 350 Acc: 94.162660256400 Loss: 1.57988934181
            Val   Epoch: 349 Acc: 73.214285714300 Loss: 1.78531025137



    TensorNet-Do:
    	LR: 0.001
    		Train Epoch: 4 Acc: 62.638221153800 Loss: 1.90835942274
            Val   Epoch: 4 Acc: 61.049107142900 Loss: 1.93236982822

            Train Epoch: 40 Acc: 77.319711538500 Loss: 1.7634648589
            Val   Epoch: 40 Acc: 67.968750000000 Loss: 1.85635471344

    TensorNet-Do(keep_prob:0.6):
    	LR: 0.001
	        Train Epoch: 37 Acc: 87.974759615400 Loss: 1.65785433207
	        Val   Epoch: 37 Acc: 73.995535714300 Loss: 1.79015743732

	TensorNet-Do(keep_prob:0.4):
    	LR: 0.001
	        Train Epoch: 64 Acc: 86.772836538500 Loss: 1.6666231837
            Val   Epoch: 64 Acc: 74.107142857100 Loss: 1.78527544226

            Train Epoch: 100 Acc: 87.237580128200 Loss: 1.65784983176
            Val   Epoch: 100 Acc: 75.111607142900 Loss: 1.77397845473

            Train Epoch: 151 Acc: 89.126602564100 Loss: 1.63725438821
            Val   Epoch: 151 Acc: 74.553571428600 Loss: 1.77334306921

            Train Epoch: 199 Acc: 90.202323717900 Loss: 1.62385286368
            Val   Epoch: 199 Acc: 75.334821428600 Loss: 1.7673305784


    TensorNet-Do(keep_prob_all:0.4):
    	LR: 0.001
    		Train Epoch: 4 Acc: 76.264022435900 Loss: 1.75627836845
            Val   Epoch: 4 Acc: 74.441964285700 Loss: 1.77686892237

            Train Epoch: 64 Acc: 82.742387820500 Loss: 1.69298068468
            Val   Epoch: 64 Acc: 75.781250000000 Loss: 1.77063551971

            Train Epoch: 601 Acc: 85.757211538500 Loss: 1.6581106837
            Val   Epoch: 601 Acc: 78.571428571400 Loss: 1.74184863908

            Train Epoch: 1501 Acc: 89.122596153800 Loss: 1.62225629214
            Val   Epoch: 1501 Acc: 79.575892857100 Loss: 1.73654455798




            Train Epoch: 22 Acc: 82.491987179500 Loss: 1.73395571495
            Val   Epoch: 22 Acc: 78.683035714300 Loss: 1.77622209276

            Train Epoch: 85 Acc: 84.959935897400 Loss: 1.69916025278
            Val   Epoch: 85 Acc: 80.022321428600 Loss: 1.7527108533

            Train Epoch: 500 Acc: 85.709134615400 Loss: 1.66936492461
			Val   Epoch: 499 Acc: 79.575892857100 Loss: 1.73396444321


    TensorNet-Do(keep_prob_all:0.4, vr:0.01):
    	LR: 0.001


"""


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

"""
Train Epoch: 121 Acc: 96.804888 Loss: 2.30667
Val   Epoch: 121 Acc: 80.388622 Loss: 2.468243
"""
