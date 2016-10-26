import numpy as np
import tensorflow as tf

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
