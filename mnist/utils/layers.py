import numpy as np
import tensorflow as tf
import math

def fc_layer(x, in_dim, out_dim, act='relu', bn=True):
    with tf.name_scope('fc_layer'):
        #stddev = math.sqrt(3.0 / (in_dim + out_dim))
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.01))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[out_dim]))
        with tf.name_scope('activation_wx_plus_b'):
            if bn == True:
                activations = tf.nn.relu(batch_norm(tf.matmul(x, weights) + biases)) if(act == 'relu') else tf.nn.softmax(tf.matmul(x, weights) + biases)
            else:
                activations = tf.nn.relu(tf.matmul(x, weights) + biases) if(act == 'relu') else tf.nn.softmax(tf.matmul(x, weights) + biases)
    return activations

def conv_layer(x, kernel_shape, out_dim):
    with tf.name_scope('conv_layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal(kernel_shape, stddev=0.01))
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[out_dim]))
        with tf.name_scope('activation_wx_plus_b'):
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

def batch_norm(X, eps=1e-8, g=None, b=None):
    with tf.name_scope('batch_norm'):  
        if X.get_shape().ndims == 4:
            mean = tf.reduce_mean(X, [0,1,2])
            std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
            X = (X-mean) / tf.sqrt(std+eps)

            if g is not None and b is not None:
                g = tf.reshape(g, [1,1,1,-1])
                b = tf.reshape(b, [1,1,1,-1])
                X = X*g + b

        elif X.get_shape().ndims == 2:
            mean = tf.reduce_mean(X, 0)
            std = tf.reduce_mean(tf.square(X-mean), 0)
            X = (X-mean) / tf.sqrt(std+eps)

            if g is not None and b is not None:
                g = tf.reshape(g, [1,-1])
                b = tf.reshape(b, [1,-1])
                X = X*g + b

        else:
            raise NotImplementedError

        return X

def batch_norm2(x, phase_train, n_out):
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