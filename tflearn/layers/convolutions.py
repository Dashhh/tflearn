import tensorflow as tf
import numpy as np

def alpha(weights):
    w_shape = tf.shape(weights)
    n = tf.cast(tf.reduce_prod(w_shape[:-1]),tf.float32)
    w_abs = tf.abs(weights)
    a = tf.reduce_sum(w_abs, [0,1,2]) / n
    return tf.stop_gradient(a)

def gamma(inp, weights, strides, padding):
    inp_shape = inp.get_shape().as_list()
    w_shape = weights.get_shape().as_list()

    A = tf.reduce_sum(tf.abs(inp), [3]) / inp_shape[3]
    A = tf.reshape(A, [-1] + inp_shape[1:3] + [1])

    k = tf.ones(w_shape[:2] + [1,1], dtype=tf.float32) / (w_shape[0]*w_shape[1])
    
    K = conv2d(A, k, strides, padding)
    a = alpha(weights)
    return tf.stop_gradient(K*a)

def conv2d(inp, weights, strides=[1,1,1,1], padding='SAME'):
    return tf.nn.conv2d(inp, weights, strides=strides, padding=padding)

def conv_bin_activ(inp, weights, strides=[1,1,1,1], padding='SAME'):
    g = gamma(inp, weights, strides, padding)
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
        x_sign = tf.sign(inp)
        w_sign = tf.sign(weights) 
    return conv2d(x_sign, w_sign, strides, padding) * g

def conv_bin_weights(inp, weights, strides=[1,1,1,1], padding='SAME'):
    a = alpha(weights)
    G = tf.get_default_graph()
    with G.gradient_override_map({"Sign": "Identity"}):
        w_sign = tf.sign(weights)
    return conv2d(inp, w_sign, strides, padding)*a

convolutions = {
    None: conv2d,
    'weights': conv_bin_weights,
    'full': conv_bin_activ,
}
