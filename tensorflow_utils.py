import tensorflow as tf


def weight_variable(shape,var_name):
	temp_var = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(temp_var,name=var_name)

def bias_variable(shape,var_name):
	temp_var = tf.constant(0.1,shape=shape)
	return tf.Variable(temp_var,name=var_name)

def full_layer(input_vector,size,prefix_name,activation="relu"):
	in_size = int(input_vector.get_shape()[1])
	W = weight_variable([in_size,size],prefix_name + "_dense_W")
	b = bias_variable([size],prefix_name+"_dense_b")
	if activation == "relu":
		return tf.nn.relu(tf.matmul(input_vector,W) + b)
	elif activation == "sigmoid":
		return tf.nn.sigmoid(tf.matmul(input_vector,W) + b)
	elif activation == "None":
		return tf.matmul(input_vector,W)+b