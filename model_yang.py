import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib import layers
import random


def siamese_net(input, reuse=False, is_training=True):  # without batch norm
    with tf.name_scope("model"):
        with tf.variable_scope("conv1_1") as scope:
            # ,normalizer_fn=slim.batch_norm,normalizer_params={'is_training':is_training}
            # tf.truncated_normal_initializer(stddev=0.001)
            conv1_1 = layers.conv2d(input,48,[3,3],activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv1_2") as scope:
            conv1_2 = layers.conv2d(conv1_1,48,[3,3],activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv2_1") as scope:
            conv2_1 = layers.avg_pool2d(conv1_2,[2,2])
        with tf.variable_scope("conv2_2") as scope:
            conv2_2 = layers.conv2d(conv2_1,96,[3,3],activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv2_3") as scope:
            conv2_3 = layers.conv2d(conv2_2, 96, [3, 3], activation_fn=tf.nn.relu, weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv3_1") as scope:
            conv3_1 = layers.avg_pool2d(conv2_3,[2,2])
        with tf.variable_scope("conv3_2") as scope:
            conv3_2 = layers.conv2d(conv3_1,192,[3,3],activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv3_3") as scope:
            conv3_3 = layers.conv2d(conv3_2, 192, [3, 3], activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv4_1") as scope:
            conv4_1 = layers.avg_pool2d(conv3_3,[2,2])
        with tf.variable_scope("conv4_2") as scope:
            conv4_2 = layers.conv2d(conv4_1,384,[3,3],activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv4_3") as scope:
            conv4_3 = layers.conv2d(conv4_2, 384, [3, 3], activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)

        with tf.variable_scope("conv43") as scope:
            conv43 = layers.conv2d_transpose(conv4_3,192,[2,2],stride=2,activation_fn=None,weights_initializer=
                                             layers.xavier_initializer_conv2d(),weights_regularizer=
                                             slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv3_4") as scope:
            conv3_4 = tf.concat([conv43, conv3_3],3)
        with tf.variable_scope("conv3_5") as scope:
            conv3_5 = layers.conv2d(conv3_4, 192, [3, 3], activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv3_6") as scope:
            conv3_6 = layers.conv2d(conv3_5, 192, [3, 3], activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv32") as scope:
            conv32 = layers.conv2d_transpose(conv3_6, 64, [2, 2], stride=2, activation_fn=None, weights_initializer=
                                             layers.xavier_initializer_conv2d(), weights_regularizer=
                                             slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv2_4") as scope:
            conv2_4 = tf.concat([conv32, conv2_3],3)
        with tf.variable_scope("conv2_5") as scope:
            conv2_5 = layers.conv2d(conv2_4, 96, [3, 3], activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv2_6") as scope:
            conv2_6 = layers.conv2d(conv2_5, 96, [3, 3], activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv21") as scope:
            conv21 = layers.conv2d_transpose(conv2_6,48, [2, 2], stride=2, activation_fn=None, weights_initializer=
                                             layers.xavier_initializer_conv2d(), weights_regularizer=
                                             slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv1_3") as scope:
            conv1_3 = tf.concat([conv21, conv1_2],3)
        with tf.variable_scope("conv1_4") as scope:
            conv1_4 = layers.conv2d(conv1_3, 48, [3, 3], activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv1_5") as scope:
            conv1_5 = layers.conv2d(conv1_4, 24, [3, 3], activation_fn=tf.nn.relu,weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv1_6") as scope:
            conv1_6 = layers.conv2d(conv1_5, 1, [3, 3], activation_fn=None, weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
    return conv1_6




def siamese_net_bn(input, reuse=False, is_training = True):  # with Batch Normalization
    with tf.name_scope("model"):
        with tf.variable_scope("conv1_1") as scope:
            # ,normalizer_fn=slim.batch_norm,normalizer_params={'is_training':is_training}
            # tf.truncated_normal_initializer(stddev=0.001)
            conv1_1 = layers.conv2d(input,48,[3,3],activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv1_2") as scope:
            conv1_2 = layers.conv2d(conv1_1,48,[3,3],activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv2_1") as scope:
            conv2_1 = layers.avg_pool2d(conv1_2,[2,2])
        with tf.variable_scope("conv2_2") as scope:
            conv2_2 = layers.conv2d(conv2_1,96,[3,3],activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv2_3") as scope:
            conv2_3 = layers.conv2d(conv2_2, 96, [3, 3], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv3_1") as scope:
            conv3_1 = layers.avg_pool2d(conv2_3,[2,2])
        with tf.variable_scope("conv3_2") as scope:
            conv3_2 = layers.conv2d(conv3_1,192,[3,3],activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv3_3") as scope:
            conv3_3 = layers.conv2d(conv3_2, 192, [3, 3], activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv4_1") as scope:
            conv4_1 = layers.avg_pool2d(conv3_3,[2,2])
        with tf.variable_scope("conv4_2") as scope:
            conv4_2 = layers.conv2d(conv4_1,384,[3,3],activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(),weights_regularizer=
                                    slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv4_3") as scope:
            conv4_3 = layers.conv2d(conv4_2, 384, [3, 3], activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv43") as scope:
            conv43 = layers.conv2d_transpose(conv4_3,192,[2,2],stride=2,activation_fn=None,weights_initializer=
                                             layers.xavier_initializer_conv2d(),weights_regularizer=
                                             slim.l2_regularizer(0.0001),scope=scope,reuse=reuse)
        with tf.variable_scope("conv3_4") as scope:
            conv3_4 = tf.concat([conv43, conv3_3],3)
            # conv3_4 = attention_module(conv43,conv3_3)
        with tf.variable_scope("conv3_5") as scope:
            conv3_5 = layers.conv2d(conv3_4, 192, [3, 3], activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv3_6") as scope:
            conv3_6 = layers.conv2d(conv3_5, 192, [3, 3], activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv32") as scope:
            conv32 = layers.conv2d_transpose(conv3_6, 96, [2, 2], stride=2, activation_fn=None, weights_initializer=
                                             layers.xavier_initializer_conv2d(), weights_regularizer=
                                             slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv2_4") as scope:
            conv2_4 = tf.concat([conv32, conv2_3],3)
            # conv2_4=attention_module(conv32,conv2_3)
        with tf.variable_scope("conv2_5") as scope:
            conv2_5 = layers.conv2d(conv2_4, 96, [3, 3], activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv2_6") as scope:
            conv2_6 = layers.conv2d(conv2_5, 96, [3, 3], activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv21") as scope:
            conv21 = layers.conv2d_transpose(conv2_6,48, [2, 2], stride=2, activation_fn=None, weights_initializer=
                                             layers.xavier_initializer_conv2d(), weights_regularizer=
                                             slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv1_3") as scope:
            conv1_3 = tf.concat([conv21, conv1_2],3)
            # conv1_3 = attention_module(conv21,conv1_2)
        with tf.variable_scope("conv1_4") as scope:
            conv1_4 = layers.conv2d(conv1_3, 48, [3, 3], activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv1_5") as scope:
            conv1_5 = layers.conv2d(conv1_4, 24, [3, 3], activation_fn=tf.nn.relu,normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': is_training},weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
        with tf.variable_scope("conv1_6") as scope:
            conv1_6 = layers.conv2d(conv1_5, 1, [3, 3], padding='same',activation_fn=None, weights_initializer=
                                    layers.xavier_initializer_conv2d(), weights_regularizer=
                                    slim.l2_regularizer(0.0001), scope=scope, reuse=reuse)
    return conv1_6





def attention_module(upconv, skipconv):
    """
    Implementation of Attention Module (2 pool in soft mask branch)
    Input:
    --- upconv: Module input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- skipconv: Module input, 4-D Tensor, with shape [bsize, height, width, channel]
    --- name: Module name
    Output:
    --- outputs: Module output
    """
    # Sigmoid
    masks_4 = tf.nn.sigmoid(upconv, "mask_sigmoid")
    # Fusing
    with tf.name_scope("fusing"), tf.variable_scope("fusing"):
        outputs = tf.multiply(skipconv, masks_4, name="fuse_mul")
        outputs = tf.add(upconv, outputs, name="fuse_add")
        return outputs

def addlayer(input,outfeature,is_training):
    comp_out = conv_layer(input,int(input.get_shape()[-1]),outfeature,3,stride=1)
    comp_out = tf.layers.batch_normalization(comp_out,training=is_training)
    comp_out = relu(comp_out)
    output=tf.concat([input,comp_out],3)
    return output

