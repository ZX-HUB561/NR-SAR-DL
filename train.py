# from utils import *
import time
import random
import os
import argparse
import scipy
import numpy as np
from model import siamese_net
import tensorflow as tf


def rsmain(args):
    model_path = './model'
    logs_path = './logs'
    image_size = 80
    batch_size = 32
    epoch = 200
    epsilon = 1e-6
    x1 = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])
    x2 = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 1])
    si12 = tf.placeholder(tf.float32, [batch_size, image_size, image_size,1])
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    y1 = siamese_net(x1, is_training=True)
    # loss=tf.reduce_mean(tf.add(tf.multiply(y1,tf.exp(-x2)),x2))
    y2 = siamese_net(x2, reuse=True,is_training=True)
    loss1=tf.reduce_mean(tf.abs(y1-x2)*si12) #+0.05*tf.reduce_mean(x1-y1)  #+tf.log(tf.abs(y1)))  #tf.divide(x1,y1)+
    loss2 = tf.reduce_mean(tf.abs(y2-x1)*si12) #+0.05*tf.reduce_mean(x2-y2) #+ 0.05 * tf.reduce_mean(tf.divide(x2, y2) + tf.log(tf.abs(y2)))  #tf.divide(x2, y2) +
    loss=loss1+loss2
    #
    # losses = loss12  #+loss21  #regular_loss  #+loss21
    global_step = tf.Variable(0)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_opt = optimizer.minimize(loss, global_step=global_step)
    saver = tf.train.Saver(max_to_keep=5)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    counter = 0
    start_time = time.time()
    mat_im1 = scipy.io.loadmat('data_t1.mat')
    im1_val = mat_im1['time1_amplitude']
    mx1 = np.amax(im1_val)
    mat_im2 = scipy.io.loadmat('data_t2.mat')
    im2_val = mat_im2['time2_amplitude']
    mx2 = np.amax(im1_val)
    mxv = max(mx1,mx2)
    mat_ss = scipy.io.loadmat('data_t2.mat')
    ss12_val = mat_ss['time2_amplitude']
    im1_val_nor = im1_val/mxv
    im2_val_nor = im2_val/mxv
    train_batch = np.zeros((5000,image_size,image_size,1))
    target_batch = np.zeros((5000,image_size,image_size,1))
    sim12_batch = np.zeros((5000,image_size,image_size,1))
    i = 0
    while i < 5000:

        x = random.randrange(image_size,1600,25)
        y = random.randrange(image_size,1600,25)
        # train_matrix = im1_val_nor[x - image_size:x, y - image_size:y]
        train_matrix = np.expand_dims(im1_val_nor[x - image_size:x, y - image_size:y], axis=2)
        target_matrix = np.expand_dims(im2_val_nor[x - image_size:x, y - image_size:y], axis=2)
        # target_matrix =im2_val_nor[x - image_size:x, y - image_size:y]
        similar12_matrix = np.expand_dims(ss12_val[x - image_size:x, y - image_size:y], axis=2)

        train_batch[i] = train_matrix
        target_batch[i] = target_matrix
        sim12_batch[i] = similar12_matrix
             
        i += 1
        print(i)
    lrinput1 = train_batch
    lrinput2 = target_batch
    simi12 = sim12_batch
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        summary_writer = tf.summary.FileWriter(logs_path, sess.graph)
        epoch_learning_rate = 1e-4
        for one_epoch in range(1, epoch+1):
            if one_epoch == epoch * 0.2 or one_epoch == epoch * 0.4 or one_epoch == epoch * 0.6:
                epoch_learning_rate = epoch_learning_rate * 0.1
            train_loss = 0.0
            batch_idxs = len(lrinput1) // batch_size
            for idx in range(0, batch_idxs):
                batch_lrinput1 = lrinput1[idx*batch_size: (idx+1)*batch_size]
                batch_lrinput2 = lrinput2[idx*batch_size: (idx+1)*batch_size]
                batch_simi12 = simi12[idx*batch_size: (idx+1)*batch_size]
                counter = counter + 1
                train_feed_dict = {
                    x1: batch_lrinput1,
                    x2: batch_lrinput2,
                    si12: batch_simi12,
                    learning_rate: epoch_learning_rate
                 }
                _, error = sess.run([train_opt, loss], feed_dict=train_feed_dict)
                train_loss = train_loss + error
                if counter % 20 == 0:
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]"
                          % (one_epoch, counter, time.time() - start_time, error))
                if counter % 500 == 0:
                    train_loss = train_loss/500.0
                    train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss)])
                    summary_writer.add_summary(summary=train_summary, global_step=epoch)
                    summary_writer.flush()
                    save_path = os.path.join(model_path, 'LST_CNN_itr%d.ckpt'%counter)
                    saver.save(sess, save_path)
                    print('model parameter has been saved in %s'%save_path)





if __name__ == '__main__':
    parser = argparse.ArgumentParser(usage="it's usage tip.", description="help info.")
    parser.add_argument("--gpu", choices=['0', '1', '2', '3'], default='0', help="gpu_id")
    args = parser.parse_args()
    # print(count1())
    rsmain(args)









