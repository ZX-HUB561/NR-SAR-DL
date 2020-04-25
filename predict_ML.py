import cv2 as cv
import os
import shutil
import time
import model_yang
import scipy
import tensorflow as tf
# from model_yang import res_dense_net as model_net
import numpy as np

def predict():
    start_time = time.time()
  
    outtif = 'result.tif'
    outmat= 'result.mat'
    image_size =1600

    model_path = './model_sim'
  
    sess = tf.Session()
    start_time = time.time()
    hrx1 = tf.placeholder(tf.float32, [1, image_size, image_size, 1])

 
    pred_one=model_yang.siamese_net_bn(hrx1,is_training=False)
 
    checkpoint_dir = model_path
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver = tf.train.Saver()
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
    mat_im = scipy.io.loadmat("time1_amplitude.mat")

    im_val = mat_im['time1_amplitude']

    ori_val = im_val[0:image_size,0:image_size]
 

    
    max_val=np.amax(ori_val)
    ori_val = (ori_val*1-0)/max_val

    ori_val = ori_val.reshape([1, image_size, image_size, 1])

    predict_result = sess.run([pred_one], feed_dict={hrx1: ori_val})
    predict_result = predict_result[0]
    predict_result = predict_result.squeeze()
    predict_result = predict_result * max_val

    cv.imwrite(outtif, predict_result)
    scipy.io.savemat(outmat, {'result': predict_result})
    print("time: [%4.4f]" % (time.time() - start_time))



if __name__ == '__main__':
    predict()