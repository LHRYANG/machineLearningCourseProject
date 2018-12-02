from numpy import *
import time
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf
from PIL import Image
from caffe_classes import class_names
from alexnet import Model
import numpy as np

import copy
def saliency(img,patch_size=3):
    """
    :param target_class: The class u want get
    :return:
    """
    # process the image
    im1 = img - mean(img)
    im1=np.flip(im1,2)
    img_list=[]
    img_list.append(im1)

    model = Model(is_training=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(model.prob, feed_dict={model.x: img_list})
    target=0
    for input_im_ind in range(output.shape[0]):
        inds = argsort(output)[input_im_ind, :]
        print("top-class", output[input_im_ind, inds[-1]])
        target=output[input_im_ind, inds[-1]]

    model = Model(is_training=False)
    # model.layer=model.prob

    saliency_array=[]
    step=0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # 227/3=75
        for x in range(int(227/patch_size)):
            for y in range(int(227/patch_size)):
                temp_img=copy.deepcopy(im1)
                temp_img[patch_size*x:patch_size*(x+1),patch_size*y:patch_size*(y+1),:]=0
                aa=[]
                aa.append(temp_img)
                prob=sess.run(model.prob,{model.x: aa})
                for input_im_ind in range(prob.shape[0]):
                    inds = argsort(prob)[input_im_ind, :]

                    saliency_array.append(target-prob[input_im_ind, inds[-1]])

                step+=1
                if(step%500==0):
                    print(step)
                    #print(saliency_array[-50:])
    saliency_map=np.zeros((227,227))

    step=0
    for x in range(int(227/patch_size)):
        for y in range(int(227/patch_size)):
            if(saliency_array[step]<0):
                saliency_array[step]=0

            saliency_map[patch_size * x: patch_size * (x + 1), patch_size * y: patch_size * (y + 1)]=saliency_array[step]*255
            step+=1
    print(len(saliency_array))

    imsave("./generated/saliency.png",saliency_map)
if __name__=='__main__':

    im1 = (imread("image/poodle.png")[:, :, :3]).astype(float32)
    patch_size=3 # how large area do you want to remove at each step
    saliency(im1,3)
