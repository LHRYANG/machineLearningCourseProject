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
def fool(img,target_class,minimum_confidence):
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

    for input_im_ind in range(output.shape[0]):
        inds = argsort(output)[input_im_ind, :]
        print("Image", input_im_ind)
        for i in range(5):
            print(inds[-1-i])
            print(class_names[inds[-1 - i]], output[input_im_ind, inds[-1 - i]])
    img_list=np.array(img_list)
    img_list = tf.convert_to_tensor(img_list)

    model2 = Model(is_training=False)
    sess2 = tf.Session()
    init = tf.global_variables_initializer()
    sess2.run(init)

    model = Model(is_training=True, layer=None, target=target_class,img_real =img_list)
    # model.layer=model.prob



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            img_generated= sess.run([model.x,model.opt])
            img_list=img_generated[0]

            prob = sess2.run(model2.prob, feed_dict={model2.x: img_list})


            print(str(i)+": "+str(prob[0][target_class]))
            if(prob[0][target_class]>minimum_confidence):
                break


            image = img_list[0]
            image=np.flip(image,2)
            image=image+mean(img)
            imsave("generated/" + str(i) + ".png", image)


if __name__=='__main__':
    target_class=354
    minimum_confidence=0.8 #想要目标值的概率达到多大
    im1 = (imread("image/dog.png")[:, :, :3]).astype(float32)
    #im2 =( imread("image/dog-0.8-0.005.png")[:, :, :3]).astype(float32)
    #print(np.sum(im1-im2))
    fool(im1,target_class,minimum_confidence)
