from numpy import *
import time
from scipy.misc import imread
from scipy.misc import imsave
import tensorflow as tf
from PIL import Image
from caffe_classes import class_names
from alexnet import Model


def visualization(input_image):
    """
    :param layer_name:  the layer you need to visulization. e.g. alexnet.conv2_in,conv_2,fc7,fc8
    :param input_image: the image you want to visualisze ,only one image
    :return: visualization
    """

    model=Model(is_training=False)
    model.layer=model.fc8 # what layer you want to visualize 
    #process image
    input_image=input_image-mean(input_image)
    input_image[:, :, 0], input_image[:, :, 2] = input_image[:, :, 2], input_image[:, :, 0]
    image_list=[]
    image_list.append(input_image)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        target=sess.run(model.layer, feed_dict = {model.x:image_list})
    #prepare imgae that need to be optimized
    #print(target.shape)
    # for input_im_ind in range(target.shape[0]):
    #     inds = argsort(target)[input_im_ind, :]
    #     print("Image", input_im_ind)
    #     for i in range(5):
    #         print(class_names[inds[-1 - i]], target[input_im_ind, inds[-1 - i]])
    ################################################################################



    #train the opt_img
    model = Model(is_training=True,layer=None,target=target)
    #model.layer=model.prob
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(500):
            loss=sess.run([model.x,model.l2diff,model.regularizer,model.loss,model.opt])
            #print(loss[1],loss[2],loss[3])
            if i%10==0:
                image=loss[0][0]
                #image[:, :, 0], image[:, :, 2] = image[:, :, 2], image[:, :, 0]
                #image=image+mean(input_image)
                imsave("generated/"+str(i)+".png",image)




def main():

    im1 = (imread("image/dog.png")[:,:,:3]) .astype(float32)

    visualization(im1)


main()