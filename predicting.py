from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop, SGD

import numpy as np

import argparse
import os
from os import path
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy import misc
from PIL import Image
import imageio
from utils import Settings
from utils import ElapsedTimer
from depthmapnet import DenseMapNet
from data_reader_predict import Reader
import time
import random

print("Start ................... ")

os.environ["CUDA_VISIBLE_DEVICES"]="2"


class Predictor(object):
    def __init__(self, settings=Settings(),reader=Reader()):
        self.settings = settings
        self.reader = reader
        self.data_loaded = False
        print("Dataset Path:",self.settings.dataset)
    def predict_network(self,iter_num):
        self.load_data()
        self.predict_disparity(iter_num)

    def load_data(self):
        Nimgs=self.reader.pick_up_epochs()
        self.channels = self.settings.channels = self.reader.all_left_img.shape[3]
        self.xdim = self.settings.xdim = self.reader.all_left_img.shape[2]
        self.ydim = self.settings.ydim = self.reader.all_left_img.shape[1]
        self.data_loaded = True
        if not hasattr(self, 'network'):
            self.result_path = self.settings.dataset+"/"+time.asctime( time.localtime(time.time()) )
            try:
                os.mkdir(self.result_path)
            except OSError:
                print ("Creation of the directory %s failed" % self.result_path)
            else:
                print ("Successfully created the directory %s " % self.result_path)
            self.network = DenseMapNet(settings=self.settings)
            self.model = self.network.build_model()
            self.model.load_weights(self.settings.model_weights)
        return Nimgs

    def predict_disparity(self,iter_num):
        if self.data_loaded:
            predicted_disparity = self.model.predict([self.reader.all_left_img, self.reader.all_right_img])

            for i in range(predicted_disparity[0].shape[0]):
                #self.save_images_depth(predicted_disparity[0][i,:,:,:],"depth",i+iter_num)
                self.save_images_depth(np.power(predicted_disparity[0][i,:,:,:],1/1.5),"depth",i+iter_num)

            #for i in range(predicted_disparity[1].shape[0]):
            #    self.save_images_left(predicted_disparity[1][i,:,:,:],"left_rec",i+iter_num)


    def save_images_depth(self,image,name, num):
        size = [image.shape[0], image.shape[1]]
        image =  1 - np.clip(image, 0.0, 1.0)
        #image *= 255
        #image = image.astype(np.uint8)
        #image = np.reshape(image, size)
        image *= 15000
        image = image.astype(np.uint16)
        image = np.reshape(image, size)
        imageio.imwrite(self.result_path+"/"+name+"_image_"+ str(num).zfill(4) + ".png", image)
        #pillow_version = getattr(Image, "__version__", "0")
        #if pillow_version.split(".")[0] >= 6:
        #image=Image.fromarray(image, "I;16")
        #else:
        #    image = Image.new("I", image.T.shape)
        #    array_buffer = image.tobytes()
        #    image.frombytes(array_buffer, 'raw', "I;16")
        #image.save(self.result_path+"/image_" + str(num).zfill(4) + ".png")
    def save_images_left(self,image,name, num):
        size = [image.shape[0], image.shape[1]]
        image *= 255
        image = image.astype(np.uint16)
        image = np.reshape(image, size)
        imageio.imwrite(self.result_path+"/"+name+"_image_"+ str(num).zfill(4) + ".png", image)


if __name__ == '__main__':
    print("---main---")
    parser = argparse.ArgumentParser()
    help_ = "Load checkpoint hdf5 file of model trained weights"
    parser.add_argument("-w",
                        "--weights",
                        required=True,
                        help=help_)
    parser.add_argument("-d",
                        "--dataset",
                        required=True,
                        help="The path of dataset to load")
    parser.add_argument("-i",
                        "--imgnum",
                        type=int,
                        required=True,
                        help="The Number of images in the dataset. If you are applying more than one datasets, this value is the max(number of images in dataset_0,number of images in dataset_1,number of images in dataset_2, . . .) ")
    parser.add_argument("-dn",
                        "--datanum",
                        type=int,
                        default=1,
                        help="The Number of different database")
    help_ = "No padding"
    parser.add_argument("-a",
                        "--nopadding",
                        action='store_true',
                        help=help_)
    parser.add_argument("-f",
                        "--filter",
                        type=int,
                        default=32,
                        help="The Filter value in DnCNN algorithm, layer Conv2D")
    parser.add_argument("-k",
                        "--kernelsize",
                        type=int,
                        default=3,
                        help="The kernal size value in DnCNN algorithm, layer Conv2D")
    parser.add_argument("-s",
                        "--strides",
                        type=int,
                        default=1,
                        help="The strides value in DnCNN algorithm, layer Conv2D")
    parser.add_argument("-bn",
                        "--batchnorm",
                        type=int,
                        default=1,
                        help="The batch norm value in DnCNN algorithm")
    parser.add_argument("-de",
                        "--depth",
                        type=int,
                        default=11,
                        help="The depth value in DnCNN algorithm")
    print("read input parameters ..... ")
    args = parser.parse_args()
    settings = Settings()
    settings.model_weights = args.weights
    settings.dataset = args.dataset
    settings.nopadding = args.nopadding
    settings.filters=args.filter
    settings.kl = args.kernelsize
    settings.s = args.strides
    settings.batch_norm = args.batchnorm
    settings.depth = args.depth
    ldir = []
    rdir = []
    for d in range(args.datanum,args.datanum+1):
        pathd = os.path.join(settings.dataset,str(d))
        ldir=np.append(ldir,os.path.join(pathd,"left"))
        rdir=np.append(rdir,os.path.join(pathd,"right"))
    print("The number of images: %i" % args.imgnum)
    reader = Reader(ldir=ldir,rdir=rdir)
    predictor = Predictor(settings=settings,reader=reader)
    numPerIter = 100
    Iter = range(0, args.imgnum,numPerIter)
    for i in Iter:
        list = range(i,i+numPerIter,1)
        print("Initialized input data")
        reader.re_inti(list=list)
        print("Predict images from " + str(i)+" to "+ str(i+numPerIter))
        predictor.predict_network(i)
