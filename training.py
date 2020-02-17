from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop, SGD, Adam
import keras.backend as K

import numpy as np
import random

import argparse
import os
from os import path
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy import misc

from utils import Settings
from utils import ElapsedTimer
from depthmapnet import DenseMapNet
from data_reader import Reader

os.environ["CUDA_VISIBLE_DEVICES"]="2"

class Predictor(object):
    def __init__(self, settings=Settings(),reader=Reader()):
        self.settings = settings
        self.reader = reader
        self.train_data_loaded = False
        if self.settings.epe:
            self.best_epe = self.settings.epe
        else:
            self.best_epe = 100.0

        print("Dataset Path:",self.settings.dataset)

    def train_network(self, Iter=0):
        self.load_train_data()
        self.train_all(Iter=Iter)

    def load_train_data(self):
        Nimgs=self.reader.pick_up_epochs(powerDepth=self.settings.powerDepth)
        self.channels = self.settings.channels = self.reader.all_left_img.shape[3]
        self.xdim = self.settings.xdim = self.reader.all_left_img.shape[2]
        self.ydim = self.settings.ydim = self.reader.all_left_img.shape[1]
        self.train_data_loaded = True
        if not hasattr(self, 'network'):
            self.network = DenseMapNet(settings=self.settings)
            self.model = self.network.build_model()
            self.model.load_weights(self.settings.model_weights)
        return Nimgs

    def train_all(self, epochs=1, lr=1e-3, Iter=1):
        if path.exists(self.settings.model_weights):
            pwd=os.path.abspath(os.getcwd())
            pathw =os.path.join(pwd,"weights_Pow"+str(self.settings.powerDepth))
            if not path.exists(pathw):
                os.mkdir(pathw)

            checkpoint = ModelCheckpoint(filepath= os.path.join(pathw,'dnn.weights.'+str(Iter).zfill(4) +'.h5'),# do you want to save the new weights in a new file or overwrite the the previous file
                                         save_weights_only=True,
                                         verbose=0,
                                         save_best_only=False)
            predict_callback = LambdaCallback(on_epoch_end=lambda epoch,
                                              logs: self.predict_disparity())#forward
            callbacks = [checkpoint, predict_callback]

            x = [self.reader.all_left_img, self.reader.all_right_img]

            print("weights_Pow"+str(self.settings.powerDepth))
            self.model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr))
            #self.model.compile(loss=self.weight_squared_error(), optimizer=Adam(lr=lr))
            #print("Using loss=crossent on sigmoid output layer")
            #self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr, decay=1e-6))
            
            self.model.fit(x,
                           self.reader.all_depth_img,
                           epochs=1,
                           batch_size=4,
                           shuffle=True,
                           callbacks=callbacks)#backward
        else:
            print("Please specify the initial wieght path!!")



    def predict_disparity(self):
        self.get_epe(use_train_data=False, get_performance=True)


    def get_epe(self, use_train_data=True, get_performance=False):
        lx = self.reader.all_left_img
        rx = self.reader.all_right_img
        dx = self.reader.all_depth_img
        print("Using train data... Size: ", lx.shape[0])

        # sum of all errors (normalized)
        epe_total = 0
        # count of images
        t = 0
        nsamples = lx.shape[0]
        elapsed_total = 0.0
        for i in range(0, nsamples, 1):
            indexes = np.arange(i, i + 1)
            left_images = lx[indexes, :, :, : ]
            right_images = rx[indexes, :, :, : ]
            disparity_images = dx[indexes, :, :, : ]
            # measure the speed of prediction on the 10th sample to avoid variance
            if get_performance:
                start_time = time.time()
                predicted_disparity = self.model.predict([left_images, right_images])
                elapsed_total += (time.time() - start_time)
            else:
                predicted_disparity = self.model.predict([left_images, right_images])

            predicted = predicted_disparity[0, :, :, :]
            ground = disparity_images[0, :, :, :]
            dim = predicted.shape[0] * predicted.shape[1]
            epe = predicted - ground
            # normalized error on all pixels
            epe = np.sum(np.absolute(epe))
            # epe = epe.astype('float32')
            epe = epe / dim
            epe_total += epe

        epe = epe_total / nsamples
        print("EPE: %0.2fpix" % epe)
        if epe < self.best_epe:
            self.best_epe = epe
            print("------------------- BEST EPE : %f ---------------------" % epe)
            tmpdir = "tmp"
            try:
                os.mkdir(tmpdir)
            except FileExistsError:
                print("Folder exists: ", tmpdir)
            filename = open('tmp/epe.txt', 'a')
            datetime = time.strftime("%H:%M:%S")
            filename.write("%s : %s EPE: %f\n" % (datetime, self.settings.dataset, epe))
            filename.close()
        # speed in sec
        if get_performance:
            print("Speed: %0.4fsec" % (elapsed_total / nsamples))
            print("Speed: %0.4fHz" % (nsamples / elapsed_total))

    def weight_squared_error(self):
        def loss(y_true,y_pred):
            y_true_f = K.flatten(y_true)
            y_pred_f = K.flatten(y_pred)
            weightdot = y_true_f * K.square(y_pred_f - y_true_f)
            return K.mean(weightdot)
        return loss

    
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
                        help="The Number of images in the dataset. If you are applying more than one datasets, this value is the min(number of images in dataset_0,number of images in dataset_1,number of images in dataset_2, . . .) ")
    parser.add_argument("-dn",
                        "--datanum",
                        type=int,
                        default=1,
                        help="The Number of different database")
    parser.add_argument("-pD",
                        "--powerDepth",
                        type=float,
                        default=1.,
                        help="depth^pD for enhance the depth value at the nearer object")
    help_ = "No training. Just prediction based on test data. Must load weights."
    parser.add_argument("-p",
                        "--predict",
                        action='store_true',
                        help=help_)
    help_ = "Best EPE"
    parser.add_argument("-e",
                        "--epe",
                        type=float,
                        help=help_)
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
    settings.predict = args.predict
    settings.epe = args.epe
    settings.nopadding = args.nopadding
    settings.filters=args.filter
    settings.kl = args.kernelsize
    settings.s = args.strides
    settings.batch_norm = args.batchnorm
    settings.depth = args.depth
    settings.powerDepth = args.powerDepth
    ldir = []
    rdir = []
    ddir = []
    for d in range(args.datanum):
        pathd = os.path.join(settings.dataset,str(d))
        ldir=np.append(ldir,os.path.join(pathd,"left"))
        rdir=np.append(rdir,os.path.join(pathd,"right"))
        ddir=np.append(ddir,os.path.join(pathd,"depth"))
    print("The number of images: %i" % args.imgnum)
    reader = Reader(ldir=ldir,rdir=rdir,ddir=ddir)
    predictor = Predictor(settings=settings,reader=reader)
    #epock has 1000 images
    maxEpochs = 500
    numPerIter = 100
    Iter = 0
    for epoch in range(0,maxEpochs):
        img_n = range(0, args.imgnum)
        while len(img_n)>0:
            print(len(img_n))
            if (len(img_n)>numPerIter):
                list = random.sample(img_n,numPerIter) #pickup numPerIter random images from dataset
                print("Initialized input data")
                reader.re_inti(list=list)
                print("TRAIN_NETWORK" + str(Iter))
                predictor.train_network(Iter)
                for i in list:
                    img_n = [j for j in img_n if j != i]
            else:
                list = random.sample(img_n,len(img_n))
                print("Initialized input data")
                reader.re_inti(list=list)
                print("TRAIN_NETWORK")
                predictor.train_network(Iter)
                for i in list:
                    img_n = [j for j in img_n if j != i]
            Iter += 1



