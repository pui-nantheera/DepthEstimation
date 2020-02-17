import numpy as np
from PIL import Image
import os
from os import path

#Paths:
#ldir = left images directory
#rdir = right images directory
#ddir = depth images directory
class Reader(object):
    def __init__(self,ldir=[],rdir=[]):
        self.list = []
        self.ldir = ldir
        self.rdir = rdir
        self.all_left_img = []
        self.all_right_img = []

    def pick_up_epochs(self):
        n=0
        for d in range(len(self.ldir)):
            for k in self.list:
                limgdir=os.path.join(self.ldir[d], "image_" + str(k).zfill(4) + ".png")
                rimgdir=os.path.join(self.rdir[d], "image_" + str(k).zfill(4) + ".png")
                if path.exists(limgdir) and path.exists(rimgdir):
                    left = Image.open(limgdir)
                    left=np.array(left, dtype='float64')
                    left = np.reshape(left[:,:,0:3] , (480,640,3,1))
                    if len(self.all_left_img)>0:
                        self.all_left_img = np.append( self.all_left_img , left, axis = 3)
                    else:
                        self.all_left_img = left
                    right = Image.open(rimgdir)
                    right = np.array(right, dtype='float64')
                    right = np.reshape(right[:,:,0:3] , (480,640,3,1))
                    if len(self.all_right_img)>0:
                        self.all_right_img = np.append(self.all_right_img , right, axis = 3)
                    else:
                        self.all_right_img = right
                    n+=1

        if len(self.all_left_img)>0 and len(self.all_right_img)>0:
                self.all_left_img = np.transpose(self.all_left_img, (3, 0, 1,2))
                self.all_right_img = np.transpose(self.all_right_img, (3, 0, 1,2))
                print(self.all_left_img.shape)
                print(self.all_right_img.shape)
        return n

    def re_inti(self,list):
        self.list = list
        self.all_left_img = []
        self.all_right_img = []
