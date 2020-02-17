import numpy as np
from PIL import Image
import os
from os import path

#Paths:
#ldir = left images directory
#rdir = right images directory
#ddir = depth images directory
class Reader(object):
    def __init__(self,ldir=[],rdir=[],ddir=[],rgb=True):
        self.list = []
        self.ddir = ddir
        self.ldir = ldir
        self.rdir = rdir
        self.all_left_img = []
        self.all_right_img = []
        self.all_depth_img = []
        self.rgb=rgb

    def pick_up_epochs(self, powerDepth=1):
        n=0
        for d in range(len(self.ddir)):
            for k in self.list:
                dimgdir=os.path.join(self.ddir[d], "image_" + str(k).zfill(4) + ".png")
                limgdir=os.path.join(self.ldir[d], "image_" + str(k).zfill(4) + ".png")
                rimgdir=os.path.join(self.rdir[d], "image_" + str(k).zfill(4) + ".png")
                if path.exists(dimgdir) and path.exists(limgdir) and path.exists(rimgdir):
                    depth = Image.open(dimgdir)
                    depth = np.array(depth, dtype='float64') # this proccess needs to be changed as we are coding depth into rgb
                    if len(depth.shape)>2:
                        depth = 1-((1/256.0*depth[:,:,0])+(1/65536.0*depth[:,:,1])+(1/16777216.0*depth[:,:,2]))# decode rgd to depth
                    else:
                        depth = 1-(depth/15000)
                    # nonlinearly enhance the near distance
                    depth = np.power(depth,powerDepth)
                    if len(self.all_depth_img)>0:
                        self.all_depth_img = np.dstack((self.all_depth_img,depth))
                    else:
                        self.all_depth_img = depth
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

        if len(self.all_depth_img)>0 and len(self.all_left_img)>0 and len(self.all_right_img)>0:
                self.all_depth_img = np.transpose(self.all_depth_img, (2, 0, 1))
                self.all_left_img = np.transpose(self.all_left_img, (3, 0, 1,2))
                self.all_right_img = np.transpose(self.all_right_img, (3, 0, 1,2))
                self.all_depth_img = np.reshape(self.all_depth_img[:,:,:] , (n,480,640,1))
                print(self.all_depth_img.shape)
                print(self.all_left_img.shape)
                print(self.all_right_img.shape)
        return n

    def re_inti(self,list):
        self.list = list
        self.all_left_img = []
        self.all_right_img = []
        self.all_depth_img = []
