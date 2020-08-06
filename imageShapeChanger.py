#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:13:26 2020

@author: ocrusr
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import cv2
import os
from os.path import isfile, join
'''
folname = 'bounding_box_train_origin'
savepath = 'bounding_box_train'

filelist = [f for f in os.listdir(folname) if isfile(join(folname, f))]

img = cv2.imread(savepath+'/'+filelist[2],cv2.IMREAD_COLOR)
plt.imshow(img)
for i in range(len(filelist)):
    print("cur number:",i)
    image = cv2.imread(folname+'/'+filelist[i],cv2.IMREAD_COLOR)
    image = cv2.resize(image,(64,128))
    filename=savepath+'/'+filelist[i]
    cv2.imwrite(filename, image)
    
    
print("save finished")
'''
'''
testpath = 'bounding_box_test_origin'
testsavepath = 'bounding_box_test'
querypath = 'query_origin'
querysavepath = 'query'

filelist = [f for f in os.listdir(querysavepath) if isfile(join(querysavepath, f))]

img = cv2.imread(querysavepath+'/'+filelist[1],cv2.IMREAD_COLOR)
plt.imshow(img)
'''



