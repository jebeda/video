#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:05:44 2020

@author: ocrusr
"""
'''
AnnotationToolV2.py
Input : video folder
Output : Images with PERSONID_CAMSEQUENCE_FRAMEIDX.jpg

Later add making folder automatically(with checking)

Changes:
Include 'lie_down' (line 72)
modified for changes in filename of img and ann
line 134 & line 53 : use 11 instead of 5
'''

'''
Load modules for support
'''
import numpy as np
import xml.etree.ElementTree as et
import os
from os import listdir
from os.path import isfile, join
from cv2 import cv2
import matplotlib.pyplot as plt
import argparse
import sys
%matplotlib inline
'''
Load list of img && ann
'''
#
# inputpath = '190503_3' #Later receive as argument
# annfolder = '/annotations/'
# imgfolder = '/images/'
# savefolder = '/result/'
# annpath =inputpath+annfolder 
# imgpath = inputpath+imgfolder
# savepath = inputpath+savefolder
# annfiles = [f for f in listdir(annpath) if isfile(join(annpath, f))]
# imgfiles = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
# imgfiles.sort()
def sortName(annfiles):
    sortedAnnName=[]
    i=0
    for ks in annfiles:
        num=ks[:-4]
        num=num[11:]
        sortedAnnName.append([int(num),i])
        i=i+1
    sortedAnnName.sort()
    desiredArray=np.asarray(sortedAnnName)
    want = desiredArray[:,1]
    annfiles=[annfiles[i] for i in want]
    return annfiles
'''
Load xml file
Read object list
Extract if object is person
'''
def XML2personlist(annpath,annfiles):
    current_xml = et.parse(annpath+annfiles).getroot()
    get_object = current_xml.findall('object')
    personlist = []
    for obj in get_object:
        is_person = obj.find('name').text
        if is_person =="person" or is_person =="lie_down":
            if is_person=="lie_down":
                print("lie_down detected")
            personlist.append(obj)
    return personlist
#delte below
# current_xml = et.parse(annpath+annfiles[0]).getroot()
# get_object = current_xml.findall('object')
# personlist = []
# for obj in get_object:
#     is_person = obj.find('name').text
#     if is_person =="person":
#         personlist.append(obj)

'''
For each person
Read Xmin Ymin Xmax Ymax frin bndbox
Result : array with [Xmin, Ymin, Xmax, Ymax]
'''
def locatePerson(personlist):
    person_coordinate = []
    for perlis in personlist:
        bndbox = perlis.findall('bndbox')
        #print(bndbox)
        for bdbox in bndbox:
            path_xmin = bdbox.find('xmin')
            xmin = int(path_xmin.text)
            path_ymin = bdbox.find('ymin')
            ymin = int(path_ymin.text)
            path_xmax = bdbox.find('xmax')
            xmax = int(path_xmax.text)
            path_ymax = bdbox.find('ymax')
            ymax = int(path_ymax.text)
        person_coordinate.append([xmin,ymin,xmax,ymax])
    person_coordinate=np.asarray(person_coordinate)
    return person_coordinate
    
#delte below
# person_coordinate = []
# for perlis in personlist:
#     bndbox = perlis.findall('bndbox')
#     #print(bndbox)
#     for bdbox in bndbox:
#         path_xmin = bdbox.find('xmin')
#         xmin = int(path_xmin.text)
#         path_ymin = bdbox.find('ymin')
#         ymin = int(path_ymin.text)
#         path_xmax = bdbox.find('xmax')
#         xmax = int(path_xmax.text)
#         path_ymax = bdbox.find('ymax')
#         ymax = int(path_ymax.text)
#     person_coordinate.append([xmin,ymin,xmax,ymax])
# person_coordinate=np.asarray(person_coordinate)
    
'''
Load img based on img name saved in xml file in 'filename'
And extract frame number as frameIdx
'''
def loadImg(annpath,annfiles,imgpath):
    current_xml = et.parse(annpath+annfiles).getroot()
    get_filename = current_xml.findall('filename')
    for name in get_filename:
        jpgname = name.text
        temp = jpgname[:-4]
        temp = temp[11:]
    frameIdx = int(temp)
    img = cv2.imread(imgpath+jpgname)
    return img,frameIdx,jpgname

#Delete below
# get_filename = current_xml.findall('filename')
# for name in get_filename:
#     jpgname = name.text
#     temp = jpgname[:-4]
#     temp = temp[5:]
# frameIdx = int(temp)
# img = cv2.imread(imgpath+jpgname)

'''
Support Visulization
For each img
Crop based on person_coordinate
crop area : img[y:y+h,x:x+w]
Visualize for accurate cropping
'''
def visualization(img,curname):
    #img.astype(np.uint8)
    plt.title("cur: "+curname)
    plt.imshow(img)
    plt.show()
def visualize_previousimg(img,previous_coordinate,idxlist):
    #idxlist.sort()
    #img.astype(np.uint8)
    prevImg=img.copy()
    plt.title('previous img')
    i=0
    for numOfPeople in previous_coordinate:
        startpoint=(numOfPeople[0],numOfPeople[1])
        endpoint=(numOfPeople[2],numOfPeople[3])
        color = (0,0,255)
        thickness = 2
        #print('cjl:',idxlist)
        plt.text(numOfPeople[0],numOfPeople[1],idxlist[i])
        i=i+1
        prevImg = cv2.rectangle(prevImg,startpoint,endpoint,color,thickness)        
    plt.imshow(prevImg)
    plt.show()
# for cord in person_coordinate:
#     #print(cord)
#     height = cord[3]-cord[1]
#     width = cord[2]-cord[0]
#     img4crop = img.copy()
#     img_crop = img4crop[cord[1]:cord[1]+height,cord[0]:cord[0]+width]
#     #print(img_crop)
#     #visualization(img_crop)

def cropANDsave(img,cord,path,index,camSeq,frameIdx):
    height = cord[3]-cord[1]
    width = cord[2]-cord[0]
    img4crop = img.copy()
    img_crop = img4crop[cord[1]:cord[1]+height,cord[0]:cord[0]+width]
    imgName = path+str(index)+'_'+str(camSeq)+'_'+str(frameIdx)+'.jpg'
    print(imgName)
    cv2.imwrite(imgName,img_crop)
    
'''
(For verify right detection)
Draw box using Xmin, Ymin, Xmax, Ymax extracted from bndbox

'''
'''
Initialize tool
'''
def Annotation(imgpath,jpgname,person_coordinate,previousImg,previousCord,frameIdx,savepath,idxlist,camSeq):
    previousName = 'Previous annotated img'
    if previousImg is not 0:
        #cv2.imshow(previousName,previousImg)
        #cv2.waitKey(1)
        visualize_previousimg(previousImg,previousCord,idxlist)
    i = 0
    idxlist = []
    for numOfPeople in person_coordinate:
        img = cv2.imread(imgpath+jpgname)
        winName = str(numOfPeople[0])
        startpoint=(numOfPeople[0],numOfPeople[1])
        endpoint=(numOfPeople[2],numOfPeople[3])
        color = (0,0,255)
        thickness = 2
        imgbr = cv2.rectangle(img.copy(),startpoint,endpoint,color,thickness)
        visualization(imgbr,jpgname)
        #cv2.imshow(winName,imgbr)
        #idx=sys.argv()
        idx=input("index:")
        idxlist.append(idx)
        #print(idxlist)
        #cv2.destroyAllWindows()
        cropANDsave(img,numOfPeople,savepath,idx,camSeq,frameIdx)
        plt.close()
    previousImg = img
    previousCord = person_coordinate
    if i!=0:
        pass
        #cv2.destroyWindow(previousName)
    i=i+1
    #print(previousCord)
    return previousImg,previousCord,idxlist
    
# idxlist = []
# i = 0

# for numOfPeople in person_coordinate:
#     img = cv2.imread(imgpath+jpgname)
#     winName = str(numOfPeople[0])
#     startpoint=(numOfPeople[0],numOfPeople[1])
#     endpoint=(numOfPeople[2],numOfPeople[3])
#     color = (0,0,255)
#     thickness = 2
#     imgbr = cv2.rectangle(img.copy(),startpoint,endpoint,color,thickness)
#     visualization(imgbr)
#     #cv2.imshow(winName,imgbr)
#     idx=input()
#     idxlist.append(idx)
#     print(idxlist)
#     #cv2.destroyAllWindows()
#     cropANDsave(img,numOfPeople,savepath,idx,1,frameIdx)
#     plt.close()
# previous = person_coordinate




def main():
    cv2.startWindowThread()
    camSeq =12 #6 must be used for 00004
    inputpath = '00010'#'190503_3' #Later receive as argument
    annfolder = '/annotations/'
    imgfolder = '/images/'
    savefolder = '/result/'
    annpath =inputpath+annfolder 
    imgpath = inputpath+imgfolder
    savepath = inputpath+savefolder
    annfiles = [f for f in listdir(annpath) if isfile(join(annpath, f))]
    imgfiles = [f for f in listdir(imgpath) if isfile(join(imgpath, f))]
    annfiles=sortName(annfiles)
    #print(annfiles)
    previousImg = 0
    previousCord=0
    idxlist=0
    #170 error
    for i in range(len(annfiles)):
        personlist=XML2personlist(annpath,annfiles[i])
        person_coordinate=locatePerson(personlist)
        img,frameIdx,jpgname=loadImg(annpath,annfiles[i],imgpath)
        previousImg,previousCord,idxlist=Annotation(imgpath,jpgname,person_coordinate,previousImg,previousCord,frameIdx,savepath,idxlist,camSeq)
        #print(previousImg)


if __name__ == "__main__":
    main()

'''
Information
videoName : number of unique humans
05xxx : 4(4)
00000 : 3(7)
00001 : 15(22)
00002 : 1(23)
00003 : 7(30)
00004 : skip
00005 : 2(32)
00006 : 1(33)
00007 : 2(35)
00008 : 4(39)
00009 : 2(41)
00010 : 

Since same people occurs, below I recorded a list in case of changing for same identity
00000 : 5 / 6
00001 : 8 / 9
00002 : 23
00003 : 24 / 27 /26(person with smart phone capturing)     
00004 : skip
00005 : 31 / /32(?)
00006 : 33
00007 : 34 /  /  / 35
00008 : 36 /  /  / 37
00009 : 40 /  /  / 41
00010 : 42 /

Warning - Error occured in :
05xxx : 170
00006 : 8


'''
