#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 09:05:44 2020

@author: ocrusr
"""
'''
AnnotationToolV3.py
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
#%matplotlib inline


from deep_sort.application_util import preprocessing
from deep_sort.deep_sort import nn_matching
from deep_sort.tools import generate_detections
from deep_sort.deep_sort.detection import Detection # this file is modified
from deep_sort.deep_sort.tracker import Tracker





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
        color = (0,255,0)
        thickness = 4
        #print('cjl:',idxlist)
        plt.text(numOfPeople[0],numOfPeople[1],idxlist[i])
        i=i+1
        prevImg = cv2.rectangle(prevImg,startpoint,endpoint,color,thickness)        
    plt.imshow(prevImg)
    plt.show()


def cropANDsave(img,cord,path,index,camSeq,frameIdx):
    height = cord[3]-cord[1]
    width = cord[2]-cord[0]
    img4crop = img.copy()
    img_crop = img4crop[cord[1]:cord[1]+height,cord[0]:cord[0]+width]
    imgName = path+str(index)+'_'+str(camSeq)+'_'+str(frameIdx)+'.jpg'
    print(imgName)
    cv2.imwrite(imgName,img_crop)
    
'''
Initialize tool
'''



def Annotation(imgpath,jpgname,tracker,previousImg,previousCord,frameIdx,savepath,idxlist,camSeq):
    previousName = 'Previous annotated img'
    if previousImg is not 0:
        pass
        #cv2.imshow(previousName,previousImg)
        #cv2.waitKey(1)
        #visualize_previousimg(previousImg,previousCord,idxlist)
    i = 0
    idxlist = []
    bbox_list=[]
    img_origin = cv2.imread(imgpath+jpgname)
    img=img_origin.copy()
    
    
    
    winName = str(jpgname)
    for tracked in tracker.tracks:
        bbox = [max(0, int(x)) for x in tracked.to_tlbr()]
        cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)
        cv2.putText(img,str(tracked.track_id),(bbox[0], bbox[1]+30),0,5e-3 * 200,(0,255,0),2)
    img = np.asarray(img,dtype='uint8')
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow(winName,img)
    cv2.waitKey(2000)
    yorn=input("y or n:")
    if yorn=="y":
        for js in range(len(tracker.tracks)):
            bbox = [max(0, int(x)) for x in tracker.tracks[js].to_tlbr()]
            cropANDsave(img_origin,bbox,savepath,tracker.tracks[js].track_id,camSeq,frameIdx)
        cv2.destroyWindow(winName)
    
    else:           
        for numOfPeople in tracker.tracks:
            img = cv2.imread(imgpath+jpgname)
            winName = str('aaa')
            bbox = [max(0, int(x)) for x in numOfPeople.to_tlbr()]
            cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)
            cv2.putText(img,str(numOfPeople.track_id),(bbox[0], bbox[1]+30),0,5e-3 * 200,(0,255,0),2)
            idxlist.append(numOfPeople.track_id)
            #startpoint=(numOfPeople[0],numOfPeople[1])
            #endpoint=(numOfPeople[2],numOfPeople[3])
            #color = (0,0,255)
            #thickness = 4
            #imgbr = cv2.rectangle(img.copy(),startpoint,endpoint,color,thickness)
            #visualization(imgbr,jpgname)
            #frame=cv2.cvtColor(imgbr, cv2.COLOR_BGR2RGB)
            #frame = cv2.resize(frame,(1024,1024))
            frame = np.asarray(img,dtype='uint8')
            #frame = cv2.resize(frame,(4096,4096))
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow('curvideo',frame)
            cv2.waitKey(1000)
            #cv2.imshow(winName,frame)
            #cv2.waitKey(200)
            #idx=sys.argv()
            idx=input("index:")
            cropANDsave(img,bbox,savepath,idx,camSeq,frameIdx)
            numOfPeople.track_id=idx
            idxlist.append(idx)
            #print(idxlist)
            #cv2.destroyAllWindows()
            #print(numOfPeople)
            bbox_list.append(bbox)
    previousImg = img
    previousCord = bbox_list
    if i!=0:
        pass
        #cv2.destroyWindow(previousName)
    i=i+1
    #print("idxlist:",idxlist)
    #print(previousCord)
    cv2.destroyAllWindows()
    return previousImg,previousCord,idxlist




def main():
    cv2.startWindowThread()
    camSeq =210 #6 must be used for 00004
    inputpath = '20190510/00000'#'190503_3' #Later receive as argument
    annfolder = '/annotations/'
    imgfolder = '/images/'
    savefolder = '/result/'
    annpath =inputpath+annfolder 
    imgpath = inputpath+imgfolder
    savepath = inputpath+savefolder
    annfiles = [f for f in listdir(annpath) if isfile(join(annpath, f))]
    annfiles=sortName(annfiles)
    #print(annfiles)
    
    metricFileName = 'deep_sort/market1501.pb'
    min_confidence = 0.3
    nn_budget =100
    nms_max_overlap=1.0
    
    frameIdx=0 # When video frame loaded, frameIdx is required to support MOT format. Increment in while loop
    patch_shape=None
    min_detection_height=0
    max_cosine_distance=0.2
    metric = nn_matching.NearestNeighborDistanceMetric("cosine",matching_threshold=0.2)
    tracker = Tracker(metric)
    encoder = generate_detections.create_box_encoder(metricFileName,batch_size=1)    
    
    previousImg = 0
    previousCord=0
    idxlist=0
    
    for i in range(len(annfiles)):
        personlist=XML2personlist(annpath,annfiles[i])
        person_coordinate=locatePerson(personlist)
        img,frameIdx,jpgname=loadImg(annpath,annfiles[i],imgpath)
        personDetected_box = person_coordinate
        
        for kk in range(len(person_coordinate)):
            person_coordinate[kk][2]=person_coordinate[kk][2]-person_coordinate[kk][0]
            person_coordinate[kk][3]=person_coordinate[kk][3]-person_coordinate[kk][1]
        
        
        
        features = encoder(img,person_coordinate)
        detections = [Detection(box,1,feature) for box,feature in zip(personDetected_box,features)]
        detections_box = np.array([dlc.tlwh for dlc in detections])
        detections_score = np.array([dlc.confidence for dlc in detections])
        detections_indices = preprocessing.non_max_suppression(detections_box,nms_max_overlap,detections_score)
        detections = [detections[i] for i in detections_indices]
        tracker.predict()
        tracker.update(detections)
        
        
        previousImg,previousCord,idxlist=Annotation(imgpath,jpgname,tracker,previousImg,previousCord,frameIdx,savepath,idxlist,camSeq)
        
        
        frameIdx=frameIdx+1    




if __name__ == "__main__":
    main()
