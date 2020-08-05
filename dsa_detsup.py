#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:46:32 2020

@author: ocrusr
"""

import numpy as np
import os
import sys
import tensorflow as tf
import cv2
import argparse

#From Tensorflow Object Detection API
import zipfile
import tarfile
import six.moves.urllib as urllib
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
#sys.path.append("..")
from object_detection.utils import ops as utils_ops
#from TFObjectDetection.object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

#From DeepSort
from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from tools import generate_detections
#from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import generate_detector as gendet
from tools.generate_detections import extract_image_patch
import deep_sort_app as dsa
#From DeepSort for Video
from insuranceTool import generate_detections as gdet
'''
Code from tensorflow object detection API
Load pretrained detector
from tensorflow object detection API
'''
metricFileName = 'market1501.pb'
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('TFObjectDetection/object_detection/data', 'mscoco_label_map.pbtxt')
opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())
'''    
Code from tensorflow object detection API
Load inference tensorflow graph from loaded pretrained detector
And apply to image
'''


def run_inference_for_single_image(image, graph,PATH_TO_FROZEN_GRAPH):
  with graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
          serialized_graph = fid.read()
          od_graph_def.ParseFromString(serialized_graph)
          tf.import_graph_def(od_graph_def, name='')
          with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
              tensor_name = key + ':0'
              #print("tensorname:",tensor_name)
              if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
                #print("tensordict:",tensor_dict)
            if 'detection_masks' in tensor_dict:
              # The following processing is only for single image
              detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
              detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
              # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
              real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
              detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
              detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
              detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                  detection_masks, detection_boxes, image.shape[0], image.shape[1])
              detection_masks_reframed = tf.cast(
                  tf.greater(detection_masks_reframed, 0.5), tf.uint8)
              # Follow the convention by adding back the batch dimension
              tensor_dict['detection_masks'] = tf.expand_dims(
                  detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
      
            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})
      
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            #print(output_dict)
            #print('*'*50,output_dict['detection_classes'])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
              output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


def tlbr2tlwh(tlbr):
    tlwh = tlbr.copy()
    for i in range(len(tlbr)):
        tlwh[2:] -=tlwh[:2]
    return tlwh





'''
Load video and Initialize Tracker
'''
#capture = cv2.VideoCapture(args.videofile)
capture = cv2.VideoCapture('video.mp4')
metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine",matching_threshold=0.2)
tracker = Tracker(metric)

'''
variable fix
'''
min_confidence = 0.3
nn_budget =100
nms_max_overlap=1.0
IMAGE_SIZE = (12, 8)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
frameIdx=0 # When video frame loaded, frameIdx is required to support MOT format. Increment in while loop
patch_shape=None
min_detection_height=0
max_cosine_distance=0.2

savepath=''
'''
Load Detection
'''



'''
Load Cosine Metric
'''
#desiredShape = (128,64)

encoder = generate_detections.create_box_encoder(metricFileName,batch_size=1)


def metric(frame,detectionResult,metric_graph,metricFileName):
    img = extract_image_patch(frame,detectionResult[0:4],patch_shape)
    print(np.asarray(img))
    #plt.imshow(img)
    #return img    
    with metric_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(metricFileName, 'rb') as trkgph:
        serialized_graph = trkgph.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='net')
        #To check inside of pretrained model, use below information
      with tf.Session() as sesss:
          ops = tf.get_default_graph().get_operations()
          all_tensor_names = {output.name for op in ops for output in op.outputs}
          input_var = tf.get_default_graph().get_tensor_by_name("net/%s:0" % "images")
          output_var = tf.get_default_graph().get_tensor_by_name("net/%s:0" % "features")
          assert len(output_var.get_shape()) == 2
          assert len(input_var.get_shape()) == 4
          feature_dim = output_var.get_shape().as_list()[-1]
          image_shape = input_var.get_shape().as_list()[1:]
          output = sesss.run(img)
          return output


#detections_out += [np.r_[(row, feature)] for row, feature
#                               in zip(rows, features)]

def ratio2realsize(frame,detectionResult):
    WID = frame.shape[1]
    HEI = frame.shape[0]
    for i in range(len(detectionResult)):
        detectionResult[i][0] = round(detectionResult[i][0]*WID)
        detectionResult[i][1] = round(detectionResult[i][1]*HEI)
        detectionResult[i][2] = round(detectionResult[i][2]*WID)
        detectionResult[i][3] = round(detectionResult[i][3]*HEI)
    return detectionResult



    

'''
Run Tracker while video is played
'''
while capture.isOpened():
    ret, frame = capture.read()
    #frame = np.expand_dims(frame, axis=0)
    #frame = cv2.resize(frame,(64,128))
    detection_graph = tf.Graph()
    detectionResult = run_inference_for_single_image(frame,detection_graph,PATH_TO_FROZEN_GRAPH)
    tf.reset_default_graph()
    #visualization_of_detection(frame,res,category_index)
    box,det_class,score = detectionResult["detection_boxes"], detectionResult["detection_classes"], detectionResult["detection_scores"]
    combined=np.asarray(list(zip(box[:,0],box[:,1],box[:,2],box[:,3],det_class,score))) # 0to3 box 4 class 5 conf
    personDetected = np.asarray([d for d in combined if d[4]==1 or d[5]>=min_confidence])
    personDetected = ratio2realsize(frame,personDetected)
    # metric_graph = tf.Graph()
    # featmat = []
    # for i in range(len(personDetected)):
    #     feature = metric(frame,personDetected[i],metric_graph,metricFileName)
    #     featmat.append(feature)
    #np.asarray(featmat)
    #feature = metric(frame,personDetected,metric_graph,metricFileName)
    
    # for i in range(len(personDetected)):
    #     want = personDetected[i][:4]
    #     feature.append(encoder(frame,personDetected[i][:4].copy()))
    
    feature = encoder(frame,personDetected[:,:4].copy())
    motformat_front=np.zeros((len(personDetected),2))
    motformat_front[:,0]=frameIdx
    motformat_front[:,1]=-1
    motformat_rear=np.zeros((len(personDetected),3))
    motformat_rear[:,:]=-1
    detected = zip(motformat_front,personDetected,motformat_rear,feature)
    #detected = np.asarray([motformat_front,personDetected,motformat_rear,feature])
    print()
    print("kkk")
    print()
    tf.reset_default_graph()

    seqinfo = {
        "sequence_name": "None",
        "image_filenames": "No need",
        "detections": detected,
        "groundtruth": None,
        "image_size": [frame.shape[0],frame.shape[0]],
        "min_frame_idx": 1,
        "max_frame_idx": 1,
        "feature_dim": 128,
        "update_ms": None
    }    





    dsa.run(frame,seqinfo," ",savepath,min_confidence,nms_max_overlap,min_detection_height,max_cosine_distance,nn_budget,display=True)
    frameIdx=frameIdx+1
    #buildDetectionList_TFOD(res)
    
    #box=tlbr2tlwh(box) #tlbr -> tlwh
    #boxes=box_size(frame,res)


#detections_out += [np.r_[(row, featmat)] for row, featmat in zip(rows, featmat)]



#Below codes : Requirement for Cosine Metric
    
    
    


    # # Run non-maxima suppression.
    # boxes = np.array([d.tlwh for d in detections])
    # scores = np.array([d.confidence for d in detections])
    # indices = preprocessing.non_max_suppression(
    #     boxes, nms_max_overlap, scores
    # )
    # detections = [detections[i] for i in indices]

    # # Call the tracker
    # tracker.predict()
    # tracker.update(detections)    
    # cv2.rectangle(frame,
    #               (boxes[0], boxes[1]), (boxes[2], boxes[3]),
    #               (255,255,255), 2)
    # cv2.imshow('a',frame)    
    
    
    
capture.release()
cv2.destroyAllWindows()
