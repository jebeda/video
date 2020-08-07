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
#%matplotlib inline
#From Tensorflow Object Detection API
import tarfile
import six.moves.urllib as urllib
from matplotlib import pyplot as plt
#sys.path.append("..")
from object_detection.utils import ops as utils_ops
from TFObjectDetection.object_detection.utils import ops as utils_ops
from TFObjectDetection.object_detection.utils import visualization_utils as vis_util
from TFObjectDetection.object_detection.utils import label_map_util


#From DeepSort
from application_util import preprocessing
from deep_sort import nn_matching
from tools import generate_detections
from deep_sort.detection import Detection # this file is modified
from deep_sort.tracker import Tracker

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

savepath='outputvideo/testRecord.txt'
'''
Load Detection
'''
def detectionVisualization(image_np,output_dict,category_index):
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    #plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    plt.show()

def visualize_previousimg(img,previous_coordinate,message):
    #idxlist.sort()
    #img.astype(np.uint8)
    prevImg=img.copy()
    plt.title(message)
    i=0
    for numOfPeople in previous_coordinate:
        startpoint=(numOfPeople[0],numOfPeople[1])
        endpoint=(numOfPeople[2],numOfPeople[3])
        #startpoint=(numOfPeople[1],numOfPeople[0])
        #endpoint=(numOfPeople[3],numOfPeople[2])

        color = (0,0,255)
        thickness = 2
        #print('cjl:',idxlist)
        plt.text(numOfPeople[0],numOfPeople[1],'Person')
        i=i+1
        prevImg = cv2.rectangle(prevImg,startpoint,endpoint,color,thickness)        
    plt.imshow(prevImg)
    plt.show()

'''
Load Cosine Metric
'''
#desiredShape = (128,64)

encoder = generate_detections.create_box_encoder(metricFileName,batch_size=1)



def ratio2realsize(frame,detectionResult):
    WID = frame.shape[1]
    HEI = frame.shape[0]
    for i in range(len(detectionResult)):
        detectionResult[i][0] = round(detectionResult[i][0]*WID)
        detectionResult[i][1] = round(detectionResult[i][1]*HEI)
        detectionResult[i][2] = round(detectionResult[i][2]*WID)
        detectionResult[i][3] = round(detectionResult[i][3]*HEI)
    return detectionResult

def formatting2mot(feature,personDetected):
    motformat_front=np.zeros((len(personDetected),2))
    motformat_front[:,0]=frameIdx
    motformat_front[:,1]=-1
    motformat_rear=np.zeros((len(personDetected),3))
    motformat_rear[:,:]=-1
    detected= np.concatenate((motformat_front,personDetected,motformat_rear,feature),axis=1)
    return detected

detection_graph = tf.Graph()

'''
Run Tracker while video is played
'''
while capture.isOpened():
    ret, frame = capture.read()
    #frame = np.expand_dims(frame, axis=0)
    #frame = cv2.resize(frame,(64,128))
    #detection_graph = tf.Graph()
    detectionResult = run_inference_for_single_image(frame,detection_graph,PATH_TO_FROZEN_GRAPH)
    #tf.reset_default_graph()
    boxes,det_classes,scores = detectionResult["detection_boxes"], detectionResult["detection_classes"], detectionResult["detection_scores"]

    combined=np.asarray(list(zip(boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2],det_classes,scores))) # 0to3 box 4 class 5 conf
    personDetected = np.asarray([d for d in combined if d[4]==1 and d[5]>=min_confidence])
    personDetected = ratio2realsize(frame,personDetected)
    personDetected_box = personDetected[:,0:4]
    for kk in range(len(personDetected_box)):
        personDetected_box[kk][2]=personDetected_box[kk][2]-personDetected_box[kk][0]
        personDetected_box[kk][3]=personDetected_box[kk][3]-personDetected_box[kk][1]
    #visualize_previousimg(frame,personDetected,"after detection") #until here is good
    features = encoder(frame,personDetected[:,:4].copy())
    detected =formatting2mot(features,personDetected)


    #tf.reset_default_graph()

    seqinfo = {
        "sequence_name": "None",
        "image_filenames": "No need",
        "detections": detected,
        "groundtruth": None,
        "image_size": [frame.shape[0],frame.shape[1]],
        "min_frame_idx": 0,
        #"max_frame_idx": frameIdx+1,
        "feature_dim": 128,
        "update_ms": None
    }
    ##If error occurs, need to set if statement for zero cases of detection
    detections = [Detection(box,det_class,feature) for box,det_class,feature in zip(personDetected_box,det_classes,features)]
    check = [Detection(box,det_class,feature) for box,det_class,feature in zip(boxes,det_classes,features)]
    
    detections_box = np.array([dlc.tlwh for dlc in detections])
    detections_score = np.array([dlc.confidence for dlc in detections])
    detections_indices = preprocessing.non_max_suppression(detections_box,nms_max_overlap,detections_score)
    detections = [detections[i] for i in detections_indices]
    tracker.predict()
    tracker.update(detections)
    frameIdx=frameIdx+1
    
    img4show=frame.copy()
    for track in tracker.tracks:
        bbox = [max(0, int(x)) for x in track.to_tlbr()]
        cv2.rectangle(img4show,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)
        cv2.putText(img4show,str(track.track_id),(bbox[0], bbox[1]+30),0,5e-3 * 200,(0,255,0),2)
    #plt.imshow(img4show)
    #plt.show()
    img4show=np.asarray(img4show,dtype='uint8')
    print()
    print("shape of img4show:",img4show.shape)
    print()
    cv2.imshow('a',img4show)
    cv2.waitKey(1)
    #cv2.destroyAllWindows()



    #dsa.run(frame,seqinfo," ",savepath,min_confidence,nms_max_overlap,min_detection_height,max_cosine_distance,nn_budget,display=True)
    
    #'''
    
    
capture.release()
#cv2.destroyAllWindows()
