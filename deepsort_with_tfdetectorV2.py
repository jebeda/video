#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:46:32 2020

@author: ocrusr
"""
'''
deepsort_with_tfdetectorV2.py
Input : Video
Output : frame with tracking result

Major Changes :
    1. For better detection, implemented ssd_resnet50 + (640,640)
    
To Increase Speed, below changes are implemented:
    1. Model should be downloaded before code is running(To prevent dowloading everytime)
    2. Detection Graph is loaded before video is loaded -> video capture will be included in session
    3. Tensorflow detection API-run inference function is included in session.
    4. Tried to apply function results into 'frame' itself

'''


'''
Import
'''
import numpy as np
import tensorflow as tf
import cv2
import argparse
#%matplotlib inline
#From Tensorflow Object Detection API
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
Variable Controller
metricFileName : Loading Cosine Metric function
MODEL_NAME : Detection model

'''
#metric models : 'market1501.pb','MTMC.pb','ours.pb','mars-small128.pb'
metricFileName = 'ours.pb'
#metricFileName = 'mars-small128.pb'
#MODEL_NAME='ssd_resnet50_v1_fpn_tf1'
MODEL_NAME='tf1_faster_rcnn_resnet50_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
#MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
min_confidence = 0.7
nms_max_overlap=1.0
frameIdx=0 # When video frame loaded, frameIdx is required to support MOT format. Increment in while loop

#patch_shape=None
max_cosine_distance=0.05
matching_threshold=0.8
savepath='outputvideo/testRecord.txt'

#for TF2
#MODEL_NAME = 'ssd_resnet50_v1_fpn_640x640'
#PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/saved_model/saved_model.pb'
'''    
INITIALIZATION


Code from tensorflow object detection API
Load inference tensorflow graph from loaded pretrained detector
'''
detection_graph = tf.Graph()
with detection_graph.as_default():
     od_graph_def = tf.GraphDef()
     with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
         serialized_graph = fid.read()
         od_graph_def.ParseFromString(serialized_graph)
         tf.import_graph_def(od_graph_def, name='')

'''
Load video and Initialize Tracker & encoder
'''
#capture = cv2.VideoCapture(args.videofile)
#capture = cv2.VideoCapture('video.mp4')
capture = cv2.VideoCapture('00019.MTS')

metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine",matching_threshold=matching_threshold)
tracker = Tracker(metric,1)
encoder = generate_detections.create_box_encoder(metricFileName,batch_size=1)



'''
Built functions for support
1. ratio2realsize : convert normalized detection scale into actual width & height


'''
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
Main function



'''
BASE_FPS=30

with tf.Session(graph=detection_graph) as sess:
  # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()

    '''
    Run Tracker while video is played
    '''
    while capture.isOpened():
        ret, frame = capture.read()
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_interval = BASE_FPS // fps if fps > 0 else 0
        #print("fps:",fps)
        delay = int(1000/fps)
        frame = cv2.resize(frame,(640,640))
        
        if 0 < fps and frameIdx % frame_interval != 0:
            frameIdx += 1
            print()
            print("Continue from frameIDx activated")
            print()
            continue        
        
        '''
        From Tensorflow Object Detection API
        Below codes are used in order to get keys
        '''
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
              detection_masks, detection_boxes, frame.shape[0], frame.shape[1])
          detection_masks_reframed = tf.cast(
              tf.greater(detection_masks_reframed, 0.5), tf.uint8)
          # Follow the convention by adding back the batch dimension
          tensor_dict['detection_masks'] = tf.expand_dims(
              detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    
        # Run inference
        detectionResult = sess.run(tensor_dict,
                               feed_dict={image_tensor: np.expand_dims(frame, 0)})
  
        # all outputs are float32 numpy arrays, so convert types as appropriate
        detectionResult['num_detections'] = int(detectionResult['num_detections'][0])
        detectionResult['detection_classes'] = detectionResult['detection_classes'][0].astype(np.uint8)
        detectionResult['detection_boxes'] = detectionResult['detection_boxes'][0]
        detectionResult['detection_scores'] = detectionResult['detection_scores'][0]
        if 'detection_masks' in detectionResult:
          detectionResult['detection_masks'] = detectionResult['detection_masks'][0]
        '''
        Tensorflow Object Detection API code ended
        
        
        '''
        boxes,det_classes,scores = detectionResult["detection_boxes"], detectionResult["detection_classes"], detectionResult["detection_scores"]
        combined=np.asarray(list(zip(boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2],det_classes,scores))) # 0to3 box 4 class 5 conf
        personDetected = np.asarray([d for d in combined if d[4]==1 and d[5]>=min_confidence])
        personDetected = ratio2realsize(frame,personDetected)
        #personDetected = preprocessing.non_max_suppression(personDetected,1)
        #In case of no detection result, continue code is implemented
        if len(personDetected) ==0:
            print('Cont activated')
            continue
        personDetected_box = personDetected[:,0:4]
        for kk in range(len(personDetected_box)):
            personDetected_box[kk][2]=personDetected_box[kk][2]-personDetected_box[kk][0]
            personDetected_box[kk][3]=personDetected_box[kk][3]-personDetected_box[kk][1]
        
        #for ks in range(len(personDetected)):
            #cv2.rectangle(frame,(personDetected_box[ks][0],personDetected_box[ks][1]),(personDetected_box[ks][2],personDetected_box[ks][3]),(255,255,0),3)
            #cv2.rectangle(frame,(personDetected[ks][0],personDetected[ks][1]),(personDetected[ks][3],personDetected[ks][2]),(255,255,0),3)
            #cv2.rectangle(frame,(personDetected[ks][1],personDetected[ks][0]),(personDetected[ks][2],personDetected[ks][3]),(255,255,255),3)
            #cv2.rectangle(frame,(personDetected[ks][3],personDetected[ks][2]),(personDetected[ks][1],personDetected[ks][0]),(255,255,255),3)
            #cv2.putText(frame,str(ks),(int(personDetected[ks][0]), int(personDetected[ks][1])+30),0,5e-2 * 200,(255,255,255),2)        
        
        
        features = encoder(frame,personDetected[:,:4])
        #print('a')
        
        ##If error occurs, need to set if statement for zero cases of detection
        detections = [Detection(box,det_class,feature) for box,det_class,feature in zip(personDetected_box,det_classes,features)]
        
        detections_box = np.array([dlc.tlwh for dlc in detections])
        detections_score = np.array([dlc.confidence for dlc in detections])
        detections_indices = preprocessing.non_max_suppression(detections_box,nms_max_overlap,detections_score)
        detections = [detections[i] for i in detections_indices]
        
        #
        tracker.predict()
        tracker.update(detections)
        
        
        cv2.putText(frame,str(frameIdx),(0,30),0,5e-3 * 200,(0,255,0),2) #To check frame number
        
        
        '''
        Support Visualization
        
        '''
        for track in tracker.tracks:
            bbox = [max(0, int(x)) for x in track.to_tlbr()]
            cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),3)
            cv2.putText(frame,str(track.track_id),(bbox[0], bbox[1]+30),0,5e-3 * 200,(0,255,0),2)
        #frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('curvideo',frame)
        cv2.waitKey(delay)
        #if cv2.waitKey()=='':
        #    print('y')
        frameIdx=frameIdx+1
        
capture.release()
cv2.destroyAllWindows()
