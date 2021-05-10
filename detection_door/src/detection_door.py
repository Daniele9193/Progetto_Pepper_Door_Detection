#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ros node for door detection with Pepper Robot.
"""

import colorsys
import os, sys, argparse
import subprocess
import webbrowser
import time
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
#from tensorflow_model_optimization.sparsity import keras as sparsity
from PIL import Image as IMAGE
from std_msgs.msg import String

#from yolo5.model import get_yolo5_model, get_yolo5_inference_model
#from yolo5.postprocess_np import yolo5_postprocess_np
from yolo3.model import get_yolo3_model, get_yolo3_inference_model
from yolo3.postprocess_np import yolo3_postprocess_np
#from yolo2.model import get_yolo2_model, get_yolo2_inference_model
#from yolo2.postprocess_np import yolo2_postprocess_np
from common.data_utils import preprocess_image
from common.utils import get_classes, get_anchors, get_colors, draw_boxes, optimize_tf_gpu
from tensorflow.keras.utils import multi_gpu_model


import rospy
from cv_bridge import CvBridge, CvBridgeError
#from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
import cv2

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#optimize_tf_gpu(tf, K)

#tf.enable_eager_execution()

default_config = {
        "model_type": 'yolo3_efficientnet',
        "weights_path": "model.h5",
        "pruning_model": False,
        "anchors_path": os.path.join('configs', 'yolo3_anchors.txt'),
        "classes_path": "class.txt",
        "score" : 0.1,
        "iou" : 0.4,
        "model_image_size" : (416, 416),
        "elim_grid_sense": False,
        "gpu_num" : 1,
    }


class YOLO_np(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO_np, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.yolo_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3


        yolo_model, _ = get_yolo3_model(self.model_type, num_feature_layers, num_anchors, num_classes, input_shape=self.model_image_size + (3,), model_pruning=self.pruning_model)

        yolo_model.load_weights(weights_path) # make sure model, anchors and classes match
            #if self.pruning_model:
            #    yolo_model = sparsity.strip_pruning(yolo_model)
        yolo_model.summary()
        
        print('{} model, anchors, and classes loaded.'.format(weights_path))

        return yolo_model


    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)
        #origin image shape, in (height, width) format
        #image = Image.fromarray(image)
        image_shape = tuple(reversed(image.size))

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        end = time.time()
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        print("Fine detect image 1")
        return IMAGE.fromarray(image_array), out_boxes, out_classnames, out_scores, end , start


    def predict(self, image_data, image_shape):
        num_anchors = len(self.anchors)
        if self.model_type.startswith('scaled_yolo4_') or self.model_type.startswith('yolo5_'):
            # Scaled-YOLOv4 & YOLOv5 entrance, enable "elim_grid_sense" by default
            out_boxes, out_classes, out_scores = yolo5_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100, confidence=self.score, iou_threshold=self.iou, elim_grid_sense=True)
        elif self.model_type.startswith('yolo3_') or self.model_type.startswith('yolo4_') or \
             self.model_type.startswith('tiny_yolo3_') or self.model_type.startswith('tiny_yolo4_'):
            # YOLOv3 & v4 entrance
            out_boxes, out_classes, out_scores = yolo3_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100, confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)
        elif self.model_type.startswith('yolo2_') or self.model_type.startswith('tiny_yolo2_'):
            # YOLOv2 entrance
            out_boxes, out_classes, out_scores = yolo2_postprocess_np(self.yolo_model.predict(image_data), image_shape, self.anchors, len(self.class_names), self.model_image_size, max_boxes=100, confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)
        else:
            raise ValueError('Unsupported model type')

        return out_boxes, out_classes, out_scores


    def dump_model_file(self, output_model_file):
        self.yolo_model.save(output_model_file)



class YOLO(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(YOLO, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.colors = get_colors(self.class_names)
        K.set_learning_phase(0)
        self.inference_model = self._generate_model()

    def _generate_model(self):
        '''to generate the bounding boxes'''
        weights_path = os.path.expanduser(self.weights_path)
        assert weights_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        #YOLOv3 model has 9 anchors and 3 feature layers but
        #Tiny YOLOv3 model has 6 anchors and 2 feature layers,
        #so we can calculate feature layers number to get model type
        num_feature_layers = num_anchors//3

        if self.model_type.startswith('scaled_yolo4_') or self.model_type.startswith('yolo5_'):
            # Scaled-YOLOv4 & YOLOv5 entrance, enable "elim_grid_sense" by default
            inference_model = get_yolo5_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,), confidence=self.score, iou_threshold=self.iou, elim_grid_sense=True)
        elif self.model_type.startswith('yolo3_') or self.model_type.startswith('yolo4_') or \
             self.model_type.startswith('tiny_yolo3_') or self.model_type.startswith('tiny_yolo4_'):
            # YOLOv3 & v4 entrance
            inference_model = get_yolo3_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,), confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)
        elif self.model_type.startswith('yolo2_') or self.model_type.startswith('tiny_yolo2_'):
            # YOLOv2 entrance
            inference_model = get_yolo2_inference_model(self.model_type, self.anchors, num_classes, weights_path=weights_path, input_shape=self.model_image_size + (3,), confidence=self.score, iou_threshold=self.iou, elim_grid_sense=self.elim_grid_sense)
        else:
            raise ValueError('Unsupported model type')

        inference_model.summary()
        return inference_model

    def predict(self, image_data, image_shape):
        out_boxes, out_scores, out_classes = self.inference_model.predict([image_data, image_shape])

        out_boxes = out_boxes[0]
        out_scores = out_scores[0]
        out_classes = out_classes[0]

        out_boxes = out_boxes.astype(np.int32)
        out_classes = out_classes.astype(np.int32)
        return out_boxes, out_classes, out_scores

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'

        image_data = preprocess_image(image, self.model_image_size)

        # prepare origin image shape, (height, width) format
        image_shape = np.array([image.size[1], image.size[0]])
        image_shape = np.expand_dims(image_shape, 0)

        start = time.time()
        out_boxes, out_classes, out_scores = self.predict(image_data, image_shape)
        end = time.time()
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        print("Inference time: {:.8f}s".format(end - start))

        #draw result on input image
        image_array = np.array(image, dtype='uint8')
        image_array = draw_boxes(image_array, out_boxes, out_classes, out_scores, self.class_names, self.colors)

        out_classnames = [self.class_names[c] for c in out_classes]
        print("Fine detect image 2")
        return Image.fromarray(image_array), out_boxes, out_classnames, out_scores

    def dump_model_file(self, output_model_file):
        self.inference_model.save(output_model_file)

    def dump_saved_model(self, saved_model_path):
        model = self.inference_model
        os.makedirs(saved_model_path, exist_ok=True)

        tf.keras.experimental.export_saved_model(model, saved_model_path)
        print('export inference model to %s' % str(saved_model_path))





def detect_img(yolo, image, pub):
    r_image, boxes, classes, confidences, end, start = yolo.detect_image(image)
    risultati = "" 
    duration = end-start
    for i in range(0, len(boxes)):
        if(classes[i] in 'door'):
            risultato = "Porta trovata con confidenza: "+str(confidences[i])+" con bounding box con coordinate: x="+str(boxes[i][0])+" y="+str(boxes[i][1])+" larghezza="+str(boxes[i][2])+" altezza="+str(boxes[i][3])+" e il tempo di inferenza e' stato di "+str(duration)+"     "
            risultati += risultato
        if(classes[i] in 'handle'):
            risultato = "Maniglia trovata con confidenza: "+str(confidences[i])+" con bounding box con coordinate: x="+str(boxes[i][0])+" y="+str(boxes[i][1])+" larghezza="+str(boxes[i][2])+" altezza="+str(boxes[i][3])+" e il tempo di inferenza e' stato di "+str(duration)+"      "
            risultati += risultato
    print(risultati)
    msg = String()
    msg.data = risultati
    pub.publish(msg)
    try:
        r_image.save("risultato.jpg")
        webbrowser.open('risultato.jpg')
        print("Salvataggio riuscito")
    except:
        print('Salvataggio non riuscito')
    



def convert_depth_image(ros_image):
    cv_bridge = CvBridge()
    #depth_image = cv_bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
    depth_image = cv_bridge.compressed_imgmsg_to_cv2(ros_image, 'passthrough')
    images_pepper.append(depth_image)
    cv2.imwrite("depth_img.png", depth_image)


def main(yolo, pub):
    print("Image detection mode")
    image_rob = IMAGE.fromarray(images_pepper[-1])
    detect_img(yolo, image_rob, pub)


def pixel2depth(pub):
    rospy.init_node('detection_door',anonymous=True)
    #rospy.Subscriber("/pepper/camera/front/image_raw", Image, callback=convert_depth_image, queue_size=1)
    rospy.Subscriber("/pepper_robot/camera/front/image_raw/compressed", CompressedImage, callback=convert_depth_image, queue_size=10)
    time.sleep(3)
    while not rospy.is_shutdown():
        if len(images_pepper)>0:
            main(yolo=yolo3, pub=pub)
            




if __name__ == '__main__':
    path = raw_input('Inserisci il path della directory in cui è contenuto il nodo ros: ')
    path2 = ""
    path2 = path + "/detection_door/src"
    print(path2)
    os.chdir(path2)
    print("Python version")
    print (sys.version)
    print("Version info.")
    print (sys.version_info)
    print(tf.version.VERSION)
    yolo3 = YOLO_np()
    pub = rospy.Publisher("/results", String, queue_size=10)
    global images_pepper
    images_pepper = []
    pixel2depth(pub=pub)
