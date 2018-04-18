import sys
sys.path.append('..')
import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from utils import label_map_util
from utils import visualization_utils as vis_util

PATH_TEST_IMAGE = sys.argv[1]
PATH_TEST_IMAGE_NAME = sys.argv[1].split('/')[-1]

PATH_TO_CKPT = '/home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/frozen_inference_graph/frozen_inference_graph_50000/frozen_inference_graph.pb'
#PATH_TO_CKPT = '/home/quanwang/work/tensorflow/COCO_Model_Net/VOCtrainval2012/frozen_inference_graph_12789_ok/frozen_inference_graph.pb'
#PATH_TO_CKPT = '/home/quanwang/work/tensorflow/COCO_Model_Net/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
PATH_TO_LABELS = '/home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/config/nvp_label_map.pbtxt'
#PATH_TO_LABELS = '/home/quanwang/work/tensorflow/models/research/object_detection/data/mscoco_label_map.pbtxt'

AFTER_IDENTIFICATION_PATH = '/home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/infer_test_image/after_identification_image'

NUM_CLASSES = 20
IMAGE_SIZE = (18, 12)

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with detection_graph.as_default():
    with tf.Session(graph=detection_graph, config=config) as sess:
        start_time = time.time()
        print(time.ctime())
        image = Image.open(PATH_TEST_IMAGE)
        image_np = np.array(image).astype(np.uint8)
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        print('{} elapsed time: {:.3f}s'.format(time.ctime(), time.time() - start_time))
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
            category_index, use_normalized_coordinates=True, line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)
        plt.savefig(AFTER_IDENTIFICATION_PATH+'/'+PATH_TEST_IMAGE_NAME)
        plt.show()
