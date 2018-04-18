import sys
sys.path.append('..')
import os
import time
import tensorflow as tf
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# 需添加环境路径，否则 no utils module 
sys.path.insert(0, '/home/quanwang/work/tensorflow/models/research/object_detection')
from utils import label_map_util
from utils import visualization_utils as vis_util

num_steps = sys.argv[1]

PATH_TO_CKPT = '/home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/config/auto_config/output/frozen_inference_graph/frozen_inference_graph_%s/frozen_inference_graph.pb' % num_steps

print(PATH_TO_CKPT)

PATH_TO_LABELS = '/home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/config/auto_config/input/nvp_label_map.pbtxt'
NUM_CLASSES = 20
IMAGE_SIZE = (18, 12)

ORIGINAL_IMAGE_PATH = '/home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/config/auto_config/input/infer_test_image'
AFTER_IDENTIFICATION_IMAGE_PATH = '/home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/config/auto_config/output/object_detections_box'

def get_all_image_path(path):
    all_img_path = []
    for name in os.listdir(path):
        all_img_path.append(path+'/'+name)
    return all_img_path

all_img_path = get_all_image_path(ORIGINAL_IMAGE_PATH)

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
       for i in range(len(all_img_path)):
            start_time = time.time()
            print(time.ctime())
            image = Image.open(all_img_path[i])
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
            plt.savefig(AFTER_IDENTIFICATION_IMAGE_PATH+'/'+all_img_path[i].split('/')[-1])
            #plt.show()    #plt.show()显示图片，是阻塞的，每次都要先关闭窗口后才会继续运行
