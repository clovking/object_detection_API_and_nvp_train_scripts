#!/usr/bin/env python

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert nvp dataset to TFRecord for object_detection.

Example usage:
    ./create_nvp_tf_record --data_dir=/mnt/extdisk1/datasets/datasetNVP/ \
        --list_txt=/home/hugo/work/models/nvp/trainval.txt \
        --output_dir=/home/hugo/work/models/nvp/output \
        --label_map_path=data/nvp_label_map.pbtxt
"""

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

TRAIN_RECORD_NAME = 'nvp_train.record'
VAL_RECORD_NAME = 'nvp_val.record'

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to nvp dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('list_txt', '', 'txt file for trainval.txt')
flags.DEFINE_string('label_map_path', 'data/nvp_label_map.pbtxt', 'Path to label map proto')
FLAGS = flags.FLAGS

def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """
  match = re.match(r'([A-Za-z_]+)(_[0-9]+\.jpg)', file_name, re.I)
  return match.groups()[0]


def dict_to_tf_example(data,
                       label_map_dict,
                       img_path,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  try:
    image = PIL.Image.open(encoded_jpg_io)
    image.load()
  except:
    print("Image corrupt %s" % img_path)
    return None
  if image.format != 'JPEG':
    print("Image not JPEG: %s" % img_path)
    return None
    #raise ValueError('Image format not JPEG')
  '''
  # decode jpeg begin
  try:
    image_contents = tf.read_file(img_path)
    image_contents_temp = tf.image.decode_jpeg(image_contents, channels = 3)
    with tf.Session() as sess:
      sess.run(image_contents_temp)
  except:
    print("Image can not decode: %s" % img_path)
    return None
  # decode jpeg end
  '''

  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue

    difficult_obj.append(int(difficult))

    xmin_v = float(obj['bndbox']['xmin']) / width
    ymin_v = float(obj['bndbox']['ymin']) / height
    xmax_v = float(obj['bndbox']['xmax']) / width
    ymax_v = float(obj['bndbox']['ymax']) / height

    if xmin_v < 0 or ymin_v < 0 or xmax_v > 1 or ymax_v > 1 or xmin_v > xmax_v or ymin_v > ymax_v:
        logging.warning("bad bnd parameters for %s: xmin %f ymin %f xmax %f ymax %f !!" , img_path, xmin_v, ymin_v, xmax_v, ymax_v)
        continue
    xmin.append(xmin_v)
    ymin.append(ymin_v)
    xmax.append(xmax_v)
    ymax.append(ymax_v)

    class_name = obj['name']
    classes_text.append(class_name.encode('utf8'))
    classes.append(label_map_dict[class_name])
    truncated.append(int(obj['truncated']))
    poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }))
  return example


def create_tf_record(output_filename,
                     label_map_dict,
                     examples):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    label_map_dict: The label map dictionary.
    examples: Examples to parse and save to tf record.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(examples):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(examples))
    img_path = os.path.join(FLAGS.data_dir, example)
    path = img_path.replace('JPEGImages','Annotations')
    path = path.replace('jpg','xml')
    path = path.replace('jpeg','xml')
    path = path.replace('JPG','xml')
    path = path.replace('JPEG','xml')
    path = path.replace('png','xml')

    if not os.path.exists(path):
      logging.warning('Could not find %s, ignoring example.', path)
      continue
    if not os.path.exists(img_path):
      logging.warning('Could not find %s, ignoring example.', img_path)
      continue
    with tf.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
    tf_example = dict_to_tf_example(data, label_map_dict, img_path)
    if tf_example:
        writer.write(tf_example.SerializeToString())

  writer.close()


# TODO: Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from nvp dataset.')
  examples_list = dataset_util.read_examples_list(FLAGS.list_txt)
  
  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.9 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, TRAIN_RECORD_NAME)
  val_output_path = os.path.join(FLAGS.output_dir, VAL_RECORD_NAME)
  create_tf_record(train_output_path, label_map_dict, 
                    train_examples)
  create_tf_record(val_output_path, label_map_dict, 
                    val_examples)

if __name__ == '__main__':
  tf.app.run()
