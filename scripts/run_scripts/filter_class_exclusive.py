#!/usr/bin/env python

# -*- coding: utf-8 -*-
__author__ = 'hugo'

import os
import sys
import argparse
import random
import logging
import time
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET

if len(sys.argv) < 4:
    print("Error: Filter_class_exclusive.py, please enter model, dataset path and root path")
    sys.exit()

which_model = sys.argv[1]
dataset_path = sys.argv[2]
root_path = sys.argv[3]
input_path = "%s%s" % (root_path, "scripts/input_config/")
output_path = "%s%s" % (root_path, "scripts/output/")

TRAIN_SET = "%s%s" % (output_path, "trainval.txt")
TEST_SET = "%s%s" % (output_path, "test.txt")
TRAIN_FILTER = "%s%s" % (output_path, "trainval_filter.txt")
generate_filter_txt = "%s%s%s" % ("generate_filter_", which_model, ".txt")
FILTER_TXT = "%s%s" % (input_path, generate_filter_txt)

logging.basicConfig(
   level = logging.ERROR,
   #level = logging.DEBUG,
   format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)

def init_words(file):
    words = []
    with open(file) as f:
        contents = [x.strip() for x in f.readlines()]
        for content in contents:
            words.append(content)
    return words

def process(txt):
    obj_cnt = 0
    file_cnt = 0
    label_cnt = {}

    all_items = []
    rm_items = []

    words = init_words(FILTER_TXT)
    print(words)

    class_num = dict(zip(words,[0 for x in range(len(words))]))

    with open(txt, 'r') as f:
        for x in f:
            #logging.debug(x)
            file_cnt += 1    
            fn, ann_fn = x.split(' ')
            ann_fn = ann_fn.strip()
            ann_fn = os.path.join(dataset_path, ann_fn)
            valid = 0
            if ".xml" in ann_fn:
                try:
                    tree = ET.parse(ann_fn)
                    root = tree.getroot()
                    objects = root.findall('object')
                    myobject_tmp = []
                    for myobject in objects:
                        objname = myobject.find('name')

                        for word in words:
                            if (objname.text == word): #图像中类别为目标类之一，则继续判断图片中下一个类别
                                valid = 1
                                class_num[word] += 1
                                myobject_tmp.append(word)
                                break
                            else:
                                valid = 0
                        
                        if valid == 0: #必须保证所有类别均为目标类之一
                           #避免某图片中前面类别通过，后有一类别不通过，而前面已经添加类别数情况
                           for m in myobject_tmp:
                               class_num[m] -= 1
                           #print("####################################")
                           break
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except:
                    logging.error("can't parse xml %s" % ann_fn)

            if valid == 1:
                #logging.debug("%s is added" % fn)
                yield x
    class_num_path = os.path.join(output_path, "class_num.txt") 
    with open(class_num_path, 'w') as class_num_file:
        class_num_file.write(str(class_num))

# This function is used find specific class from dataset
# example: filter_class.py stairs
# output: "/mnt/extdisk1/datasets/datasetNVP/ssd-class/trainval_stairs.txt
if __name__ == "__main__":
    with open(TRAIN_FILTER, 'w') as filter_file:
        for x in process(TRAIN_SET):
            filter_file.write(x)

