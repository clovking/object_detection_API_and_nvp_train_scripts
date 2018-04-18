#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'hugo'

import os
import sys
import argparse
import random
import logging
try:
    from xml.etree.cElementTree import ElementTree
except:
    from xml.etree.ElementTree import ElementTree

if len(sys.argv) < 3:
	print("Error: Run convert_traintxt.py command line parameter missing")
	sys.exit()

dataset_path = sys.argv[1]
root_path = sys.argv[2]
input_path = "%s%s" % (root_path, "scripts/input_config/")
output_path = "%s%s" % (root_path, "scripts/output/")

FILE_NAME=["google","nvp","imagenet","money-wil"]
TEST_RATIO = 0
logging.basicConfig(
   level = logging.ERROR,
   #level = logging.DEBUG,
   format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s"
)

global cnt_all_nvp
global cnt_all_cand_nvp
global cnt_test_nvp
global cnt_trainval_nvp
cnt_all_nvp = 0
cnt_all_cand_nvp = 0
cnt_test_nvp = 0
cnt_trainval_nvp = 0

global global_trainval_list
global_trainval_list = []

global global_trainval_file

# "google","nvp","voc"
def generate_txt(root, name, dir_path):
    global cnt_all_nvp
    global cnt_all_cand_nvp
    global cnt_test_nvp
    global cnt_trainval_nvp
    
    root_name = "%s/%s" % (root, name)
    img_path = "%s/%s/JPEGImages/" % (root_name, dir_path)
    ann_path = "%s/%s/Annotations/" % (root_name, dir_path)

    cnt_all = 0
    cnt_all_cand = 0
    cnt_test = 0
    cnt_trainval = 0

    for dir_info in os.walk(img_path):
        dir_name, _, fns = dir_info
        if len(fns) > 0 :
           for fn in fns:
               #logging.debug(fn)
               sufix = os.path.splitext(fn)[1][1:]
               if sufix != "JPEG" and sufix != "jpg":
                   continue

               cnt_all += 1
               ann_fn = "%s.xml" % (os.path.splitext(fn)[0])
               logging.debug(os.path.join(ann_path, ann_fn))
               if os.path.exists(os.path.join(ann_path, ann_fn)):
                   content = ("%s/%s/JPEGImages/%s %s/%s/Annotations/%s" 
                               % (name, dir_path, fn, name, dir_path, ann_fn))
                   global_trainval_list.append(content)

    cnt_all_nvp += cnt_all
    cnt_all_cand_nvp += cnt_all_cand
    cnt_trainval_nvp += cnt_trainval
    cnt_test_nvp += cnt_test


# "imagenet","money-wil"
def generate_txt2(root, name):
    global cnt_all_nvp
    global cnt_all_cand_nvp
    global cnt_test_nvp
    global cnt_trainval_nvp

    root_name = "%s/%s" % (root, name)
    img_path = "%s/JPEGImages/" % (root_name)
    ann_path = "%s/Annotations/" % (root_name)

    cnt_all = 0
    cnt_all_cand = 0
    cnt_test = 0
    cnt_trainval = 0

    for dir_info in os.walk(img_path):
        dir_name, _, fns = dir_info
        if len(fns) > 0 :
           for fn in fns:
               #logging.debug(fn)
               sufix = os.path.splitext(fn)[1][1:]
               if sufix != "JPEG" and sufix != "jpg":
                   continue

               cnt_all += 1
               ann_fn = "%s.xml" % (os.path.splitext(fn)[0])
               logging.debug(os.path.join(ann_path, ann_fn))
               if os.path.exists(os.path.join(ann_path, ann_fn)):
                   content = ("%s/JPEGImages/%s %s/Annotations/%s"
                               % (name, fn, name, ann_fn))
                   global_trainval_list.append(content)
    
    cnt_all_nvp += cnt_all
    cnt_all_cand_nvp += cnt_all_cand
    cnt_trainval_nvp += cnt_trainval
    cnt_test_nvp += cnt_test
    
if __name__ == "__main__":
    trainval = "%s/trainval.txt" % (output_path) 
    global global_trainval_file
    global_trainval_file = open(trainval, 'w')

    for name in FILE_NAME:
        if name == "google" or name == "nvp" or name == "voc":
            dir_items = os.listdir("%s%s" % (dataset_path, name))
            logging.debug(dir_items)
            
            for item in dir_items:
                if os.path.isdir(os.path.join(dataset_path, name, item)):
                    if not (name == "voc" and item == "VOC2012-test"):
                        generate_txt(dataset_path, name, item)
            print("All nvp datasets, total data %d, nvp data %d, trainval data %d, test data %d" 
                    % (cnt_all_nvp, cnt_all_cand_nvp, cnt_trainval_nvp, cnt_test_nvp))
        elif name == "imagenet" or name == "money-wil":
            generate_txt2(dataset_path, name)
            print("All nvp datasets, total data %d, nvp data %d, trainval data %d, test data %d"
                    % (cnt_all_nvp, cnt_all_cand_nvp, cnt_trainval_nvp, cnt_test_nvp))

    random.shuffle(global_trainval_list)
    for each in global_trainval_list:
        global_trainval_file.write(each)
        global_trainval_file.write('\n')
    global_trainval_file.close()
