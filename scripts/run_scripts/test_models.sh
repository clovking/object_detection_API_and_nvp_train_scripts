# =======================================================================================
# test image.
# announcements: the script must be running by vnc view, which can show us the result , not putty
# input: PATH_TO_CKPT and PATH_TO_LABELS parameter, in infer_test.py, 
#        should be modify when you want to change the other frozen inference graph.
#        test image is saved in infer_test_image folder
# =======================================================================================
python /home/quanwang/work/tensorflow/models/research/object_detection/infer_test.py \
    /home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/infer_test_image/1.jpg
