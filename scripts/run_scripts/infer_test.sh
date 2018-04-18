if [ ! ${1} ]; then           #环境变量判断
    echo error! please enter number
    exit
fi

rm -rf /home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/config/auto_config/output/object_detections_box/
mkdir /home/quanwang/work/tensorflow/COCO_Model_Net/AngelEye/config/auto_config/output/object_detections_box

# python3 environment
source activate tensorflow-gpu

CUDA_VISIBLE_DEVICES="1" \
python infer_test.py ${1}
