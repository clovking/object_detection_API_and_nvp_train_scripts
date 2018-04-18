# 修改tfrecord文件：需修改ssd_mobilenet_v1_coco.config中input_path, label_map_path路径和
#					infer_detections.sh中input_tfrecord_paths和
# 					create_evaluation_pbtxt.sh中label_map_path
# 修改迭代步数:	ssd_mobilenet_v1_coco.config num_steps
# 修改IOU参数：	修改./object_detection/utils/object_detection_evaluation.py文件中Line301行 matching_iou_threshold值即可
# 修改metrics标准：修改脚本create_evaluation_pbtxt.sh中metrics即可

# 迭代步数 230w
num_steps=20000
# 'general' / 'RMB' / 'dollar' / 'money' # money: 人民币 美金 台币 日币
which_model='general'
# root path
root_path='\/home\/quanwang\/work\/object_detection_nvp_train\/'
root_path_linux='/home/quanwang/work/object_detection_nvp_train/'
# ssd_mobilenet_v1 / faster_rcnn_inception_v2
fine_tune_model='ssd_mobilenet_v1' 
# record path
record_path='\/mnt\/extdisk1\/datasets\/tfrecord\/quanwang\/1\/'
record_path_linux='/mnt/extdisk1/datasets/tfrecord/quanwang/1/'
# dataset path
dataset_path='/mnt/extdisk1/datasets/datasetNVP/'

# get the script path
scripts_path=$(cd "$(dirname $0)"; pwd) #该行并不影响根目录

# 修改ssd_mobilenet_v1_nvp.config中迭代步数
python modify_iteration_steps.py ${num_steps} ${which_model} ${root_path} ${fine_tune_model} ${record_path}
echo "modify iteration steps and which model over!!!"

# 显示开始时间
date

# 创建output下文件夹
. ./mkdir_folder.sh ${root_path_linux}

# prepare tfrecord
# "google","nvp","voc","imagenet","money-wil"文件夹下的图像数据生成对应的trainval.txt
python convert_traintxt.py ${dataset_path} ${root_path_linux}
echo "generate trainval.txt over!!!"

# 将trainval.txt 按 generate_filter.txt 过滤，生成 trainval_filter.txt 
python filter_class_exclusive.py ${which_model} ${dataset_path} ${root_path_linux} 
echo "generate trainval_filter.txt over!!!"

# 生成 nvp_train.record , nvp_val.record, python 2
. ./nvp_to_tfrecord.sh ${which_model} ${dataset_path} ${root_path_linux} ${record_path_linux}
echo "generate nvp_train.record and nvp_val.record over!!!"

# train models, python3, need one GPU to train
. ./train_models.sh ${root_path_linux} 
echo "train models over!!!"

# frozen inference graph, python3, need GPU to generate graph
. ./frozen_inference_graph.sh ${num_steps} ${root_path_linux}
echo "frozen inference graph over!!!"

# Inferring detections, generate nvp_val_detections.record, python2
. ./infer_detections.sh ${num_steps} ${root_path_linux} ${record_path_linux}
echo "generate nvp_val_detections.record over!!!"

# evaluate models, python2
. ./evaluate_models.sh ${root_path_linux}
echo "evaluate models over!!!"

# export object detections box
#. ./infer_test.sh ${num_steps}

# 显示结束时间
date

